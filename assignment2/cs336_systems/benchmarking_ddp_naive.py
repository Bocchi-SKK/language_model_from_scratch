import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from einops import rearrange

from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.model import transformer_lm

from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

WORLD_SIZE = 2
BATCH_SIZE = 4
VOCAB_SIZE = 128
CONTEXT_LENGTH = 64
D_MODEL = 128
NUM_LAYERS = 2
NUM_HEADS = 4
D_FF = 256
ROPE_THETA = 10000
RANDOM_SEED = 42
TOTAL_STEPS = 200
WARMUP_STEPS = 5
DEVICE = 'cpu'
if DEVICE == 'cuda' and (WORLD_SIZE > torch.cuda.device_count()):
    WORLD_SIZE = torch.cuda.device_count()

def setup_gloo(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(
        "gloo", 
        rank=rank, 
        world_size=world_size)

def setup_nccl(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    # For NCCL, make sure each rank uses a distinct CUDA device.
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
    )

def ddp_worker(rank, world_size, split_input_tensor_batchs, split_target_tensor_batchs, total_steps=TOTAL_STEPS, device=DEVICE):
    if device == 'cpu':
        setup_gloo(rank, world_size)
        device_obj = torch.device('cpu')
    elif device == 'cuda':
        setup_nccl(rank, world_size)
        device_obj = torch.device(f'cuda:{rank}')
    else:
        raise ValueError(f"Unsupported device: {device}")
    torch.manual_seed(RANDOM_SEED)
    if device == 'cuda':
        torch.cuda.manual_seed(RANDOM_SEED)
    model = transformer_lm(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        rope_theta=ROPE_THETA).to(device_obj)
    optimizer = AdamW(model.parameters())
    loss_fn = cross_entropy
    
    local_input = split_input_tensor_batchs[rank].to(device_obj)
    local_target = split_target_tensor_batchs[rank].to(device_obj)
    local_target = rearrange(local_target, 'batch_size context_length -> (batch_size context_length)')

    step_times = []
    comm_times = []

    for step in range(total_steps):
        # Use perf_counter for higher-resolution wall-clock timing.
        # For CUDA, synchronize so timing includes async kernels.
        if device == 'cuda':
            torch.cuda.synchronize()
        step_start = time.perf_counter()

        local_output = model(local_input)
        local_output = rearrange(local_output, 'batch_size context_length vocab_size -> (batch_size context_length) vocab_size')
        local_loss = loss_fn(inputs=local_output, targets=local_target)
        optimizer.zero_grad()
        local_loss.backward()

        if device == 'cuda':
            torch.cuda.synchronize()
        comm_start = time.perf_counter()

        # All-reduce and average gradients
        if world_size  > 1:

            # # Navie method to call communication for multiple times
            # for param in model.parameters():
            #     if param.grad is not None:
            #         dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

            # Smater method to reduce the commnucation calling times
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            flat_grad = _flatten_dense_tensors(grads)
            dist.all_reduce(flat_grad, op=dist.ReduceOp.AVG)
            grads_after = _unflatten_dense_tensors(flat_grad, grads)
            for g, p in zip(grads_after, [p for p in model.parameters() if p.grad is not None]):
                p.grad.copy_(g)


        if device == 'cuda':
            torch.cuda.synchronize()
        comm_end = time.perf_counter()

        optimizer.step()

        if device == 'cuda':
            torch.cuda.synchronize()
        step_end = time.perf_counter()

        step_time = step_end - step_start
        comm_time = comm_end - comm_start
        if comm_time < 1e-4:
            comm_time=0.0

        # Drop warmup steps from statistics (startup/caching effects).
        if step >= WARMUP_STEPS:
            step_times.append(step_time)
            comm_times.append(comm_time)

        if ((step + 1) % 100 == 0 and rank == 0):
            ratio = (comm_time / step_time) if step_time > 0 else 0.0
            print(
                f"Step {step+1}: loss={local_loss.item():.4f}, total_time={step_time:.4f}s, "
                f"comm_time={comm_time:.4f}s, comm_ratio={ratio:.2%}"
            )

    # Summarize across ranks.
    # For overall step time in data-parallel, the slowest rank is the bottleneck,
    # so we report both mean-across-ranks and max-across-ranks.
    avg_step_local = sum(step_times) / max(len(step_times), 1)
    avg_comm_local = sum(comm_times) / max(len(comm_times), 1)

    stats_device = device_obj if device == 'cuda' else torch.device('cpu')
    avg_step_t = torch.tensor([avg_step_local], device=stats_device, dtype=torch.float64)
    avg_comm_t = torch.tensor([avg_comm_local], device=stats_device, dtype=torch.float64)

    avg_step_sum = avg_step_t.clone()
    avg_comm_sum = avg_comm_t.clone()
    dist.all_reduce(avg_step_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(avg_comm_sum, op=dist.ReduceOp.SUM)

    avg_step_mean = (avg_step_sum / world_size).item()
    avg_comm_mean = (avg_comm_sum / world_size).item()

    avg_step_max = avg_step_t.clone()
    avg_comm_max = avg_comm_t.clone()
    dist.all_reduce(avg_step_max, op=dist.ReduceOp.MAX)
    dist.all_reduce(avg_comm_max, op=dist.ReduceOp.MAX)

    if rank == 0:
        ratio_mean = (avg_comm_mean / avg_step_mean) if avg_step_mean > 0 else 0.0
        ratio_max = (avg_comm_max.item() / avg_step_max.item()) if avg_step_max.item() > 0 else 0.0
        print(
            f"\n[Rank 0] After warmup (steps {WARMUP_STEPS+1}..{total_steps}):\n"
            f"  mean step time (across ranks): {avg_step_mean:.6f}s\n"
            f"  mean comm time (across ranks): {avg_comm_mean:.6f}s (ratio {ratio_mean:.2%})\n"
            f"  max  step time (bottleneck rank): {avg_step_max.item():.6f}s\n"
            f"  max  comm time (bottleneck rank): {avg_comm_max.item():.6f}s (ratio {ratio_max:.2%})"
        )

    dist.destroy_process_group()

if __name__ == "__main__":
    random_input_tensor = torch.randint(low=0, high=VOCAB_SIZE,size=(BATCH_SIZE, CONTEXT_LENGTH))
    random_target_tensor = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, CONTEXT_LENGTH))

    split_input_tensor_batchs = rearrange(random_input_tensor, '(num_device batch_size) context_length -> num_device batch_size context_length', num_device=WORLD_SIZE)
    split_target_tensor_batchs = rearrange(random_target_tensor, '(num_device batch_size) context_length -> num_device batch_size context_length', num_device=WORLD_SIZE)
    
    mp.spawn(ddp_worker, args=(WORLD_SIZE, split_input_tensor_batchs, split_target_tensor_batchs, TOTAL_STEPS, DEVICE), nprocs=WORLD_SIZE, join=True)