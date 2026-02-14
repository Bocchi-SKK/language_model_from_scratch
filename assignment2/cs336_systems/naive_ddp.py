import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.model import transformer_lm
from einops import rearrange

WORLD_SIZE = 4
BATCH_SIZE = 8
VOCAB_SIZE = 128
CONTEXT_LENGTH = 64
D_MODEL=16
NUM_LAYERS=2
NUM_HEADS=4
D_FF=32
ROPE_THETA=10000
RANDOM_SEED = 42
TOTAL_STEPS = 1
DEVICE = 'cuda'
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

def single_training(input_tensor, target_tensor, TOTAL_STEPS=TOTAL_STEPS, device=DEVICE):
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
        rope_theta=ROPE_THETA).to(device)
    
    loss_fn = cross_entropy
    optimizer = AdamW(params=model.parameters())
    # To match the naive DDP math exactly, we split the batch
    # into WORLD_SIZE shards and accumulate gradients over shards
    # before taking a single optimizer step.
    split_input = rearrange(
        input_tensor,
        '(num_device batch_size) context_length -> num_device batch_size context_length',
        num_device=WORLD_SIZE,
    ).to(device)
    split_target = rearrange(
        target_tensor,
        '(num_device batch_size) context_length -> num_device batch_size context_length',
        num_device=WORLD_SIZE,
    ).to(device)

    for step in range(TOTAL_STEPS):
        optimizer.zero_grad()
        total_loss = 0.0
        for shard_input, shard_target in zip(split_input, split_target):
            shard_target_flat = rearrange(
                shard_target,
                'batch_size context_length -> (batch_size context_length)',
            )
            output = model(shard_input)
            output_flat = rearrange(
                output,
                'batch_size context_length vocab_size -> (batch_size context_length) vocab_size',
            )
            loss = loss_fn(inputs=output_flat, targets=shard_target_flat)
            loss.backward()
            total_loss += loss.item()

        # Divide accumulated gradients so they correspond to the
        # global mean loss, matching the DDP gradient exactly.
        for param in model.parameters():
            if param.grad is not None:
                param.grad /= WORLD_SIZE

        optimizer.step()
        if (step+1) % 100 == 0:
            print(total_loss)

    torch.save(model.state_dict(), f"temp/model_single.pt")

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

    for step in range(total_steps):
        local_output = model(local_input)
        local_output = rearrange(local_output, 'batch_size context_length vocab_size -> (batch_size context_length) vocab_size')
        local_loss = loss_fn(inputs=local_output, targets=local_target)
        optimizer.zero_grad()
        local_loss.backward()

        # All-reduce and average gradients
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
                # param.grad /= world_size

        optimizer.step()
        if ((step+1) % 100==0 and rank == 0):
            print(local_loss)

    # save the model for each rank
    torch.save(model.state_dict(), f"temp/model_rank{rank}.pt")
    dist.destroy_process_group()  # <-- add this line

if __name__ == "__main__":
    random_input_tensor = torch.randint(low=0, high=VOCAB_SIZE,size=(BATCH_SIZE, CONTEXT_LENGTH))
    random_target_tensor = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, CONTEXT_LENGTH))

    split_input_tensor_batchs = rearrange(random_input_tensor, '(num_device batch_size) context_length -> num_device batch_size context_length', num_device=WORLD_SIZE)
    split_target_tensor_batchs = rearrange(random_target_tensor, '(num_device batch_size) context_length -> num_device batch_size context_length', num_device=WORLD_SIZE)

    single_training(input_tensor=random_input_tensor, target_tensor=random_target_tensor,TOTAL_STEPS=TOTAL_STEPS, device=DEVICE)
    
    mp.spawn(ddp_worker, args=(WORLD_SIZE, split_input_tensor_batchs, split_target_tensor_batchs, TOTAL_STEPS, DEVICE), nprocs=WORLD_SIZE, join=True)
    state_dicts = [torch.load(f"temp/model_rank{rank}.pt") for rank in range(WORLD_SIZE)]
    all_equal = True
    for k in state_dicts[0]:
        for i in range(1, WORLD_SIZE):
            if not torch.allclose(state_dicts[0][k], state_dicts[i][k], rtol=1e-4, atol=1e-5):
                diff = (state_dicts[0][k] - state_dicts[i][k]).abs().max().item()
                print(f"Mismatch found in parameter across ranks: {k}, max abs diff = {diff}")
                all_equal = False
    print(f"All model weights equal across ranks (within tolerance): {all_equal}")

    # Compare single-process model with DDP model (rank 0)
    single_state_dict = torch.load("temp/model_single.pt")
    single_vs_ddp_equal = True
    for k in single_state_dict:
        if not torch.allclose(single_state_dict[k], state_dicts[0][k], rtol=1e-4, atol=1e-5):
            diff = (single_state_dict[k] - state_dicts[0][k]).abs().max().item()
            print(f"Mismatch between single-process and DDP in parameter: {k}, max abs diff = {diff}")
            single_vs_ddp_equal = False
    print(f"Single-process model and DDP model (rank 0) weights equal (within tolerance): {single_vs_ddp_equal}")