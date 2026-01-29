import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time

def setup_gloo(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def setup_nccl(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def benchmark_all_reduce_gloo(rank, world_size, data_size):
    setup_gloo(rank, world_size)
    num_elements = data_size // 4
    data = torch.randn(num_elements, dtype=torch.float32)
    dist.barrier()
    start = time.time()
    dist.all_reduce(data, async_op=False)
    dist.barrier()
    end = time.time()
    elapsed = end - start

    # Gather all elapsed times to all processes
    all_times = [None for _ in range(world_size)]
    dist.all_gather_object(all_times, elapsed)
    if rank == 0:
        avg_time = sum(all_times) / len(all_times)
        print(f"Average all-reduce time for {data_size / (2**20):.1f} MB with {world_size} processes: {avg_time:.4f} seconds")
    dist.destroy_process_group()

def benchmark_all_reduce_nccl(rank, world_size, data_size):
    setup_nccl(rank, world_size)
    torch.cuda.set_device(rank)
    num_elements = data_size // 4
    data = torch.randn(num_elements, dtype=torch.float32, device='cuda')
    dist.barrier()
    start = time.time()
    dist.all_reduce(data, async_op=False)
    dist.barrier()
    end = time.time()
    elapsed = end - start

    all_times = [None for _ in range(world_size)]
    dist.all_gather_object(all_times, elapsed)
    if rank == 0:
        avg_time = sum(all_times) / len(all_times)
        print(f"Average all-reduce time for {data_size / (2**20):.1f} MB with {world_size} processes: {avg_time:.4f} seconds")
    dist.destroy_process_group()

if __name__ == "__main__":
    # world_sizes = [2, 4, 6]
    world_sizes = [1]
    sizes_mb = [1, 10, 100, 1024]
    for world_size in world_sizes:
        for size_mb in sizes_mb:
            print(f"\nBenchmarking all-reduce for {size_mb} MB with {world_size} processes")
            mp.spawn(benchmark_all_reduce_nccl, args=(world_size, size_mb*(2**20)), nprocs=world_size, join=True)