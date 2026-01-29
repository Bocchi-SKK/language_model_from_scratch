import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def distributed_demo(rank, world_size, data_size):
    setup(rank, world_size)
    num_elements = data_size // 4
    data = torch.randint(0, 10, (num_elements,))
    # print(f"rank {rank} data (before all-reduce): {data}")
    dist.all_reduce(data, async_op=False)
    # print(f"rank {rank} data (after all-reduce): {data}")

if __name__ == "__main__":
    world_size = 4
    mp.spawn(fn=distributed_demo, args=(world_size,1 * (2**20)), nprocs=world_size, join=True)