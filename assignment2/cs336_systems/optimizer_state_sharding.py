import torch
from collections.abc import Callable
from typing import Type, Any, Optional
from torch.optim import Optimizer
import torch.distributed as dist

def shard_parameters(params, rank, world_size):
    params = list(params)
    return [p for i, p in enumerate(params) if i % world_size == rank]

class shardedOptimizer(Optimizer):
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        self.all_params = list(params)  # Convert to list so it's not consumed
        self.rank = kwargs.pop("rank", 0)
        self.world_size = kwargs.pop("world_size", 1)

        # sharded
        sharded = shard_parameters(self.all_params, self.rank, self.world_size)
        super().__init__(params=sharded, defaults=kwargs)
        self._optim = optimizer_cls(sharded, **kwargs)

        # precompute which parameters each rank owns
        self.owner_to_indices: dict[int, list[int]] = {r: [] for r in range(self.world_size)}
        for idx, _p in enumerate(self.all_params):
            owner = idx % self.world_size
            self.owner_to_indices[owner].append(idx)
    
    def step(self, closure: Optional[Callable] = None, **kwargs:Any):
        loss = self._optim.step(closure)

        for owner, indices in self.owner_to_indices.items():
            if not indices:
                continue

            sizes = [self.all_params[i].numel() for i in indices]
            total = sum(sizes)
            if self.rank == owner:
                flat = torch.empty(total, device=self.all_params[0].device, dtype=self.all_params[0].dtype)
                offset = 0
                for i in indices:
                    p = self.all_params[i]
                    n = p.numel()
                    flat[offset:offset+n].copy_(p.data.view(-1))
                    offset += n
            else:
                flat = torch.empty(total, device=self.all_params[0].device, dtype=self.all_params[0].dtype)

            # synchronize this owner's shard
            dist.broadcast(flat, src=owner)

            # unpack on all ranks
            offset = 0
            for i in indices:
                p = self.all_params[i]
                n = p.numel()
                p.data.view(-1).copy_(flat[offset:offset+n])
                offset += n

        return loss
    
    def add_param_group(self, param_group: dict[str, Any]):
        if not hasattr(self, "owner_to_indices"):
            return super().add_param_group(param_group)

        # Normalize params into a list
        params = param_group.get("params", [])
        if isinstance(params, torch.nn.Parameter):
            params = [params]
        else:
            params = list(params)

        if len(params) == 0:
            # Nothing to add globally
            return

        local_params: list[torch.nn.Parameter] = []

        # Assign global indices and owners, collect local shard
        for p in params:
            if not isinstance(p, torch.nn.Parameter):
                raise TypeError("add_param_group expects torch.nn.Parameter objects")

            global_idx = len(self.all_params)
            self.all_params.append(p)

            owner = global_idx % self.world_size
            self.owner_to_indices.setdefault(owner, []).append(global_idx)

            if owner == self.rank:
                local_params.append(p)

        # This rank may own none of the params in this group
        if len(local_params) == 0:
            return

        # Build a local param group with same hyperparameters
        local_group = {k: v for k, v in param_group.items() if k != "params"}
        local_group["params"] = local_params

        # Register with the sharded optimizer (base Optimizer)
        super().add_param_group(local_group)

        # Also register with the wrapped optimizer, if it exists
        if hasattr(self, "_optim") and self._optim is not None:
            self._optim.add_param_group(local_group)