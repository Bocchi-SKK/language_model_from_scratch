import torch
from torch import nn
import torch.distributed as dist

class ddp_individual_parameters(nn.Module):
    def __init__(self, module:torch.nn.Module):
        super().__init__()
        self.module = module

        # Make sure process group is initialized and ranks are known
        if dist.is_initialized():
            # Broadcast all parameters from rank 0
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0)

        self.handles = []

        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._make_hook(param))

    def _make_hook(self, param:torch.nn.Parameter):
        def hook(_):
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=True)
            self.handles.append(handle)
        return hook

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        # Call this after backward to synchronize the gradients
        for handle in self.handles:
            handle.wait()
        self.handles.clear()

class ddp_overlap_bucketed(nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module

        if dist.is_initialized():
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0)

        self.bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)
        self.buckets = self._make_buckets()

        # Per-iteration state
        self.bucket_handles = []
        self.bucket_ready = [set() for _ in self.buckets]

        # Register hooks for each parameter
        for bucket_idx, bucket in enumerate(self.buckets):
            for param in bucket:
                param.register_post_accumulate_grad_hook(self._make_hook(bucket_idx, param))

    def _make_buckets(self):
        # Group parameters into buckets by size (reverse order)
        parameters = [p for p in self.module.parameters() if p.requires_grad]

        buckets = []
        current_bucket = []
        current_size = 0

        for param in reversed(list(parameters)):
            param_size = param.numel() * param.element_size()
            if current_size + param_size > self.bucket_size_bytes and current_bucket:
                buckets.append(current_bucket)
                current_bucket = []
                current_size = 0
            current_bucket.append(param)
            current_size += param_size

        if current_bucket:
            buckets.append(current_bucket)

        return buckets

    def _make_hook(self, bucket_idx, param):
        def hook(_):
            self.bucket_ready[bucket_idx].add(param)
            if len(self.bucket_ready[bucket_idx]) == len(self.buckets[bucket_idx]):
                for p in self.buckets[bucket_idx]:  # Use a different name here
                    if p.grad is None:
                        p.grad = torch.zeros_like(p.data)
                grads_flat = [p.grad.view(-1) for p in self.buckets[bucket_idx]]
                bucket_tensor = torch.cat(grads_flat)
                handle = dist.all_reduce(bucket_tensor, dist.ReduceOp.AVG, async_op=True)
                self.bucket_handles.append((bucket_idx, handle, bucket_tensor))
        return hook

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def reset_bucket_state(self):
        # Call this at the start of each iteration
        self.bucket_handles.clear()
        self.bucket_ready = [set() for _ in self.buckets]
    
    def finish_gradient_synchronization(self):
        # Call this after backward to synchronize the gradients
        for bucket_idx, handle, bucket_tensor in self.bucket_handles:
            handle.wait()
            offset = 0
            for param in self.buckets[bucket_idx]:
                numel = param.grad.numel()
                param.grad.copy_(bucket_tensor[offset:offset+numel].view_as(param.grad))
                offset += numel
        self.bucket_handles.clear()