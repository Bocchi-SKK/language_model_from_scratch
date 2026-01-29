import torch
import triton
import csv

from cs336_basics.model import Attention
from flash_attention import flash_attention_triton

results = []

for i in range(7, 16):
    batch_size = 1
    num_heads = 1
    seq_length = 2 ** i
    d_head = 128
    dtype = torch.float16
    device = 'cuda'

    Q0 = torch.randn(batch_size, num_heads, seq_length, d_head, device=device, dtype=dtype, requires_grad=True)
    K0 = torch.randn(batch_size, num_heads, seq_length, d_head, device=device, dtype=dtype, requires_grad=True)
    V0 = torch.randn(batch_size, num_heads, seq_length, d_head, device=device, dtype=dtype, requires_grad=True)

    Q1 = torch.randn(batch_size, num_heads, seq_length, d_head, device=device, dtype=dtype, requires_grad=True)
    K1 = torch.randn(batch_size, num_heads, seq_length, d_head, device=device, dtype=dtype, requires_grad=True)
    V1 = torch.randn(batch_size, num_heads, seq_length, d_head, device=device, dtype=dtype, requires_grad=True)

    # Forward
    flash_fwd_time = triton.testing.do_bench(lambda: flash_attention_triton.apply(Q0, K0, V0))
    naive_fwd_time = triton.testing.do_bench(lambda: Attention(Q1, K1, V1))

    # Forward pass (outside benchmark)
    Qf = Q0.detach().clone().requires_grad_(True)
    Kf = K0.detach().clone().requires_grad_(True)
    Vf = V0.detach().clone().requires_grad_(True)
    out_flash = flash_attention_triton.apply(Qf, Kf, Vf)
    grad_flash = torch.randn_like(out_flash)

    Qn = Q1.detach().clone().requires_grad_(True)
    Kn = K1.detach().clone().requires_grad_(True)
    Vn = V1.detach().clone().requires_grad_(True)
    out_naive = Attention(Qn, Kn, Vn)
    grad_naive = torch.randn_like(out_naive)

    # Backward functions (only do backward)
    def flash_backward():
        out_flash.backward(grad_flash, retain_graph=True)

    def naive_backward():
        out_naive.backward(grad_naive, retain_graph=True)

    # Benchmark backward only
    flash_bwd_time = triton.testing.do_bench(flash_backward)
    naive_bwd_time = triton.testing.do_bench(naive_backward)

    # Save results
    results.append({
        "seq_length": seq_length,
        "d_head": d_head,
        "dtype": str(dtype),
        "flash_fwd_ms": f"{flash_fwd_time:.4f}",
        "naive_fwd_ms": f"{naive_fwd_time:.4f}",
        "flash_bwd_ms": f"{flash_bwd_time:.4f}",
        "naive_bwd_ms": f"{naive_bwd_time:.4f}",
        "flash_fwd_bwd_ms": f"{flash_fwd_time + flash_bwd_time:.4f}",
        "naive_fwd_bwd_ms": f"{naive_fwd_time + naive_bwd_time:.4f}",
    })

for i in range(7, 16):
    batch_size = 1
    num_heads = 1
    seq_length = 2 ** i
    d_head = 128
    dtype = torch.float32
    device = 'cuda'

    Q0 = torch.randn(batch_size, num_heads, seq_length, d_head, device=device, dtype=dtype, requires_grad=True)
    K0 = torch.randn(batch_size, num_heads, seq_length, d_head, device=device, dtype=dtype, requires_grad=True)
    V0 = torch.randn(batch_size, num_heads, seq_length, d_head, device=device, dtype=dtype, requires_grad=True)

    Q1 = torch.randn(batch_size, num_heads, seq_length, d_head, device=device, dtype=dtype, requires_grad=True)
    K1 = torch.randn(batch_size, num_heads, seq_length, d_head, device=device, dtype=dtype, requires_grad=True)
    V1 = torch.randn(batch_size, num_heads, seq_length, d_head, device=device, dtype=dtype, requires_grad=True)

    # Forward
    flash_fwd_time = triton.testing.do_bench(lambda: flash_attention_triton.apply(Q0, K0, V0))
    naive_fwd_time = triton.testing.do_bench(lambda: Attention(Q1, K1, V1))

    # Forward pass (outside benchmark)
    Qf = Q0.detach().clone().requires_grad_(True)
    Kf = K0.detach().clone().requires_grad_(True)
    Vf = V0.detach().clone().requires_grad_(True)
    out_flash = flash_attention_triton.apply(Qf, Kf, Vf)
    grad_flash = torch.randn_like(out_flash)

    Qn = Q1.detach().clone().requires_grad_(True)
    Kn = K1.detach().clone().requires_grad_(True)
    Vn = V1.detach().clone().requires_grad_(True)
    out_naive = Attention(Qn, Kn, Vn)
    grad_naive = torch.randn_like(out_naive)

    # Backward functions (only do backward)
    def flash_backward():
        out_flash.backward(grad_flash, retain_graph=True)

    def naive_backward():
        out_naive.backward(grad_naive, retain_graph=True)

    # Benchmark backward only
    flash_bwd_time = triton.testing.do_bench(flash_backward)
    naive_bwd_time = triton.testing.do_bench(naive_backward)

    # Save results
    results.append({
        "seq_length": seq_length,
        "d_head": d_head,
        "dtype": str(dtype),
        "flash_fwd_ms": f"{flash_fwd_time:.4f}",
        "naive_fwd_ms": f"{naive_fwd_time:.4f}",
        "flash_bwd_ms": f"{flash_bwd_time:.4f}",
        "naive_bwd_ms": f"{naive_bwd_time:.4f}",
        "flash_fwd_bwd_ms": f"{flash_fwd_time + flash_bwd_time:.4f}",
        "naive_fwd_bwd_ms": f"{naive_fwd_time + naive_bwd_time:.4f}",
    })

# Write to CSV
csv_path = "attention_bechmark.csv"
with open(csv_path, "w", newline="") as csvfile:
    fieldnames = [
        "seq_length", "d_head", "dtype",
        "flash_fwd_ms", "naive_fwd_ms",
        "flash_bwd_ms", "naive_bwd_ms",
        "flash_fwd_bwd_ms", "naive_fwd_bwd_ms"
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"Results saved to {csv_path}")