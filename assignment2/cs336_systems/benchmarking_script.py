import torch
import sys
from timeit import default_timer as timer
from einops import rearrange
import pandas as pd
import torch.cuda.nvtx as nvtx
from torch.amp import autocast
from contextlib import nullcontext

sys.path.insert(0,r'./cs336-basics')
from cs336_basics import model
from cs336_basics import nn_utils
from cs336_basics import optimizer

def benchmark(vocab_size,
              context_length,
              model,
              batch_size,
              device,
              warm_up_steps,
              loss_fn,
              model_name,
              use_mixed_precision=False):

    my_adamw = optimizer.AdamW(model.parameters())
    autocast_ctx = autocast(device_type="cuda", dtype=torch.bfloat16) if use_mixed_precision else nullcontext()

    # Warm up
    for _ in range(warm_up_steps):
        my_adamw.zero_grad()
        input_tensor = torch.randint(0,vocab_size,(batch_size, context_length), dtype=torch.long)
        targets = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
        targets = rearrange(targets, "batch_size context_length -> (batch_size context_length)")
        with autocast_ctx:
            logits = model(input_tensor)
            logits = rearrange(logits, "batch_size context_length vocab_size -> (batch_size context_length) vocab_size")
            loss = loss_fn(inputs=logits, targets=targets)
        loss.backward()
        my_adamw.step()

    # Time benchmark
    input_tensor = torch.randint(0,vocab_size,(batch_size, context_length), dtype=torch.long)
    targets = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    targets = rearrange(targets, "batch_size context_length -> (batch_size context_length)")


    torch.cuda.synchronize()
    with nvtx.range(f"{model_name} forward"):
        with autocast_ctx:
            start = 1000*timer()
            logits = model(input_tensor)
            logits = rearrange(logits, "batch_size context_length vocab_size -> (batch_size context_length) vocab_size")
            torch.cuda.synchronize()
            end = 1000*timer()
    forward_time = end-start

    torch.cuda.synchronize()
    
    with nvtx.range(f"{model_name} loss"):
        with autocast_ctx:
            start = 1000*timer()
            loss = loss_fn(inputs=logits, targets=targets)
            torch.cuda.synchronize()
    end = 1000*timer()
    loss_time = end-start

    torch.cuda.synchronize()
    start = 1000*timer()
    with nvtx.range(f"{model_name} backward"):
        loss.backward()
    torch.cuda.synchronize()
    end = 1000*timer()
    backward_time = end-start

    start = 1000*timer()
    with nvtx.range(f"{model_name} optimizer"):
        my_adamw.step()
        my_adamw.zero_grad()
    torch.cuda.synchronize()
    end = 1000*timer()
    optimizer_time = end-start

    return forward_time, loss_time, backward_time, optimizer_time

def bench_loop(configs, device, compile=False):
    vocab_size = 12800
    context_length = 512
    records = []

    for cfg in configs:
        test_model = model.transformer_lm(vocab_size=vocab_size,
                                          context_length=context_length,
                                          d_model=cfg['d_model'],
                                          num_layers=cfg['num_layers'],
                                          num_heads=cfg['num_heads'],
                                          d_ff=cfg['d_ff'],
                                          rope_theta=10000)
        if (compile):
            test_model = torch.compile(test_model)
        torch.cuda.synchronize()
        # for use_mixed_precision in [False, True]:
        for use_mixed_precision in [True]:
            result = dict(cfg)
            result["mixed_precision"] = use_mixed_precision
            result["num_parameters"] = sum(p.numel() for p in test_model.parameters())
            torch.cuda.synchronize()
            result["forward_time"], result["loss_time"], result["backward_time"], result['optimizer_time'] = benchmark(vocab_size=vocab_size,
                                                                                                                       context_length=context_length,
                                                                                                                       model=test_model,
                                                                                                                       batch_size=3,
                                                                                                                       device=device,
                                                                                                                       warm_up_steps=5,
                                                                                                                       loss_fn=nn_utils.cross_entropy,
                                                                                                                       model_name=cfg['name'] + ("_bf16" if use_mixed_precision else "_fp32"),
                                                                                                                       use_mixed_precision=use_mixed_precision)
            torch.cuda.synchronize()
            records.append(result)

    df = pd.DataFrame(records)
    print(df)
    return df

def attention_benchmark(compile=False):
    d_models = [16, 32, 64, 128]
    seq_lens = [256, 1024, 4096, 8192, 16384]
    seq_lens = [256, 1024, 4096, 8192]
    batch_size = 8
    device = torch.device("cuda")
    MAX_GPU_MEM = 32 * 2 ** 30  # 32GB in bytes
    attention = model.Attention

    if compile:
        attention = torch.compile(attention)

    for d_model in d_models:
        for seq_len in seq_lens:
            try:
                Q = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
                K = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
                V = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
                if torch.cuda.memory_allocated() > MAX_GPU_MEM:
                    raise RuntimeError("Manual OOM: GPU memory usage exceeded 32GB after allocation.out of memory")
                # Warmup and check memory usage
                for _ in range(10):
                    out = attention(Q, K, V)
                    if torch.cuda.memory_allocated() > MAX_GPU_MEM:
                        raise RuntimeError("Manual OOM: GPU memory usage exceeded 32GB after allocation.out of memory")
                    grad = torch.randn_like(out)
                    out.backward(grad)
                    if torch.cuda.memory_allocated() > MAX_GPU_MEM:
                        raise RuntimeError("Manual OOM: GPU memory usage exceeded 32GB after allocation.out of memory")
                torch.cuda.synchronize()
                # Forward timing
                torch.cuda.reset_peak_memory_stats()
                start = timer()
                for _ in range(100):
                    out = attention(Q,K,V)
                    torch.cuda.synchronize()
                forward_time = (timer() - start) / 100
                forward_mem = torch.cuda.max_memory_allocated()
                # Backward timing
                grad = torch.randn_like(out)
                torch.cuda.reset_peak_memory_stats()
                start = timer()
                back_sum = 0
                for _ in range(100):
                    out = attention(Q, K, V)
                    torch.cuda.synchronize()
                    start = timer()
                    out.backward(grad)
                    torch.cuda.synchronize()
                    back_sum += timer() - start
                backward_time = back_sum / 100
                backward_mem = torch.cuda.max_memory_allocated()
                print(f"d_model={d_model}, seq_len={seq_len}, forward={forward_time*1000:.4f}ms, backward={backward_time*1000:.4f}ms, forward_mem={forward_mem/1e9:.2f}GB, backward_mem={backward_mem/1e9:.2f}GB")
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"d_model={d_model}, seq_len={seq_len}: OOM")
                    torch.cuda.empty_cache()
                else:
                    raise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')
torch.set_default_device(device)
torch.cuda.init()

# configs = [
#     dict(name="small", d_model=768, d_ff=3072, num_layers=12, num_heads=12),
#     dict(name="medium", d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
#     dict(name="large", d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
#     dict(name="xl", d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
#     dict(name="2.7B", d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
# ]

configs = [dict(name="large", d_model=1280, d_ff=5120, num_layers=36, num_heads=20)]

# torch.cuda.memory._record_memory_history(max_entries=1000000)
# bench_loop(configs, device)
bench_loop(configs, device,compile=True)
# torch.cuda.memory._dump_snapshot("memory_snapshot_forward_FP32.pickle")
# torch.cuda.memory._record_memory_history(enabled=None)

# attention_benchmark(compile=True)
# attention_benchmark(compile=False)