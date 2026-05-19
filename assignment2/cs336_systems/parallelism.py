import torch
from cs336_basics.model import transformer_lm

model = transformer_lm(
    vocab_size=128,
    context_length=256,
    d_model=64,
    num_layers=4,
    num_heads=4,
    d_ff=128,
    rope_theta=10000
)

weights = list(model.parameters())
for idx,p in enumerate(weights):
    temp_p = p
    new_tensor = torch.full_like(temp_p, fill_value=12568.0, device=temp_p.device, requires_grad=temp_p.requires_grad)
    with torch.no_grad():
        temp_p.copy_(new_tensor)

for idx,p in enumerate(weights):
    print(p)