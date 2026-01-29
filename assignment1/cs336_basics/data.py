import numpy as np
import torch

def data_loading(dataset, batch_size, context_length, device):
    num_possible = len(dataset) - context_length
    if num_possible <= 0:
        raise ValueError("Not enough data to sample a batch.")
    start_indices = np.random.randint(0, num_possible, size=batch_size, dtype=np.int64)
    input = []
    target = []
    for i in start_indices:
        input.append(dataset[i : i + context_length])
        target.append(dataset[i + 1 : i + context_length + 1])
    input = np.array(input)
    target = np.array(target)
    input = torch.tensor(input, dtype=torch.long, device=device)
    target = torch.tensor(target, dtype=torch.long, device=device)
    return input, target