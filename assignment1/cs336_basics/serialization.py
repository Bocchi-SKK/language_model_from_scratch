<<<<<<< HEAD
import torch

def save_checkpoint(model, optimizer, iteration, out):
    saving_data = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'iteration': iteration}
    torch.save(saving_data, out)
    return

def load_checkpoint(src, model, optimizer=None):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    iteration = checkpoint['iteration']
=======
import torch

def save_checkpoint(model, optimizer, iteration, out):
    saving_data = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'iteration': iteration}
    torch.save(saving_data, out)
    return

def load_checkpoint(src, model, optimizer=None):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    iteration = checkpoint['iteration']
>>>>>>> d279b46 (Initial commit: Complete Language Modeling From Scratch implementation with Assignments 1-5)
    return iteration