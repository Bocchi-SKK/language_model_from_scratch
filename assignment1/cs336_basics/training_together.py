import torch
from torch import nn
import os
import json
import time
from einops import rearrange
from tqdm import trange

try:
    from data import data_loading
    from optimizer import lr_cosine_schedule
    from nn_utils import gradient_clipping
    from serialization import save_checkpoint, load_checkpoint
except:
    try:
        from cs336_basics.data import data_loading
        from cs336_basics.optimizer import lr_cosine_schedule
        from cs336_basics.nn_utils import gradient_clipping
        from cs336_basics.serialization import save_checkpoint, load_checkpoint
    except:
        ImportError
#===========================
def run_training(model:nn.Module,
                 optimizer:torch.optim.Optimizer,
                 loss_fn,
                 training_dataset,
                 validation_dataset,
                 epochs_number:int,
                 batch_size:int,
                 context_length:int,
                 device:str,
                 iteration:int=0,
                 lr_max:float = 5e-4,
                 lr_min:float = 1e-5,
                 checkpoint_path: str = None,
                 resume: bool = False,
                 patience: int = 2,
                 log_path:str= None,
                 step_fraction:float = 1.0):
    
    # Check learning rate values
    if lr_max <= 0 or lr_min <= 0:
        raise ValueError("lr_max and lr_min must be positive numbers.")
    if lr_max < lr_min:
        raise ValueError("lr_max should be greater than or equal to lr_min.")    
    
    ITERATION = iteration
    best_validation_loss = float('inf')
    epochs_no_improve = 0
    results = []
    start_epoch = 0
    comment = 'None'
    bias_time = 0

    # Optionally resume from checkpoint
    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        ITERATION = load_checkpoint(checkpoint_path, model, optimizer)
        print(f"Resumed from checkpoint at iteration {ITERATION}")

        # Resume log/results
        if log_path and os.path.exists(log_path):
            with open(log_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            if results:
                last_result = results[-1]
                start_epoch = last_result.get("epoch", 0)
                # Set best_validation_loss to the best so far
                best_validation_loss = min(r["validation_loss"] for r in results)
                bias_time = last_result.get("wallclock_time", 0.0)
                print(f"Resuming from epoch {start_epoch+1}")

    training_steps_number = int((len(training_dataset) / (batch_size * context_length)) * step_fraction)
    validation_steps_number = int((len(validation_dataset) / (batch_size * context_length)) * step_fraction)
    T_w = int(0.03 * training_steps_number)
    T_c = int((epochs_number) * training_steps_number)

    def training_loop(model:nn.Module,
                      optimizer:torch.optim.Optimizer,
                      loss_fn,
                      training_dataset,
                      device:str,
                      batch_size:int,
                      context_length:int,
                      steps:int,
                      lr_max:float,
                      lr_min:float,
                      T_w:int,
                      T_c:int):
        '''
        return training loss
        '''
        nonlocal ITERATION
        model.train()
        loss_total = 0
        # for _ in range(steps):
        for _ in trange(steps, desc="Steps", leave=False):
            training_data, targets = data_loading(dataset=training_dataset, batch_size=batch_size, context_length=context_length, device=device)
            optimizer.zero_grad()

            logits = model(training_data)

            logits:torch.Tensor = rearrange(logits, "batch_size context_length vocab_size -> (batch_size context_length) vocab_size")
            targets:torch.Tensor = rearrange(targets, "batch_size context_length -> (batch_size context_length)")
            loss = loss_fn(inputs=logits, targets=targets)

            loss.backward()
            gradient_clipping(parameters=model.parameters(), max_l2_norm=1.0, eps=1e-6)

            # Update learning rate
            ITERATION += 1
            lr = lr_cosine_schedule(t=ITERATION, lr_max=lr_max, lr_min=lr_min, T_w=T_w, T_c =T_c)
            for param_group in optimizer.param_groups:
                param_group['lr'] =lr

            optimizer.step()            

            loss_total += loss.item()
        return loss_total/steps

    def validation_loop(model:nn.Module,
                        loss_fn,
                        validation_dataset,
                        device:str,
                        batch_size:int,
                        context_length:int,
                        steps:int):
        '''
        return validation loss
        '''
        model.eval()
        loss_total = 0
        with torch.no_grad():
            for _ in range(steps):
                training_data, targets = data_loading(dataset=validation_dataset, batch_size=batch_size, context_length=context_length, device=device)

                logits = model(training_data)
                logits:torch.Tensor = rearrange(logits, "batch_size context_length vocab_size -> (batch_size context_length) vocab_size")
                targets:torch.Tensor = rearrange(targets, "batch_size context_length -> (batch_size context_length)")

                loss = loss_fn(inputs=logits, targets=targets)
                loss_total += loss.item()
        return loss_total/steps

    stopped_early = False
    start_time = time.time()
    for epoch in trange(start_epoch, epochs_number, desc="Epoch", leave=False):
        
        training_loss = training_loop(model=model,
                                      optimizer=optimizer,
                                      loss_fn=loss_fn,
                                      training_dataset=training_dataset,
                                      device=device,
                                      batch_size=batch_size,
                                      context_length=context_length,
                                      steps=training_steps_number,
                                      lr_max=lr_max,
                                      lr_min=lr_min,
                                      T_w=T_w,
                                      T_c=T_c)
        
        validation_loss = validation_loop(model=model,
                                          loss_fn=loss_fn,
                                          validation_dataset=validation_dataset,
                                          device=device,
                                          batch_size=batch_size,
                                          context_length=context_length,
                                          steps=validation_steps_number)
        
        
        current_time = time.time()
        elapsed_time = (current_time - start_time) + bias_time

        results.append({
            "epoch": epoch + 1,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "iteration": ITERATION,
            "wallclock_time": elapsed_time
        })

        if log_path:
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            # print(f"Training log saved to {log_path}")
        print(f"\nEpoch {epoch+1}: Training loss = {training_loss:.4f}, Validation loss = {validation_loss:.4f}")

        # Early stopping and checkpointing
        if validation_loss < best_validation_loss - 1e-4:
            best_validation_loss = validation_loss
            epochs_no_improve = 0
            if checkpoint_path:
                save_checkpoint(model, optimizer, ITERATION, checkpoint_path)
                # print(f"Checkpoint saved at epoch {epoch+1}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                comment = f"Early stopping triggered. In Epoch:{epoch+1}"
                stopped_early = True
                print(f"Early stopping triggered. In Epoch:{epoch+1}")
                break

    # After training, if early stopped, reload the best checkpoint
    if stopped_early and checkpoint_path and os.path.exists(checkpoint_path):
        load_checkpoint(checkpoint_path, model, optimizer)
        print("Loaded best model from checkpoint after early stopping.")

    return results