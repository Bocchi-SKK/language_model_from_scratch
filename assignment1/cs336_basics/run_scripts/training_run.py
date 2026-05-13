import torch
import numpy as np
from filepath import *

from cs336_basics import tokenizer
from cs336_basics import model
from cs336_basics import nn_utils
from cs336_basics import optimizer
from cs336_basics.training_together import run_training
#===========================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device=DEVICE)

batch_size = 128
context_length = 256
d_model = 512
num_layers = 4
num_heads = 16
d_ff = 1344
rope_theta = 10000

lr = 1e-3
lr_end = 1e-4

NUM_EPOCHS = 5

assert_datasets_exist([TS_VOCAB_PATH, TS_MERGES_PATH, TS_TRAIN_IDS_PATH, TS_VALID_IDS_PATH])
ts_tokenizer = tokenizer.tokenizer.from_files(vocab_filepath=TS_VOCAB_PATH, merges_filepath=TS_MERGES_PATH, special_tokens=SPECIAL_TOKENS)
TS_VOCAB_SIZE = len(ts_tokenizer.vocab)

# owt_tokenizer = tokenizer.tokenizer.from_files(vocab_filepath=OWT_VOCAB_PATH, merges_filepath=OWT_MERGES_PATH, special_tokens=SPECIAL_TOKENS)
# OWT_VOCAB_SIZE = len(owt_tokenizer.vocab)

my_transformer_model = model.transformer_lm(vocab_size=TS_VOCAB_SIZE,
                                            context_length=context_length,
                                            d_model=d_model,
                                            num_layers=num_layers,
                                            num_heads=num_heads,
                                            d_ff=d_ff,
                                            rope_theta=rope_theta)
my_transformer_model.context_length = context_length

loss_fn = nn_utils.cross_entropy

# Tiny Story
training_dataset = np.memmap(filename=TS_TRAIN_IDS_PATH, dtype=np.uint16)
validation_dataset = np.memmap(filename=TS_VALID_IDS_PATH, dtype=np.uint16)

# Open web text
# training_dataset = np.memmap(filename=OWT_TRAIN_IDS_PATH, dtype=np.uint16)
# validation_dataset = np.memmap(filename=OWT_VALID_IDS_PATH, dtype=np.uint16)

my_AdamW = optimizer.AdamW(params=my_transformer_model.parameters())

comment = "TY_model"
checkpoint_path = LOGS_PATH / f'check_point_{comment}.pt'
log_path = LOGS_PATH / f'log_{comment}.json'

results = run_training(model=my_transformer_model,
                       optimizer=my_AdamW,
                       loss_fn=loss_fn,
                       training_dataset=training_dataset,
                       validation_dataset=validation_dataset,
                       epochs_number=NUM_EPOCHS,
                       batch_size=batch_size,
                       context_length=context_length,
                       device=DEVICE,
                       lr_max=lr,
                       lr_min=lr_end,
                       checkpoint_path=checkpoint_path,
                       log_path=log_path,
                       patience=2,
                       step_fraction=0.1,
                       resume=False)

torch.save(my_transformer_model, TS_MODEL_PATH)
print("Training done")