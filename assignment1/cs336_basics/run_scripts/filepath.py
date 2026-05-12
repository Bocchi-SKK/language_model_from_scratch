import os
from pathlib import Path
import sys
#===========================
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# project_root = Path(project_root) / 'cs336_basics'
if project_root not in sys.path:
    sys.path.insert(0, project_root)
#===========================
# Vocab and merges path
VOCAB_DATAPATH = Path('E:\\Data\\OneDrive\\Data\\Code\\Python\\Language_Modeling_From_Scratch\\data\\vocab')
os.makedirs(VOCAB_DATAPATH, exist_ok=True)

TS_MERGES_PATH = VOCAB_DATAPATH / 'TinyStories_GPT_merges.txt'
TS_VOCAB_PATH = VOCAB_DATAPATH / 'TinyStories_GPT_vocab.json'

OWT_MERGES_PATH = VOCAB_DATAPATH / 'owt_train_merges.txt'
OWT_VOCAB_PATH = VOCAB_DATAPATH / 'owt_train_vocab.json'
#===========================
# Training and validation data path
TEXT_DATAPATH = Path('E:\\Data\\OneDrive\\Data\\Code\\Python\\Language_Modeling_From_Scratch\\data')
IDS_PATH = TEXT_DATAPATH / 'token_ids'
os.makedirs(IDS_PATH, exist_ok=True)

TS_TRAIN_PATH = TEXT_DATAPATH / 'TinyStoriesV2-GPT4-train.txt'
TS_VALID_PATH = TEXT_DATAPATH / 'TinyStoriesV2-GPT4-valid.txt'
TS_TRAIN_IDS_PATH = IDS_PATH / 'TinyStoriesV2-GPT4-train_ids.bin'
TS_VALID_IDS_PATH = IDS_PATH / 'TinyStoriesV2-GPT4-valid_ids.bin'

OWT_TRAIN_PATH = TEXT_DATAPATH / 'owt_train.txt'
OWT_VALID_PATH = TEXT_DATAPATH / 'owt_valid.txt'
OWT_TRAIN_IDS_PATH = IDS_PATH / 'owt_train_ids.bin'
OWT_VALID_IDS_PATH = IDS_PATH / 'owt_valid_ids.bin'
#===========================
LOGS_PATH = Path('E:\\Data\\OneDrive\\Data\\Code\\Python\\Language_Modeling_From_Scratch\\data\\logs')
os.makedirs(LOGS_PATH, exist_ok=True)
#===========================
SPECIAL_TOKENS = ["<|endoftext|>"]
#===========================