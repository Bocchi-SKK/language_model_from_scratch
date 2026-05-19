import json
from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import numpy as np

def tokenize_c4_into_tokens(data_directory_path):
    c4_file_path = [data_directory_path / f'c4-validation.0000{i}-of-00008.json' for i in range(8)]

    tokenizer:PreTrainedTokenizerBase = AutoTokenizer.from_pretrained('gpt2', cache_dir=str(data_directory_path))

    output_path = data_directory_path / 'c4_validation_gpt2_tokens.bin'
    eot_token_id = tokenizer.eos_token_id  # <|endoftext|> token id
    with open(output_path, 'wb') as out_f:
        for path in c4_file_path:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    example = json.loads(line)
                    text = example['text']
                    ids = tokenizer.encode(text, add_special_tokens=False)
                    ids.append(eot_token_id)
                    arr = np.array(ids, dtype=np.uint16)
                    arr.tofile(out_f)

data_directory_path = Path('/mnt/e/Data/cs336_data/Assignment4/')
validation_tokens_path = data_directory_path / 'c4_validation_gpt2_tokens.bin'
tokenizer:PreTrainedTokenizerBase = AutoTokenizer.from_pretrained('gpt2', cache_dir=str(data_directory_path))
tokenize_c4_into_tokens(data_directory_path)