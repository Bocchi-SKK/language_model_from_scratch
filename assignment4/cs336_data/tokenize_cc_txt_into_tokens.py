from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import numpy as np
from tqdm import tqdm

def tokenize_cc_text_into_tokens(input_file_path, output_file_path, tokenizer):
    with open(input_file_path, 'r', encoding='utf-8') as f:
        num_lines = sum(1 for _ in f)
    with open(output_file_path, 'wb') as out_f:
        with open(input_file_path, 'r',) as in_f:
            for line in tqdm(in_f, total=num_lines, desc="Tokenizing"):
                ids = tokenizer.encode(line)
                arr = np.array(ids, dtype=np.uint16)
                arr.tofile(out_f)

data_directory_path = Path('/mnt/e/Data/cs336_data/Assignment4/')
gpt2_tokenizer = data_directory_path/'gpt2_tokenizer'
cc_text_file = data_directory_path/'CC_data_extracted/cc_data_extracted_dedup.txt'
cc_token_file = data_directory_path/'CC_data_extracted/cc_gpt2_tokens.bin'
# cc_train_text_file = data_directory_path/'CC_data_extracted/cc_data_extracted_dedup_train.txt'
# cc_train_token_file = data_directory_path/'CC_data_extracted/cc_gpt2_tokens_train.bin'
# cc_test_text_file = data_directory_path/'CC_data_extracted/cc_data_extracted_dedup_test.txt'
# cc_test_token_file = data_directory_path/'CC_data_extracted/cc_gpt2_tokens_test.bin'

tokenizer:PreTrainedTokenizerBase = AutoTokenizer.from_pretrained('gpt2', cache_dir=str(gpt2_tokenizer))
tokenize_cc_text_into_tokens(cc_text_file, cc_token_file, tokenizer)
# tokenize_cc_text_into_tokens(cc_train_text_file, cc_train_token_file, tokenizer)
# tokenize_cc_text_into_tokens(cc_test_text_file, cc_test_token_file, tokenizer)