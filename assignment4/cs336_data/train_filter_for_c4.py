from pathlib import Path
from fastwarc.warc import ArchiveIterator, WarcRecordType
from text_filter import gopher_quality_filter
from tqdm import tqdm
import os
import multiprocessing
import fasttext
import json
from train_fasttext_model import train_model

try:
    from text_filter import *
    import file_paths
except:
    try:
        from cs336_data.text_filter import *
        from cs336_data import file_paths
    except:
        ImportError

data_directory = Path("/mnt/e/Data/cs336_data/Assignment4")

cc_raw_examples = data_directory / "train_c4_filter/raw_cc"
wet_files = cc_raw_examples.rglob("*.wet.gz")
cc_like_english_texts = data_directory / 'train_c4_filter/cc_like.txt'
cc_raw_eng = data_directory / 'train_c4_filter/cc_raw_eng.txt'
c4_validation_json_files = [data_directory / f'c4-validation.0000{i}-of-00008.json' for i in range(8)]
c4_validation_txt_file = data_directory / 'c4-validation.txt'
c4_eng_file = data_directory / 'train_c4_filter/c4_eng.txt'
model_path = data_directory / 'train_c4_filter/c4_like_model.bin'

train_path = data_directory / 'train_c4_filter/c4_train_fasttext.txt'
val_path = data_directory / 'train_c4_filter/c4_val_fasttext.txt'

def chunkify(input_list, workers):
    output_list = [[]for _ in range(workers)]
    for i in range(len(input_list)):
        output_list[i%workers].append(input_list[i])
    return output_list

def extract_english_texts_single(input_files, output_path):
    '''
        Extract English texts from raw Common Crawl data for NLP model training.
    '''
    NLP_MODEL_PATH = '/mnt/e/Data/cs336_data/Assignment4/lid.176.bin'
    model = fasttext.load_model(NLP_MODEL_PATH)
    for wet_file in tqdm(input_files, desc=f"Worker {output_path}"):
        with open(wet_file, "rb") as stream:
            for record in ArchiveIterator(stream, record_types=WarcRecordType.conversion):
                web_text = record.reader.read().decode("utf-8", errors="ignore").strip()
                if web_text:
                    if gopher_quality_filter(web_text):
                        input_text = web_text.replace('\n',' ')
                        lang, prob = model.predict(input_text)
                        lang:str = lang[0].replace("__label__","")
                        prob = prob[0]
                        # lang, prob = identify_language(web_text)
                        if lang == 'en':
                            with open(output_path, mode='a', encoding='utf-8') as f:
                                f.write(web_text+'<|endoftext|>')

def extract_english_texts_multi(input_files, output_path, workers:int):
    input_files = list(input_files)
    file_groups = chunkify(input_files, workers)
    temp_files = [f"{output_path}.{i}.tmp" for i in range(workers)]
    processes = []

    for group, temp_file in zip(file_groups, temp_files):
        p = multiprocessing.Process(target=extract_english_texts_single, args=(group, temp_file))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Merge temp files
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for temp_file in temp_files:
            with open(temp_file, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
            os.remove(temp_file)

    return 

def convert_c4_json_into_c4_txt(input_files, output_file):
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for file in input_files:
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        text = json.loads(line)['text']
                        out_f.write(text + '<|endoftext|>')

# extract_english_texts_multi(input_files=wet_files, output_path=cc_like_english_texts, workers=8)  
# convert_c4_json_into_c4_txt(input_files=c4_validation_json_files, output_file=c4_validation_txt_file) 

train_model(first_input_path=c4_eng_file, second_input_path=cc_raw_eng, model_output_path=model_path)