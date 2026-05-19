import fasttext
from pathlib import Path
from tqdm import tqdm
from fastwarc.warc import ArchiveIterator, WarcRecordType
from text_filter import gopher_quality_filter, minhash_deduplication_one_file
import multiprocessing
import os
import shutil
import re
import random

data_directory = Path("/mnt/e/Data/cs336_data/Assignment4")
cc_raw_examples = data_directory / "CC_data"
wet_files = cc_raw_examples.rglob("*.wet.gz")
cc_data_extracted = data_directory / 'CC_data_extracted/cc_data_extracted.txt'
C4_LIKE_NLP_MODEL_PATH = str(data_directory / 'train_c4_filter/c4_like_model.bin')
LANGUAGE_NLP_MODEL_PATH = str(data_directory / 'lid.176.bin')
NSFW_MODEL_PATH = '/mnt/e/Data/cs336_data/Assignment4/jigsaw_fasttext_bigrams_nsfw_final.bin'
HEATSPEECH_MODEL_PATH = '/mnt/e/Data/cs336_data/Assignment4/jigsaw_fasttext_bigrams_hatespeech_final.bin'

cc_data_extracted_dedup = data_directory / "CC_data_extracted/cc_data_extracted_dedup.txt"
cc_data_extracted_dedup_train = data_directory / "CC_data_extracted/cc_data_extracted_dedup_train.txt"
cc_data_extracted_dedup_test = data_directory / "CC_data_extracted/cc_data_extracted_dedup_test.txt"

def chunkify(input_list, workers):
    output_list = [[]for _ in range(workers)]
    for i in range(len(input_list)):
        output_list[i%workers].append(input_list[i])
    return output_list

def extract_c4_like_texts_single(input_files, output_path):
    language_model = fasttext.load_model(LANGUAGE_NLP_MODEL_PATH)
    c4_model = fasttext.load_model(C4_LIKE_NLP_MODEL_PATH)
    toxic_model = fasttext.load_model(HEATSPEECH_MODEL_PATH)
    nsfw_model = fasttext.load_model(NSFW_MODEL_PATH)
    
    with open(output_path, mode='a', encoding='utf-8') as out_f:
        for wet_file in tqdm(input_files, desc=f"Worker {output_path}"):
            with open(wet_file, "rb") as stream:
                for record in ArchiveIterator(stream, record_types=WarcRecordType.conversion):
                    web_text = record.reader.read().decode("utf-8", errors="ignore").strip()
                    if not web_text:
                        continue

                    if not gopher_quality_filter(web_text):
                        continue

                    input_text = web_text.replace('\n',' ')
                    lang, _ = language_model.predict(input_text)
                    lang:str = lang[0].replace("__label__","")
                    if lang != 'en':
                        continue

                    toxic_label, toxic_prob = toxic_model.predict(input_text)
                    toxic_label = toxic_label[0].replace("__label__", "")
                    toxic_prob = toxic_prob[0]
                    if toxic_label == 'toxic' and toxic_prob >= 0.75:
                        continue

                    nsfw_label, nsfw_prob = nsfw_model.predict(input_text)
                    nsfw_label = nsfw_label[0].replace("__label__", "")
                    nsfw_prob = nsfw_prob[0]
                    if nsfw_label == 'nsfw' and nsfw_prob >= 0.60:
                        continue

                    label, prob = c4_model.predict(input_text)
                    label = label[0].replace("__label__","")
                    if label == 'c4_eng' and prob[0] >= 0.85:
                        web_text = re.sub(r'\n+', '\n', web_text)
                        out_f.write(web_text+'<|endoftext|>')

def extract_c4_like_texts_multi(input_files, output_path, workers:int):
    input_files = list(input_files)
    file_groups = chunkify(input_files, workers)
    temp_files = [f"{output_path}.{i}.tmp" for i in range(workers)]
    processes = []

    for group, temp_file in zip(file_groups, temp_files):
        p = multiprocessing.Process(target=extract_c4_like_texts_single, args=(group, temp_file))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Merge temp files
    with open(output_path, "wb") as outfile:
        for temp_file in temp_files:
            if not os.path.exists(temp_file):
                continue

            with open(temp_file, "rb") as infile:
                shutil.copyfileobj(infile, outfile, length=1024 * 1024)

            os.remove(temp_file)

    return 

def split_into_training_validation(
    input_file,
    train_file,
    test_file,
    train_ratio=0.8,
    delimiter='<|endoftext|>',
    chunk_size=1024 * 1024,
    seed=42,
):
    if not 0 < train_ratio < 1:
        raise ValueError('train_ratio must be between 0 and 1.')

    input_file = Path(input_file)
    train_file = Path(train_file)
    test_file = Path(test_file)

    train_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.parent.mkdir(parents=True, exist_ok=True)

    delimiter_bytes = delimiter.encode('utf-8')
    delimiter_length = len(delimiter_bytes)
    rng = random.Random(seed)
    buffer = b''
    train_count = 0
    test_count = 0

    with open(input_file, 'rb') as in_f, open(train_file, 'wb') as train_f, open(test_file, 'wb') as test_f:
        with tqdm(total=input_file.stat().st_size, unit='B', unit_scale=True, desc='Splitting train/validation') as progress_bar:
            while True:
                chunk = in_f.read(chunk_size)
                if not chunk:
                    break

                buffer += chunk
                progress_bar.update(len(chunk))
                start = 0

                while True:
                    end = buffer.find(delimiter_bytes, start)
                    if end == -1:
                        buffer = buffer[start:]
                        break

                    document = buffer[start:end]
                    start = end + delimiter_length
                    if not document.strip():
                        continue

                    if rng.random() < train_ratio:
                        train_f.write(document)
                        train_f.write(delimiter_bytes)
                        train_count += 1
                    else:
                        test_f.write(document)
                        test_f.write(delimiter_bytes)
                        test_count += 1

            if buffer.strip():
                if rng.random() < train_ratio:
                    train_f.write(buffer)
                    train_f.write(delimiter_bytes)
                    train_count += 1
                else:
                    print(buffer)
                    test_f.write(buffer)
                    test_f.write(delimiter_bytes)
                    test_count += 1

    return train_count, test_count


# extract_c4_like_texts_multi(input_files=wet_files, output_path=cc_data_extracted, workers=6)
# minhash_deduplication_one_file(input_file=cc_data_extracted, output_file=cc_data_extracted_dedup, workers=16)
split_into_training_validation(input_file=cc_data_extracted_dedup, train_file=cc_data_extracted_dedup_train, test_file=cc_data_extracted_dedup_test, train_ratio=0.8)