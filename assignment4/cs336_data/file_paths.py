from pathlib import Path
import os

data_directory = Path('data')
raw_data_directory = data_directory / 'raw_data'
filtered_data_directory = data_directory / 'filtered_data'
model_directory = data_directory / 'models'
os.makedirs(data_directory, exist_ok=True)
os.makedirs(raw_data_directory, exist_ok=True)
os.makedirs(model_directory, exist_ok=True)
os.makedirs(filtered_data_directory, exist_ok=True)

lid_176_bin_path = model_directory / 'lid.176.bin'
NSFW_model_path = model_directory / 'jigsaw_fasttext_bigrams_nsfw_final.bin'
hatespeech_model_path = model_directory / 'jigsaw_fasttext_bigrams_hatespeech_final.bin'

urls_path = raw_data_directory / 'enwiki-20240420-extracted_urls.txt'
high_quality_txt_path = filtered_data_directory / 'high_quality.txt'

warc_paths = [
    raw_data_directory / 'CC-MAIN-20250417135010-20250417165010-00065.warc',
]
low_quality_txt_path = filtered_data_directory / 'low_quality.txt'

train_path = filtered_data_directory / 'train_fasttext.txt'
val_path = filtered_data_directory /  'val_fasttext.txt'
quality_classifier_model_path = model_directory / 'quality_classifier.bin'