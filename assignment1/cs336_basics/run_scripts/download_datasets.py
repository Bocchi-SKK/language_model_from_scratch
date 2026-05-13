try:
    from cs336_basics.run_scripts import filepath
except:
    try:
        import filepath
    except:
        assert ImportError
import os
import urllib.request
from urllib.parse import urlparse
from pathlib import Path

def download_url_file(url, target_dir):
    """Download file from URL to target_dir, preserving original filename"""
    # Parse the URL to get the filename
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Full path where file will be saved
    full_path = Path(target_dir) / filename
    
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, full_path)
    print(f"Saved to {full_path}")

TEXT_DATAPATH = filepath.TEXT_DATAPATH

TS_TRAIN_URL = 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt'
TS_VALID_URL = 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt'

OWT_TRAIN_URL = 'https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz'
OWT_VALID_URL = 'https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz'

# Download TinyStories dataset
download_url_file(url=TS_VALID_URL, target_dir=TEXT_DATAPATH)
download_url_file(url=TS_TRAIN_URL, target_dir=TEXT_DATAPATH)

# Download OWT dataset
download_url_file(url=OWT_VALID_URL, path=TEXT_DATAPATH)
download_url_file(url=OWT_TRAIN_URL, path=TEXT_DATAPATH)