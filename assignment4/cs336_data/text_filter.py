from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
from resiliparse.parse.html import HTMLTree
import fasttext
import re
from collections import Counter, defaultdict
from pathlib import Path
import multiprocessing
import mmap
import os
import tempfile
import unicodedata
import random
import hashlib
from itertools import combinations
import shutil
from tqdm import tqdm

try:
    from text_filter import *
    import file_paths
except:
    try:
        from cs336_data.text_filter import *
        from cs336_data import file_paths
    except:
        ImportError

def extract_text_from_html_bytes(html_bytes: bytes) -> str:
    encoding = detect_encoding(html_bytes)
    html_str = html_bytes.decode(encoding, errors='ignore')
    tree = HTMLTree.parse(html_str)
    plain_text = extract_plain_text(tree)
    return plain_text

def identify_language(input_text: str):
    NLP_MODEL_PATH = file_paths.lid_176_bin_path
    model = fasttext.load_model(str(NLP_MODEL_PATH))
    input_text = input_text.replace('\n',' ')
    lang, prob = model.predict([input_text])
    output_lang:str = lang[0][0].replace('__label__','')
    output_prob = float(prob[0][0])
    return output_lang, output_prob

def mask_emails(input_text: str):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+.[a-zA-Z0-9-.]+'
    masked, count = re.subn(email_pattern, "|||EMAIL_ADDRESS|||", input_text)
    return masked, count

def mask_phone_numbers(input_text: str):
    phone_number_pattern = r'(\+1\s?)?(\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}' # regex pattern to match most US phone numbers
    masked, count = re.subn(phone_number_pattern, "|||PHONE_NUMBER|||", input_text)
    return masked, count
    
def mask_ips(input_text: str):
    ip_pattern = (
        r'\b('
        r'(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])\.){3}'
        r'(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])'
        r'\b'
    )
    masked, count = re.subn(ip_pattern, "|||IP_ADDRESS|||", input_text)
    return masked, count

def classify_nsfw(input_text: str):
    NSFW_MODEL_PATH = file_paths.NSFW_model_path
    model = fasttext.load_model(str(NSFW_MODEL_PATH))
    input_text = input_text.replace('\n',' ')
    label, prob = model.predict([input_text])
    output_label = label[0][0].replace("__label__", "")
    output_prob = float(prob[0][0])
    return output_label, output_prob

def classify_toxic_speech(input_text: str):
    HATESPEECH_MODEL_PATH = file_paths.hatespeech_model_path
    model = fasttext.load_model(str(HATESPEECH_MODEL_PATH))
    input_text = input_text.replace('\n',' ')
    label, prob = model.predict([input_text])
    output_label = label[0][0].replace("__label__", "")
    output_prob = float(prob[0][0])
    return output_label, output_prob

def gopher_quality_filter(input_text: str):
    '''
    Remove the documents that:
    type_1: Contain less than 50 or more than 100,000 words
    type_2: Have a mean word lenght outside the range of 3 to 10 characters
    type_3: Have more than 30% of lines ending with an ellipsis("...")
    type_4: Contain less than 80% of words with at least one alphabetic character.
    
    return(bool): The input_text pass the Gopher quality filters or not
    '''
    def type_1(text: str):
        words_count = len(text.split())
        if words_count > 100000 or words_count < 50:
            return True
        else:
            return False
        
    def type_2(text: str):
        words = text.split()
        if not words:
            return True # Empty input
        mean_length = sum(len(word) for word in words) / len(words)
        if mean_length < 3 or mean_length > 10:
            return True
        else:
            return False

    def type_3(text: str):
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            return False
        
        ellipsis_count = sum(
            1 for line in lines
            if re.search(r'(?:\.\.\.|…)\s*$', line)
        )

        if (ellipsis_count / len(lines)) > 0.3:
            return True
        else:
            return False
        
    def type_4(text: str):
        words = text.split()
        if not words:
            return True
        alpha_words = sum(1 for w in words if any(ch.isalpha() for ch in w))
        if (alpha_words / len(words)) < 0.8:
            return True
        else:
            return False

    if (type_1(input_text) or type_2(input_text) or type_3(input_text) or type_4(input_text)):
        # If is one of above type then we should not use this text to train the model.
        return False
    else:
        return True
    
def classify_quality(input_text: str):
    MODEL_PATH = file_paths.quality_classifier_model_path
    model = fasttext.load_model(str(MODEL_PATH))
    input_text = input_text.replace('\n', ' ')
    labels, probs = model.predict(text=[input_text])
    label = labels[0][0].replace("__label__", "")
    prob = float(probs[0][0])
    return label, prob

def exact_line_deduplication(input_files: list[os.PathLike], output_directory: os.PathLike):    
    output_directory = Path(output_directory)
    os.makedirs(output_directory, exist_ok=True)
    total_counter = Counter()
    for input_file in input_files:
        path = Path(input_file)
        with open(path, mode='r') as f:
            text = f.read()
            lines = [line for line in text.splitlines() if line.strip()]
            temp_counter = Counter(lines)
            total_counter += temp_counter
        
    for input_file in input_files:
        path = Path(input_file)
        with open(path, mode='r') as f:
            text = f.read()
            lines = [line for line in text.splitlines() if line.strip()]
            unique_lines = [line for line in lines if total_counter[line] == 1]
            if unique_lines:
                output_text = '\n'.join(unique_lines) + '\n'
            else:
                output_text = ''
            file_name = path.name
            target_path = output_directory / file_name
            with open(target_path, mode='w') as out_f:
                out_f.write(output_text)

def minhash_deduplication(input_paths:list,
                          output_directory,
                          jaccard_threshold,
                          num_hashes,
                          num_bands,
                          n_gram_length):
    def normalize_text(input_text:str):
        text = input_text

        # 1. lowercasing
        text = text.lower()

        # 2. removing punctuation
        text = re.sub(r'[^\w\s]', '', text)

        # 3. normalizing whitespaces
        text = re.sub(r'\s+', ' ', text)

        # 4. removing accents
        text = unicodedata.normalize('NFD', text)
        text = ''.join(char for char in text if not unicodedata.combining(char))

        return text
    
    def generate_n_grams(text, n):
        words = text.split()
        if len(words) < n:
            n_grams = [text]
        else:
            n_grams = [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
        return n_grams
    
    def create_hash_functions(num_hashes) -> list:
        max_hash = 2**31 - 1
        hash_funcs = []
        used = set()
        while len(hash_funcs) < num_hashes:
            a = random.randint(1, max_hash - 1)
            b = random.randint(0, max_hash - 1)
            if (a, b) in used:
                continue
            used.add((a, b))
            hash_funcs.append(lambda x, a=a, b=b: (a * x + b) % max_hash)
        return hash_funcs
    
    def hash_signature(n_grams:list, hash_funcs:list):
        # n-grams S = {s1,...,sn}
        # min_hash(hi,S) = min(hi(s1),hi(s2),...,hi(sn))
        signature = []
        if (not n_grams):
            return signature
        for hash_func in hash_funcs:
            min_hash = min(hash_func(int(hashlib.md5(n_gram_i.encode()).hexdigest(), 16)) for n_gram_i in n_grams)
            signature.append(min_hash)
        return signature
    
    def get_hash_signature_from_text(input_text:str, hash_funcs:list) -> list:
        normalized_text = normalize_text(input_text)
        n_grams = generate_n_grams(normalized_text, n_gram_length)
        min_hash_signature = hash_signature(n_grams, hash_funcs)
        return min_hash_signature
    
    def split_signature_into_bands(signature:list, num_bands:int) -> list:
        band_size = len(signature) // num_bands
        bands = []
        for i in range(num_bands):
            start = i * band_size
            end = start + band_size
            bands.append(signature[start:end])
        return bands
    
    def potential_similar(signatures_bands, num_bands):
        candidate_pairs = set()
        for band_idx in range(num_bands):
            buckets = defaultdict(list)
            for doc_idx, bands in enumerate(signatures_bands):
                band = tuple(bands[band_idx])  # Use tuple for hashability
                buckets[band].append(doc_idx)
            for bucket in buckets.values():
                if len(bucket) > 1:
                    for pair in combinations(bucket, 2):
                        candidate_pairs.add(tuple(sorted(pair)))
        return list(candidate_pairs)
    
    def jaccard_similarity(text_a:str, text_b:str, n:int):
        text_a = normalize_text(text_a)
        text_b = normalize_text(text_b)
        n_grams_a = generate_n_grams(text_a, n)
        n_grams_b = generate_n_grams(text_b, n)
        a = set(n_grams_a)
        b = set(n_grams_b)
        if not a and not b:
            return 1.0 # Both empty
        if not a or not b:
            return 0.0 # One empty, one not
        intersection = a & b
        union = a | b
        return len(intersection) / len(union)

    assert num_hashes % num_bands == 0, "num_hashes must be divisible by num_bands"
    hash_funcs = create_hash_functions(num_hashes)
    signatures = []
    for file_path in input_paths:
        with open(file_path, 'r') as f:
            text = f.read()
        min_hash_signature = get_hash_signature_from_text(text, hash_funcs)
        signatures.append(min_hash_signature)

    for i in range(len(signatures)):
        signatures[i] = split_signature_into_bands(signatures[i], num_bands)

    potential_list = potential_similar(signatures, num_bands)

    similar_pairs = []
    for pair in potential_list:
        a_idx, b_idx = pair
        with open(input_paths[a_idx], 'r') as f_a:
            text_a = f_a.read()
        with open(input_paths[b_idx], 'r') as f_b:
            text_b = f_b.read()
        score = jaccard_similarity(text_a, text_b, n_gram_length)
        if(score > jaccard_threshold):
            similar_pairs.append(tuple([a_idx,b_idx]))

    output_index_list = [i for i in range(len(input_paths))]
    for pair in similar_pairs:
        a, b = pair
        if a in output_index_list and b in output_index_list:
            output_index_list.remove(b)

    os.makedirs(output_directory, exist_ok=True)
    for idx, path in enumerate(input_paths):
        filename = Path(path).name
        target_path = Path(output_directory) / filename
        if idx in output_index_list:
            shutil.copy2(Path(path), target_path)

#===================================================================
_MINHASH_MAX_HASH = 2**31 - 1
_MINHASH_SIGNATURE_READER = None
_MINHASH_SIGNATURE_HASH_PARAMS = None
_MINHASH_SIGNATURE_NGRAM_LENGTH = None
_MINHASH_VERIFY_READER = None
_MINHASH_VERIFY_OFFSETS = None
_MINHASH_VERIFY_BANDED_SIGNATURES = None
_MINHASH_VERIFY_BAND_BUCKETS = None
_MINHASH_VERIFY_NGRAM_LENGTH = None
_MINHASH_VERIFY_THRESHOLD = None

def _normalize_text_for_minhash(input_text: str) -> str:
    text = input_text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = unicodedata.normalize('NFD', text)
    text = ''.join(char for char in text if not unicodedata.combining(char))
    return text.strip()

def _generate_n_grams_for_minhash(text: str, n: int) -> list[str]:
    words = text.split()
    if len(words) < n:
        return [text] if text else []
    return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]

def _create_minhash_hash_params(num_hashes: int) -> list[tuple[int, int]]:
    hash_params = []
    used = set()
    while len(hash_params) < num_hashes:
        a = random.randint(1, _MINHASH_MAX_HASH - 1)
        b = random.randint(0, _MINHASH_MAX_HASH - 1)
        if (a, b) in used:
            continue
        used.add((a, b))
        hash_params.append((a, b))
    return hash_params

def _hash_signature_for_minhash(n_grams: list[str], hash_params: list[tuple[int, int]]) -> list[int]:
    signature = []
    if not n_grams:
        return signature

    n_gram_hashes = [
        int.from_bytes(hashlib.md5(n_gram.encode('utf-8')).digest(), 'big')
        for n_gram in n_grams
    ]

    for a, b in hash_params:
        min_hash = min((a * n_gram_hash + b) % _MINHASH_MAX_HASH for n_gram_hash in n_gram_hashes)
        signature.append(min_hash)
    return signature

def _get_minhash_signature_from_text(input_text: str, hash_params: list[tuple[int, int]], n_gram_length: int) -> list[int]:
    normalized_text = _normalize_text_for_minhash(input_text)
    n_grams = _generate_n_grams_for_minhash(normalized_text, n_gram_length)
    return _hash_signature_for_minhash(n_grams, hash_params)

def _split_signature_into_bands_for_minhash(signature: list[int], num_bands: int) -> list[list[int]]:
    band_size = len(signature) // num_bands
    bands = []
    for i in range(num_bands):
        start = i * band_size
        end = start + band_size
        bands.append(signature[start:end])
    return bands

def _jaccard_similarity_for_minhash(text_a: str, text_b: str, n: int) -> float:
    normalized_a = _normalize_text_for_minhash(text_a)
    normalized_b = _normalize_text_for_minhash(text_b)
    n_grams_a = set(_generate_n_grams_for_minhash(normalized_a, n))
    n_grams_b = set(_generate_n_grams_for_minhash(normalized_b, n))
    if not n_grams_a and not n_grams_b:
        return 1.0
    if not n_grams_a or not n_grams_b:
        return 0.0
    intersection = n_grams_a & n_grams_b
    union = n_grams_a | n_grams_b
    return len(intersection) / len(union)

def _iter_document_offsets_for_minhash(file_path, delimiter, chunk_size, show_progress=False):
    delimiter_bytes = delimiter.encode('utf-8')
    delimiter_len = len(delimiter_bytes)
    buffer = b''
    doc_start = 0
    progress_bar = None

    if show_progress:
        progress_bar = tqdm(
            total=Path(file_path).stat().st_size,
            unit='B',
            unit_scale=True,
            desc='Indexing documents',
        )

    with open(file_path, 'rb') as reader:
        while True:
            chunk = reader.read(chunk_size)
            if not chunk:
                break

            if progress_bar is not None:
                progress_bar.update(len(chunk))

            buffer += chunk

            while True:
                boundary = buffer.find(delimiter_bytes)
                if boundary == -1:
                    break

                doc_bytes = buffer[:boundary]
                doc_end = doc_start + boundary
                if doc_bytes.strip():
                    yield doc_start, doc_end

                buffer = buffer[boundary + delimiter_len:]
                doc_start = doc_end + delimiter_len

        if buffer.strip():
            yield doc_start, doc_start + len(buffer)

    if progress_bar is not None:
        progress_bar.close()

def _read_text_by_offset(reader, start: int, end: int) -> str:
    reader.seek(start)
    return reader.read(end - start).decode('utf-8', errors='ignore')

def _init_minhash_signature_worker(input_file: str, hash_params: list[tuple[int, int]], n_gram_length: int):
    global _MINHASH_SIGNATURE_READER
    global _MINHASH_SIGNATURE_HASH_PARAMS
    global _MINHASH_SIGNATURE_NGRAM_LENGTH

    if _MINHASH_SIGNATURE_READER is not None:
        _MINHASH_SIGNATURE_READER.close()

    _MINHASH_SIGNATURE_READER = open(input_file, 'rb')
    _MINHASH_SIGNATURE_HASH_PARAMS = hash_params
    _MINHASH_SIGNATURE_NGRAM_LENGTH = n_gram_length

def _compute_minhash_signature_for_offset(task: tuple[int, int, int]) -> tuple[int, list[int]]:
    doc_idx, start, end = task
    text = _read_text_by_offset(_MINHASH_SIGNATURE_READER, start, end)
    signature = _get_minhash_signature_from_text(
        text,
        _MINHASH_SIGNATURE_HASH_PARAMS,
        _MINHASH_SIGNATURE_NGRAM_LENGTH,
    )
    return doc_idx, signature

def _init_minhash_verify_worker(input_file: str):
    global _MINHASH_VERIFY_READER

    if _MINHASH_VERIFY_READER is not None:
        _MINHASH_VERIFY_READER.close()

    _MINHASH_VERIFY_READER = open(input_file, 'rb')

def _set_duplicate_bitmap_flag(bitmap, doc_idx: int):
    byte_index = doc_idx // 8
    bit_mask = 1 << (doc_idx % 8)
    current_value = bitmap[byte_index]
    bitmap[byte_index:byte_index + 1] = bytes([current_value | bit_mask])

def _merge_duplicate_bitmap_files(bitmap_paths: list[Path], merged_bitmap_path: Path, bitmap_bytes: int, chunk_size: int = 1024 * 1024):
    readers = [open(path, 'rb') for path in bitmap_paths]
    try:
        with open(merged_bitmap_path, 'wb') as writer:
            remaining = bitmap_bytes
            while remaining > 0:
                current_chunk_size = min(chunk_size, remaining)
                merged_chunk = bytearray(current_chunk_size)
                for reader in readers:
                    chunk = reader.read(current_chunk_size)
                    for idx, value in enumerate(chunk):
                        merged_chunk[idx] |= value
                writer.write(merged_chunk)
                remaining -= current_chunk_size
    finally:
        for reader in readers:
            reader.close()

def _verify_candidate_pairs_for_partition(task: tuple[int, int, str]) -> str:
    partition_id, partition_count, bitmap_path = task
    total_docs = len(_MINHASH_VERIFY_BANDED_SIGNATURES)
    bitmap_bytes = max(1, (len(_MINHASH_VERIFY_OFFSETS) + 7) // 8)

    with open(bitmap_path, 'r+b') as bitmap_file:
        bitmap = mmap.mmap(bitmap_file.fileno(), bitmap_bytes)
        try:
            for doc_idx in range(partition_id, total_docs, partition_count):
                bands = _MINHASH_VERIFY_BANDED_SIGNATURES[doc_idx]
                if not bands:
                    continue

                candidate_ids = set()
                for band_idx, band in enumerate(bands):
                    band_key = tuple(band)
                    for other_idx in _MINHASH_VERIFY_BAND_BUCKETS[band_idx].get(band_key, []):
                        if other_idx > doc_idx:
                            candidate_ids.add(other_idx)

                if not candidate_ids:
                    continue

                start_a, end_a = _MINHASH_VERIFY_OFFSETS[doc_idx]
                text_a = _read_text_by_offset(_MINHASH_VERIFY_READER, start_a, end_a)

                for other_idx in sorted(candidate_ids):
                    start_b, end_b = _MINHASH_VERIFY_OFFSETS[other_idx]
                    text_b = _read_text_by_offset(_MINHASH_VERIFY_READER, start_b, end_b)
                    score = _jaccard_similarity_for_minhash(text_a, text_b, _MINHASH_VERIFY_NGRAM_LENGTH)
                    if score > _MINHASH_VERIFY_THRESHOLD:
                        _set_duplicate_bitmap_flag(bitmap, other_idx)
        finally:
            bitmap.flush()
            bitmap.close()

    return bitmap_path

def _get_minhash_pool_context():
    try:
        return multiprocessing.get_context('fork')
    except ValueError:
        return multiprocessing.get_context()

def minhash_deduplication_one_file(
    input_file,
    output_file,
    jaccard_threshold=0.8,
    num_hashes=100,
    num_bands=20,
    n_gram_length=5,
    delimiter='<|endoftext|>',
    chunk_size=1024 * 1024,
    workers=None,
    signature_chunksize=32,
    verification_partitions=None,
):
    assert num_hashes % num_bands == 0, 'num_hashes must be divisible by num_bands'

    input_file = Path(input_file)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if workers is None:
        workers = max(1, min(4, multiprocessing.cpu_count() or 1))
    workers = max(1, workers)
    signature_chunksize = max(1, signature_chunksize)

    pool_context = _get_minhash_pool_context()
    hash_params = _create_minhash_hash_params(num_hashes)

    offsets = list(_iter_document_offsets_for_minhash(input_file, delimiter, chunk_size, show_progress=True))
    if not offsets:
        with open(output_file, 'wb'):
            pass
        return

    signatures = [None] * len(offsets)

    if workers == 1:
        with open(input_file, 'rb') as reader:
            for doc_idx, (start, end) in tqdm(
                enumerate(offsets),
                total=len(offsets),
                desc='MinHash signatures',
            ):
                text = _read_text_by_offset(reader, start, end)
                signatures[doc_idx] = _get_minhash_signature_from_text(text, hash_params, n_gram_length)
    else:
        tasks = ((doc_idx, start, end) for doc_idx, (start, end) in enumerate(offsets))
        with pool_context.Pool(
            processes=min(workers, len(offsets)),
            initializer=_init_minhash_signature_worker,
            initargs=(str(input_file), hash_params, n_gram_length),
        ) as pool:
            for doc_idx, signature in tqdm(
                pool.imap_unordered(_compute_minhash_signature_for_offset, tasks, chunksize=signature_chunksize),
                total=len(offsets),
                desc='MinHash signatures',
            ):
                signatures[doc_idx] = signature

    banded_signatures = [None] * len(signatures)
    for doc_idx, signature in tqdm(
        enumerate(signatures),
        total=len(signatures),
        desc='Preparing bands',
    ):
        if signature:
            banded_signatures[doc_idx] = _split_signature_into_bands_for_minhash(signature, num_bands)

    band_buckets = [defaultdict(list) for _ in range(num_bands)]
    for doc_idx, bands in tqdm(
        enumerate(banded_signatures),
        total=len(banded_signatures),
        desc='LSH buckets',
    ):
        if not bands:
            continue

        for band_idx, band in enumerate(bands):
            band_buckets[band_idx][tuple(band)].append(doc_idx)

    global _MINHASH_VERIFY_OFFSETS
    global _MINHASH_VERIFY_BANDED_SIGNATURES
    global _MINHASH_VERIFY_BAND_BUCKETS
    global _MINHASH_VERIFY_NGRAM_LENGTH
    global _MINHASH_VERIFY_THRESHOLD

    _MINHASH_VERIFY_OFFSETS = offsets
    _MINHASH_VERIFY_BANDED_SIGNATURES = banded_signatures
    _MINHASH_VERIFY_BAND_BUCKETS = band_buckets
    _MINHASH_VERIFY_NGRAM_LENGTH = n_gram_length
    _MINHASH_VERIFY_THRESHOLD = jaccard_threshold

    if verification_partitions is None:
        verification_partitions = max(1, min(len(offsets), workers * 4))
    verification_partitions = max(1, min(len(offsets), verification_partitions))

    bitmap_bytes = max(1, (len(offsets) + 7) // 8)
    with tempfile.TemporaryDirectory(prefix='minhash_dedup_', dir=str(output_file.parent)) as temp_dir:
        temp_dir_path = Path(temp_dir)
        bitmap_paths = []
        for partition_id in range(verification_partitions):
            bitmap_path = temp_dir_path / f'duplicates_{partition_id:05d}.bin'
            with open(bitmap_path, 'wb') as bitmap_file:
                bitmap_file.truncate(bitmap_bytes)
            bitmap_paths.append(bitmap_path)

        try:
            if workers == 1:
                _verify_candidate_pairs_for_partition((0, 1, str(bitmap_paths[0])))
            else:
                tasks = [
                    (partition_id, verification_partitions, str(bitmap_paths[partition_id]))
                    for partition_id in range(verification_partitions)
                ]
                with pool_context.Pool(
                    processes=min(workers, verification_partitions),
                    initializer=_init_minhash_verify_worker,
                    initargs=(str(input_file),),
                ) as pool:
                    for _ in tqdm(
                        pool.imap_unordered(_verify_candidate_pairs_for_partition, tasks),
                        total=len(tasks),
                        desc='Jaccard verification',
                    ):
                        pass
        finally:
            _MINHASH_VERIFY_OFFSETS = None
            _MINHASH_VERIFY_BANDED_SIGNATURES = None
            _MINHASH_VERIFY_BAND_BUCKETS = None
            _MINHASH_VERIFY_NGRAM_LENGTH = None
            _MINHASH_VERIFY_THRESHOLD = None

        merged_bitmap_path = temp_dir_path / 'duplicates_merged.bin'
        _merge_duplicate_bitmap_files(bitmap_paths, merged_bitmap_path, bitmap_bytes)

        delimiter_bytes = delimiter.encode('utf-8')
        with open(input_file, 'rb') as reader, open(output_file, 'wb') as writer, open(merged_bitmap_path, 'rb') as duplicate_reader:
            current_byte_index = -1
            current_bitmap_value = 0
            for doc_idx, (start, end) in tqdm(
                enumerate(offsets),
                total=len(offsets),
                desc='Writing output',
            ):
                byte_index = doc_idx // 8
                if byte_index != current_byte_index:
                    current_byte = duplicate_reader.read(1)
                    current_bitmap_value = current_byte[0] if current_byte else 0
                    current_byte_index = byte_index

                if current_bitmap_value & (1 << (doc_idx % 8)):
                    continue

                reader.seek(start)
                writer.write(reader.read(end - start))
                writer.write(delimiter_bytes)


# if __name__ == "__main__":
    # output = classify_quality('hello world')
    # print(output)