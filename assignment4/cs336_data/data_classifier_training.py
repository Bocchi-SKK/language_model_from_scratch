import random
import re
import fasttext

try:
    import file_paths
except:
    try:
        from cs336_data import file_paths
    except:
        ImportError

high_quality_path = file_paths.high_quality_txt_path
low_quality_path = file_paths.low_quality_txt_path
train_path = file_paths.train_path
val_path = file_paths.val_path
model_path = file_paths.quality_classifier_model_path

SEED = 42
TARGET_CHUNK_WORDS = 200
MIN_CHUNK_WORDS = 100
MAX_CHUNKS_PER_DOC = 5
VALIDATION_RATIO = 0.2

random.seed(SEED)

def load_texts(path: str) -> list[str]:
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def normalize_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def split_into_chunks(
    text: str,
    target_chunk_words: int = TARGET_CHUNK_WORDS,
    min_chunk_words: int = MIN_CHUNK_WORDS,
    max_chunks_per_doc: int = MAX_CHUNKS_PER_DOC,
) -> list[str]:
    words = text.split()
    if len(words) < min_chunk_words:
        return []

    chunks = []
    start = 0

    while start < len(words) and len(chunks) < max_chunks_per_doc:
        remaining = len(words) - start

        if remaining <= target_chunk_words:
            if remaining >= min_chunk_words:
                chunks.append(' '.join(words[start:]))
            break

        end = start + target_chunk_words
        chunk_words = words[start:end]
        chunks.append(' '.join(chunk_words))
        start = end

    return chunks


def build_source_documents(texts: list[str], label: str) -> list[dict]:
    source_documents = []

    for index, raw_text in enumerate(texts):
        normalized_text = normalize_text(raw_text)
        if not normalized_text:
            continue

        source_documents.append(
            {
                'label': label,
                'source_id': f'{label}_{index}',
                'text': normalized_text,
            }
        )

    return source_documents


def split_source_documents(
    source_documents: list[dict],
    validation_ratio: float = VALIDATION_RATIO,
) -> tuple[list[dict], list[dict]]:
    shuffled = source_documents[:]
    random.shuffle(shuffled)

    split_index = int(len(shuffled) * (1 - validation_ratio))
    train_docs = shuffled[:split_index]
    val_docs = shuffled[split_index:]

    return train_docs, val_docs


def chunk_source_documents(source_documents: list[dict]) -> list[dict]:
    chunked_dataset = []

    for doc in source_documents:
        chunks = split_into_chunks(doc['text'])

        for chunk_id, chunk_text in enumerate(chunks):
            chunked_dataset.append(
                {
                    'label': doc['label'],
                    'text': chunk_text,
                    'source_id': doc['source_id'],
                    'chunk_id': chunk_id,
                }
            )

    return chunked_dataset


def balance_classes(dataset: list[dict]) -> list[dict]:
    wiki_examples = [item for item in dataset if item['label'] == 'wiki']
    cc_examples = [item for item in dataset if item['label'] == 'cc']

    random.shuffle(wiki_examples)
    random.shuffle(cc_examples)

    keep_n = min(len(wiki_examples), len(cc_examples))

    balanced = wiki_examples[:keep_n] + cc_examples[:keep_n]
    random.shuffle(balanced)
    return balanced


def average_words(dataset: list[dict]) -> float:
    if not dataset:
        return 0.0
    return sum(len(item['text'].split()) for item in dataset) / len(dataset)

def get_train_val_dataset(high_path, low_path):
    '''
    all_dataset has the structure:
    [
        {
            'label': 'high' or 'low',
            'text': 'chunked text here',
            'source_id': 'high_1',
            'chunk_id': 0,
        },
        ...
    ]
    '''
    high_texts = load_texts(high_path)
    random.shuffle(high_texts)
    high_texts = high_texts[:]
    low_texts = load_texts(low_path)

    high_source_docs = build_source_documents(high_texts, 'wiki')
    low_source_docs = build_source_documents(low_texts, 'cc')

    high_train_docs, high_val_docs = split_source_documents(high_source_docs)
    low_train_docs, low_val_docs = split_source_documents(low_source_docs)

    train_source_docs = high_train_docs + low_train_docs
    val_source_docs = high_val_docs + low_val_docs

    train_dataset = chunk_source_documents(train_source_docs)
    val_dataset = chunk_source_documents(val_source_docs)

    train_dataset = balance_classes(train_dataset)
    val_dataset = balance_classes(val_dataset)

    all_dataset = train_dataset + val_dataset
    random.shuffle(train_dataset)
    random.shuffle(val_dataset)
    random.shuffle(all_dataset)

    # print(f'high source docs: {len(high_source_docs)}')
    # print(f'low source docs: {len(low_source_docs)}')
    # print(f'train chunks: {len(train_dataset)}')
    # print(f'validation chunks: {len(val_dataset)}')
    # print(f'all chunks: {len(all_dataset)}')
    # print(f'train avg words per chunk: {average_words(train_dataset):.2f}')
    # print(f'validation avg words per chunk: {average_words(val_dataset):.2f}')

    return train_dataset, val_dataset

def save_fasttext_format(dataset, path):
    with open(path, 'w', encoding='utf-8') as f:
        for item in dataset:
            label = f"__label__{item['label']}"
            text = item['text'].replace('\n', ' ')
            f.write(f"{label} {text}\n")

def train_model(high_quality_path, low_quality_path, model_path):
    train_dataset, val_dataset = get_train_val_dataset(high_path=high_quality_path, low_path=low_quality_path)
    save_fasttext_format(train_dataset, train_path)
    save_fasttext_format(val_dataset, val_path)

    model = fasttext.train_supervised(
        input=str(train_path),
        lr=0.1,
        epoch=30,
        wordNgrams=2,
        verbose=2,
        minCount=1,
        loss='softmax'
    )

    model.save_model(str(model_path))

    result = model.test(str(val_path))
    print('='*50)
    print('test results')
    print(result)  # (num_examples, precision, recall)
    print("(num_examples, precision, recall)")

train_model(high_quality_path, low_quality_path, model_path)