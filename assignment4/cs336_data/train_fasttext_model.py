from pathlib import Path
import tempfile
import fasttext
import random
import re

DOC_SEPARATOR = '<|endoftext|>'
SEED = 42
VALIDATION_RATIO = 0.2
TARGET_CHUNK_WORDS = 200
MIN_CHUNK_WORDS = 100
MAX_CHUNKS_PER_DOC = 5

def get_label_from_file_path(input_path):
    label = re.sub(r'\W+', '_', Path(input_path).stem).strip('_').lower()
    if not label:
        raise ValueError(f'Could not derive a label from file path: {input_path}')
    return label


def load_documents(input_path, separator=DOC_SEPARATOR):
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    if separator in raw_text:
        raw_documents = raw_text.split(separator)
    else:
        raw_documents = raw_text.splitlines()

    documents = []
    for raw_document in raw_documents:
        normalized_text = re.sub(r'\s+', ' ', raw_document).strip()
        if normalized_text:
            documents.append(normalized_text)

    return documents


def split_into_chunks(
    text,
    target_chunk_words=TARGET_CHUNK_WORDS,
    min_chunk_words=MIN_CHUNK_WORDS,
    max_chunks_per_doc=MAX_CHUNKS_PER_DOC,
):
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
        chunks.append(' '.join(words[start:end]))
        start = end

    return chunks


def build_source_documents(texts, label):
    source_documents = []

    for index, text in enumerate(texts):
        source_documents.append(
            {
                'label': label,
                'source_id': f'{label}_{index}',
                'text': text,
            }
        )

    return source_documents


def split_source_documents(source_documents, validation_ratio=VALIDATION_RATIO, rng=None):
    if rng is None:
        rng = random.Random(SEED)

    shuffled_documents = source_documents[:]
    rng.shuffle(shuffled_documents)

    split_index = int(len(shuffled_documents) * (1 - validation_ratio))
    train_documents = shuffled_documents[:split_index]
    validation_documents = shuffled_documents[split_index:]
    return train_documents, validation_documents


def chunk_source_documents(source_documents):
    chunked_dataset = []

    for document in source_documents:
        chunks = split_into_chunks(document['text'])
        for chunk_index, chunk_text in enumerate(chunks):
            chunked_dataset.append(
                {
                    'label': document['label'],
                    'text': chunk_text,
                    'source_id': document['source_id'],
                    'chunk_id': chunk_index,
                }
            )

    return chunked_dataset


def balance_classes(dataset, labels, rng=None):
    if rng is None:
        rng = random.Random(SEED)

    grouped_examples = {label: [] for label in labels}
    for item in dataset:
        grouped_examples[item['label']].append(item)

    keep_n = min(len(grouped_examples[label]) for label in labels)
    if keep_n == 0:
        return []

    balanced_dataset = []
    for label in labels:
        rng.shuffle(grouped_examples[label])
        balanced_dataset.extend(grouped_examples[label][:keep_n])

    rng.shuffle(balanced_dataset)
    return balanced_dataset


def save_fasttext_format(dataset, path):
    with open(path, 'w', encoding='utf-8') as f:
        for item in dataset:
            label = f"__label__{item['label']}"
            text = item['text'].replace('\n', ' ')
            f.write(f"{label} {text}\n")


def build_train_val_dataset(first_input_path, second_input_path, validation_ratio=VALIDATION_RATIO):
    first_label = get_label_from_file_path(first_input_path)
    second_label = get_label_from_file_path(second_input_path)
    if first_label == second_label:
        raise ValueError('The two input files produce the same label. Rename one of the files or pass differently named files.')

    first_texts = load_documents(first_input_path)
    second_texts = load_documents(second_input_path)
    if not first_texts or not second_texts:
        raise ValueError('Both input files must contain at least one non-empty document.')

    rng = random.Random(SEED)

    first_source_documents = build_source_documents(first_texts, first_label)
    second_source_documents = build_source_documents(second_texts, second_label)

    first_train_docs, first_val_docs = split_source_documents(first_source_documents, validation_ratio, rng)
    second_train_docs, second_val_docs = split_source_documents(second_source_documents, validation_ratio, rng)

    train_dataset = chunk_source_documents(first_train_docs + second_train_docs)
    val_dataset = chunk_source_documents(first_val_docs + second_val_docs)

    labels = [first_label, second_label]
    train_dataset = balance_classes(train_dataset, labels, rng)
    val_dataset = balance_classes(val_dataset, labels, rng)
    if not train_dataset or not val_dataset:
        raise ValueError('Training or validation dataset is empty after chunking and balancing. Increase input data or adjust chunk sizes.')

    return train_dataset, val_dataset, labels


def train_model(
    first_input_path,
    second_input_path,
    model_output_path,
    validation_ratio=VALIDATION_RATIO,
    lr=0.1,
    epoch=30,
    word_ngrams=2,
    min_count=1,
    loss='softmax',
):
    train_dataset, val_dataset, labels = build_train_val_dataset(
        first_input_path=first_input_path,
        second_input_path=second_input_path,
        validation_ratio=validation_ratio,
    )

    model_output_path = Path(model_output_path)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='fasttext_train_') as temp_dir:
        train_path = Path(temp_dir) / 'train_fasttext.txt'
        val_path = Path(temp_dir) / 'val_fasttext.txt'
        save_fasttext_format(train_dataset, train_path)
        save_fasttext_format(val_dataset, val_path)

        model = fasttext.train_supervised(
            input=str(train_path),
            lr=lr,
            epoch=epoch,
            wordNgrams=word_ngrams,
            verbose=2,
            minCount=min_count,
            loss=loss,
        )

        model.save_model(str(model_output_path))
        result = model.test(str(val_path))

    print('='*50)
    print(f'labels: {labels[0]} vs {labels[1]}')
    print(result)  # (num_examples, precision, recall)
    return model, result