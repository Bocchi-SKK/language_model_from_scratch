import random
from fastwarc.warc import ArchiveIterator, WarcRecordType

try:
    from text_filter import *
    import file_paths
except:
    try:
        from cs336_data.text_filter import *
        from cs336_data import file_paths
    except:
        ImportError

warc_paths = file_paths.warc_paths
low_quality_data_path = file_paths.low_quality_txt_path

num_samples = 3500

all_texts = []

# 1. Load all extracted texts into memory
for warc_path in warc_paths:
    with open(warc_path, 'rb') as stream:
        for record in ArchiveIterator(stream):
            if record.record_type != WarcRecordType.response:
                continue

            try:
                html_bytes = record.reader.read()
                temp_text = extract_text_from_html_bytes(html_bytes).strip()
                if temp_text:
                    all_texts.append(temp_text)
            except Exception as e:
                print(f'SKIPPED RECORD: {e}')

# 2. Shuffle all texts in memory
print('start to shuffle')
random.shuffle(all_texts)
print('end shuffle')
print(f"all_texts_length = {len(all_texts)}")

# 3. Walk through shuffled texts and save only the accepted ones
count = 0

with open(low_quality_data_path, 'w', encoding='utf-8') as out_file:
    for temp_text in all_texts:
        try:
            lang, prob = identify_language(temp_text)
            if lang != 'en':
                continue

            if not gopher_quality_filter(temp_text):
                continue

            out_file.write(temp_text.replace('\n', ' ').strip() + '\n')
            count += 1
            print(f"count = {count}")

            if count >= num_samples:
                break

        except Exception as e:
            print(f'SKIPPED TEXT: {e}')

print(f'Total extracted texts loaded into memory: {len(all_texts)}')
print(f'Total saved texts: {count}')

if count < num_samples:
    print(f'Warning: only saved {count} texts, fewer than num_samples={num_samples}')