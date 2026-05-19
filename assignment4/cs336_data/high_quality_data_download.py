try:
    from text_filter import *
    import file_paths
except:
    try:
        from cs336_data.text_filter import *
        from cs336_data import file_paths
    except:
        ImportError

import random
import requests

urls_path = file_paths.urls_path
high_path = file_paths.high_quality_txt_path

num_text_download = 3500
temp_text = ''

def load_all_urls(file_path: str) -> list[str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

all_urls = load_all_urls(urls_path)
random.shuffle(all_urls)

saved_count = 0
checked_count = 0

with open(high_path, 'w', encoding='utf-8') as out_file:
    for url in all_urls:
        if saved_count >= num_text_download:
            break

        try:
            response = requests.get(
                url,
                timeout=5,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            response.raise_for_status()

            temp_text = extract_text_from_html_bytes(response.content).strip()
            checked_count += 1

            if not temp_text:
                continue

            lang, prob = identify_language(temp_text)
            if lang != 'en':
                continue

            if not gopher_quality_filter(temp_text):
                continue

            out_file.write(temp_text.replace('\n', ' ').strip() + '\n')
            saved_count += 1

            print(f'SAVED {saved_count}: {url}')

        except Exception as e:
            print(f'SKIPPED: {url} | {e}')

if saved_count < num_text_download:
    raise RuntimeError(
        f'Only saved {saved_count} texts after checking {checked_count} URLs. '
        f'Need {num_text_download}. Try more source URLs or relax the filters.'
    )

print(f'Checked: {checked_count}')
print(f'Saved: {saved_count}')