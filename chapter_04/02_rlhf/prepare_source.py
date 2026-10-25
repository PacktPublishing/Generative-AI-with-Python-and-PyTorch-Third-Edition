"""Extract whole Townsend fables from a frozen Gutenberg source, never paragraphs.

The raw download retains its header/footer/license. Rebuilding is offline by
 default; --download explicitly retrieves the one selected ebook. Outputs are
 raw source candidates only. Separately bundled authored_annotations.json holds
 summaries, headlines, and fact notes; this script never overwrites that file.
"""
import argparse
import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
URL = 'https://www.gutenberg.org/ebooks/21.txt.utf-8'
LANDING = 'https://www.gutenberg.org/ebooks/21'
RAW = ROOT / 'source' / 'pg21.txt'


def extract(text):
    # Downloads can use CRLF; normalize in memory, retaining raw archival bytes.
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    if any(marker not in text for marker in ('\nCONTENTS\n', '\nPREFACE\n', '\nAESOP’S FABLES\n')):
        raise ValueError('Unexpected Gutenberg edition: missing contents, preface, or fables boundary')
    contents = text.split('\nCONTENTS\n', 1)[1].split('\nPREFACE\n', 1)[0]
    title_lines = [line.strip() for line in contents.splitlines()]
    title_lines = title_lines[title_lines.index('AESOP’S FABLES') + 1:]
    titles = set(title_lines) - {'FOOTNOTES', 'INDEX'}
    body = text.split('\nAESOP’S FABLES\n', 1)[1].split('*** END OF THE PROJECT GUTENBERG EBOOK', 1)[0]
    body = body.split('\nFOOTNOTES\n', 1)[0].split('\nINDEX\n', 1)[0]
    positions = [match for match in re.finditer(r'^([^\n]+)$', body, re.M)
                 if match.group(1).strip() in titles]
    rows = []
    for index, match in enumerate(positions):
        end = positions[index + 1].start() if index + 1 < len(positions) else len(body)
        passage = body[match.end():end].strip()
        passage = re.sub(r'\s+', ' ', passage)
        title = match.group(1).strip()
        if not passage:
            continue
        rows.append({'id': f'pg21-{index:03d}', 'title': title,
                     'passage': passage, 'word_count': len(passage.split()),
                     'source_ebook': 21})
    if len(rows) < 250:
        raise ValueError('Extraction changed: verify edition/section boundaries')
    return rows


def prepare(download=False):
    RAW.parent.mkdir(exist_ok=True)
    previous_manifest = RAW.parent / 'manifest.json'
    try:
        previous = json.loads(previous_manifest.read_text(encoding='utf-8'))
        if not isinstance(previous, dict):
            previous = {}
    except (FileNotFoundError, ValueError):
        previous = {}
    retrieved_at = None
    if download:
        request = urllib.request.Request(URL, headers={'User-Agent': 'Aesop educational dataset source check'})
        with urllib.request.urlopen(request, timeout=60) as response:
            downloaded = response.read()
        # Check the edition before replacing a usable local source.
        extract(downloaded.decode('utf-8-sig'))
        RAW.write_bytes(downloaded)
        retrieved_at = datetime.now(timezone.utc).isoformat()
    if not RAW.exists():
        raise FileNotFoundError('Frozen source missing; run prepare_source.py --download')
    text = RAW.read_text(encoding='utf-8-sig')
    rows = extract(text)
    excluded = re.compile(r'\b(?:ethiop|aethiop|æthiop|negro|slave|slaves|slavery|faggots?|murder|killed|kill|slain|hanged|blood)\b', re.I)
    eligible = [row for row in rows if 60 <= row['word_count'] <= 180
                and not excluded.search(row['title'] + ' ' + row['passage'])]
    # All summaries of one source fable remain in one split. Record IDs do not
    # imply these sources were unseen during base-model pretraining.
    if len(eligible) < 80:
        raise ValueError('Insufficient eligible self-contained fables')
    for name, data in [('fables_extracted.json', rows), ('authoring_candidates.json', eligible[:80])]:
        (RAW.parent / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    source_sha256 = hashlib.sha256(RAW.read_bytes()).hexdigest()
    if not download and previous.get('source_sha256') == source_sha256:
        # Reuse retrieval metadata only for the same archived source bytes.
        recorded = previous.get('source_retrieved_at')
        retrieved_at = recorded if isinstance(recorded, str) and recorded.strip() else None
    manifest = {
        'ebook': LANDING, 'download_url': URL, 'title': 'Three hundred Aesop’s fables',
        'translator': 'George Fyler Townsend (1814–1900)',
        'source_sha256': source_sha256,
        'source_retrieved_at': retrieved_at,
        'source_retrieved': retrieved_at[:10] if retrieved_at else None,
        'candidate_records_sha256': hashlib.sha256((RAW.parent / 'authoring_candidates.json').read_bytes()).hexdigest(),
        'extracted_fables': len(rows),
        'candidate_fables': 80, 'selection': 'First 80 source-order fables, 60–180 words, exclusion screen; further AI screening required.',
    }
    (RAW.parent / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    print(json.dumps(prepare(args.download), ensure_ascii=False, indent=2))
