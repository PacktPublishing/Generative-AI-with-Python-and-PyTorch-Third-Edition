"""Build deterministic, AI-authored synthetic preference data offline."""
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'source' / 'authoring_candidates.json'
ANNOTATIONS = ROOT / 'source' / 'authored_annotations.json'
SPLITS = ('train', 'validation', 'test')
LABEL = 'AI-authored synthetic preference; not human-reviewed'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path, description, recovery):
    if not path.exists():
        raise FileNotFoundError(f'Missing {description}: {path}. {recovery}')
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f'Invalid JSON in {path}: {exc}') from exc


def _nonempty_string(value):
    return isinstance(value, str) and bool(value.strip())


def _validate():
    source = _load(SOURCE, 'raw authoring candidates', 'Run prepare_source.py --download to restore them.')
    annotations = _load(ANNOTATIONS, 'bundled AI-authored annotations',
                        'Restore source/authored_annotations.json; downloading cannot author annotations.')
    if not isinstance(source, list) or len(source) != 80:
        raise ValueError(f'{SOURCE} must be a JSON list of exactly 80 records')
    if not isinstance(annotations, list) or len(annotations) != 80:
        raise ValueError(f'{ANNOTATIONS} must be a JSON list of exactly 80 annotation records')
    ids = []
    passages = set()
    for i, row in enumerate(source):
        if not isinstance(row, dict):
            raise ValueError(f'{SOURCE} record {i} must be an object')
        if not all(_nonempty_string(row.get(k)) for k in ('id', 'title', 'passage')):
            raise ValueError(f'{SOURCE} record {i} needs nonempty id, title, and passage strings')
        if isinstance(row.get('source_ebook'), bool) or not isinstance(row.get('source_ebook'), int) or row['source_ebook'] <= 0:
            raise ValueError(f'{SOURCE} record {i} needs a positive integer source_ebook')
        if isinstance(row.get('word_count'), bool) or not isinstance(row.get('word_count'), int) or row['word_count'] < 0:
            raise ValueError(f'{SOURCE} record {row["id"]} needs a nonnegative integer word_count')
        if row['passage'] in passages:
            raise ValueError(f'{SOURCE} has duplicate passage for id {row["id"]}')
        ids.append(row['id']); passages.add(row['passage'])
    if len(set(ids)) != 80:
        raise ValueError(f'{SOURCE} must contain 80 unique IDs')
    amap = {}
    for i, row in enumerate(annotations):
        if not isinstance(row, dict) or not _nonempty_string(row.get('id')):
            raise ValueError(f'{ANNOTATIONS} record {i} must be an object with a nonempty id')
        if row['id'] in amap:
            raise ValueError(f'{ANNOTATIONS} contains duplicate id {row["id"]}')
        amap[row['id']] = row
    if set(amap) != set(ids):
        raise ValueError(f'{ANNOTATIONS} IDs must exactly match source IDs; missing={sorted(set(ids)-set(amap))}, extra={sorted(set(amap)-set(ids))}')
    for raw in source:
        ann = amap[raw['id']]
        expected = hashlib.sha256(raw['passage'].encode('utf-8')).hexdigest()
        if ann.get('passage_sha256') != expected:
            raise ValueError(f'Annotation {raw["id"]} passage_sha256 does not match its UTF-8 source passage')
        summary = ann.get('summary')
        if (not _nonempty_string(summary) or '\n' in summary or '\r' in summary
                or re.search(r'(?m)^\s*#{1,6}(?:\s|$)', summary)
                or re.search(r'<\|(?:im_|fim_|gh|system|user|assistant)', summary, re.I)
                or summary.lstrip().startswith(('```', '> '))):
            raise ValueError(f'Annotation {raw["id"]} summary must be a nonempty plain paragraph without chat markup or Markdown headings')
        headline = ann.get('headline')
        if (not _nonempty_string(headline) or '\n' in headline or '\r' in headline
                or any(mark in headline for mark in ('*', '#', '`', '<|'))
                or not 1 <= len(headline.split()) <= 12):
            raise ValueError(f'Annotation {raw["id"]} headline must be a plain single line of 1-12 words without *, #, or backticks')
        facts = ann.get('essential_facts')
        if not isinstance(facts, list) or len(facts) not in (2, 3) or any(not _nonempty_string(x) for x in facts):
            raise ValueError(f'Annotation {raw["id"]} essential_facts must contain exactly 2 or 3 nonempty strings')
        if not _nonempty_string(ann.get('author_model')):
            raise ValueError(f'Annotation {raw["id"]} author_model must be nonempty')
    return source, amap


def prepare():
    source, amap = _validate()  # Complete validation before creating or writing outputs.
    # Attribution is optional for standalone authoring fixtures, and only copied
    # when the source manifest identifies these exact raw candidate records.
    source_metadata = None
    source_manifest_path = SOURCE.parent / 'manifest.json'
    if source_manifest_path.exists():
        source_manifest = json.loads(source_manifest_path.read_text(encoding='utf-8'))
        if isinstance(source_manifest, dict) and source_manifest.get('candidate_records_sha256') == digest(SOURCE):
            source_metadata = {key: source_manifest.get(key) for key in
                               ('ebook', 'title', 'translator', 'source_sha256', 'source_retrieved_at')}
    splits = {name: [] for name in SPLITS}
    for index, raw in enumerate(source):
        split = 'train' if index < 48 else 'validation' if index < 60 else 'test'
        ann = amap[raw['id']]
        headline, body = ann['headline'].strip(), ann['summary'].strip()
        contrast = 'unbolded_headline' if len(splits[split]) % 2 == 0 else 'summary_only'
        row = {
            'id': raw['id'], 'title': raw['title'], 'passage': raw['passage'],
            'chosen': f'**{headline}**\n\n{body}',
            'rejected': f'{headline}\n\n{body}' if contrast == 'unbolded_headline' else body,
            'preference_reason': 'Format-only synthetic preference; the pair shares the same AI-authored summary and makes no factual superiority claim.',
            'rejection_type': contrast, 'essential_facts': ann['essential_facts'],
            'label_origin': LABEL, 'author_model': ann['author_model'], 'group_id': raw['id'],
        }
        splits[split].append(row)
    all_rows = [row for split in SPLITS for row in splits[split]]
    data = ROOT / 'data'
    data.mkdir(exist_ok=True)
    for split, rows in splits.items():
        (data / f'{split}.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (data / 'annotations.json').write_text(json.dumps(all_rows, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (data / 'headlines.json').write_text(json.dumps([{'id': r['id'], 'headline': amap[r['id']]['headline'], 'author_model': amap[r['id']]['author_model']} for r in source], ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    counts = {name: {kind: sum(r['rejection_type'] == kind for r in rows) for kind in ('unbolded_headline', 'summary_only')} for name, rows in splits.items()}
    manifest = {
        'format': 'headline-summary-preferences', 'label_origin': LABEL,
        'source_records_sha256': digest(SOURCE), 'authored_annotations_sha256': digest(ANNOTATIONS),
        'source_metadata': source_metadata,
        'split_sizes': {name: len(rows) for name, rows in splits.items()},
        'style_contrast_counts': counts, 'grouping': 'Source order; one pair per source ID; deterministic 48/12/20 split.',
        'provenance_notice': 'AI-authored synthetic format preferences; not human-reviewed.',
        'files': {},
    }
    for name in (*SPLITS, 'annotations', 'headlines'):
        path = data / f'{name}.json'
        manifest['files'][path.name] = {'rows': len(splits[name]) if name in splits else len(all_rows), 'sha256': digest(path)}
    (data / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    review_path = data / 'human_review_template.csv'
    if not review_path.exists() or review_path.stat().st_size == 0:
        with review_path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=['id', 'passage', 'candidate_A', 'candidate_B', 'human_choice_A_B_tie', 'reviewer', 'notes'])
            writer.writeheader()
            for row in all_rows:
                writer.writerow({'id': row['id'], 'passage': row['passage'], 'candidate_A': row['chosen'], 'candidate_B': row['rejected'], 'human_choice_A_B_tie': '', 'reviewer': '', 'notes': ''})
    return manifest


if __name__ == '__main__':
    print('Prepared splits:', prepare()['split_sizes'])
