"""Select one valid native 24-class RHSEDB reference per Unicode character."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from common import masks_and_labels, require, write_json, write_records


def character_of(item):
    split, path = item
    with np.load(path, allow_pickle=False) as z:
        character = str(z['name'])
    require(len(character) == 1, 'invalid character in ' + str(path))
    return split, path, character


def valid_reference(path, character):
    with np.load(path, allow_pickle=False) as z:
        require(str(z['name']) == character, 'character changed')
        masks_and_labels(z['reference_single_image'], z['stroke_label'])
        return len(z['stroke_label'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rhsedb', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=12)
    args = p.parse_args()
    require(args.workers > 0, 'workers must be positive')
    root, out = Path(args.rhsedb).resolve(), Path(args.output).resolve()
    require(not out.exists() or not any(out.iterdir()), 'output must be new or empty')
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(split, path) for split in ('train', 'test')
            for path in sorted((root / split).glob('*.npz'), key=lambda p: int(p.stem))]
    require(jobs, 'no RHSEDB files')
    candidates = defaultdict(list)
    counts = Counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (split, path, character) in enumerate(pool.map(character_of, jobs), 1):
            candidates[character].append((split, path))
            counts[split] += 1
            if i % 5000 == 0:
                print('indexed %d/%d' % (i, len(jobs)), flush=True)
    rows, rejected = [], []
    selected_splits = Counter()
    for i, character in enumerate(sorted(candidates), 1):
        for split, path in candidates[character]:
            try:
                n = valid_reference(path, character)
                rows.append(dict(character=character, npz_path=str(path), source='RHSEDB/%s/%s' % (split, path.name),
                                 source_split=split, candidate_count=len(candidates[character]),
                                 stroke_count=n, category_system='StrokeExtraction-24'))
                selected_splits[split] += 1
                break
            except Exception as exc:
                rejected.append(dict(character=character, path=str(path), error=str(exc)))
        else:
            rejected.append(dict(character=character, error='no valid reference among candidates'))
        if i % 500 == 0:
            print('validated %d/%d characters' % (i, len(candidates)), flush=True)
    write_records(out / 'manifest.jsonl', rows)
    write_records(out / 'rejected.jsonl', rejected)
    write_json(out / 'summary.json', dict(source_files=len(jobs), source_by_split=dict(counts),
               unique_characters=len(candidates), selected=len(rows), selected_by_split=dict(selected_splits),
               rejected_candidates=len(rejected), characters_without_valid_reference=len(candidates)-len(rows),
               selection='train before test; ascending numeric source file ID; first valid native 24-class reference'))
    if len(rows) != len(candidates):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
