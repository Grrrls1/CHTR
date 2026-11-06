"""Audit all raw POT records and freeze writer splits plus a random-access training index."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.datasets.casia_pot import iter_pot
from models.datasets.casia_training import SCHEMA, DEFAULT_RENDER, sha256_file


def prepare(root, output, seed=42, val_writers=24):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('output must be new; preserve existing dataset versions')
    sources, files = [], {}
    for split in ('train', 'test'):
        files[split] = sorted((root/'1.1'/split).rglob('*.pot'))
        if not files[split]:
            raise ValueError('missing official '+split+' POT files')
        names = [p.stem for p in files[split]]
        if len(set(names)) != len(names):
            raise ValueError('repeated writer files; cannot infer writer identity')
    train_writers = sorted(p.stem for p in files['train'])
    test_writers = {p.stem for p in files['test']}
    if set(train_writers) & test_writers:
        raise ValueError('official train/test writer overlap')
    if not 0 < val_writers < len(train_writers):
        raise ValueError('val_writers must leave nonempty train and val')
    selected = set(np.random.default_rng(seed).choice(train_writers, val_writers, replace=False).tolist())
    output.mkdir(parents=True)
    db = sqlite3.connect(output/'index.sqlite')
    db.execute('CREATE TABLE samples (sid TEXT PRIMARY KEY, character TEXT, writer TEXT, source TEXT, record_index INTEGER, byte_offset INTEGER, lengths TEXT, split TEXT)')
    stats = {s: Counter() for s in ('train','val','test')}
    histograms = {s: Counter() for s in stats}
    ids = {s: [] for s in stats}
    began = time.monotonic()
    try:
        for official, paths in files.items():
            for p in paths:
                split = 'test' if official == 'test' else 'val' if p.stem in selected else 'train'
                source = p.relative_to(root).as_posix()
                sources.append(dict(path=source, sha256=sha256_file(p), bytes=p.stat().st_size, official_split=official, writer_id=p.stem))
                for sample in iter_pot(p):
                    st = stats[split]
                    st['raw_records'] += 1
                    if sample['label_error']:
                        # Undecodable labels cannot be silently dropped from a Chinese benchmark.
                        raise ValueError(f'{p}:{sample["index"]}: {sample["label_error"]}')
                    if not '\u3400' <= sample['char'] <= '\u9fff':
                        st['excluded_non_chinese'] += 1
                        continue
                    paths_xy = sample['strokes']
                    lengths = [len(a) for a in paths_xy]
                    sid = f'{official}/{p.stem}/{sample["index"]:06d}'
                    db.execute('INSERT INTO samples VALUES (?,?,?,?,?,?,?,?)',
                               (sid, sample['char'], p.stem, source, sample['index'], sample['byte_offset'], json.dumps(lengths), split))
                    ids[split].append(sid)
                    st['characters'] += 1; st['strokes'] += len(lengths); st['points'] += sum(lengths)
                    st['max_strokes'] = max(st['max_strokes'],len(lengths))
                    st['max_stroke_points'] = max(st['max_stroke_points'],max(lengths))
                    st['max_character_points'] = max(st['max_character_points'],sum(lengths))
                    st['single_point_strokes'] += sum(n == 1 for n in lengths)
                    st['consecutive_duplicate_points'] += sum(int((np.diff(a,axis=0)==0).all(1).sum()) for a in paths_xy)
                    st['strokes_over_2048_points'] += sum(n>2048 for n in lengths)
                    st['characters_over_64_strokes'] += int(len(lengths)>64)
                    histograms[split].update(lengths)
                db.commit()
                print(json.dumps(dict(file=source, split=split, characters=len(ids[split]), seconds=round(time.monotonic()-began,1))), flush=True)
        db.execute('CREATE INDEX split_index ON samples(split,sid)')
        db.commit()
    finally:
        db.close()
    protocol = dict(DEFAULT_RENDER, seed=seed)
    index_hash = sha256_file(output/'index.sqlite')
    payload = dict(schema=SCHEMA, sources=sources, index_sha256=index_hash, render=protocol,
                   split_seed=seed, val_writer_count=val_writers,
                   **{s+'_ids':sorted(v) for s,v in ids.items()})
    payload['dataset_fingerprint'] = hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    payload['index_file'] = 'index.sqlite'
    (output/'split.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    report = dict(schema=SCHEMA, dataset_fingerprint=payload['dataset_fingerprint'],
                  writer_counts=dict(train=len(train_writers)-val_writers,val=val_writers,test=len(test_writers)),
                  statistics=stats, stroke_length_histograms={s:dict(sorted(h.items())) for s,h in histograms.items()},
                  render=protocol, point_policy='all original XY, duplicates and stroke breaks preserved; no resampling/truncation',
                  character_filter='U+3400..U+9FFF (same as local external CASIA evaluator)',
                  test_used_for_parameter_selection=False, seconds=time.monotonic()-began)
    (output/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pot-root',type=Path,default=Path('data/CASIA-OLHWDB'))
    p.add_argument('--output',type=Path,default=Path('data/CASIA-OLHWDB/training_v1'))
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--val-writers',type=int,default=24)
    a=p.parse_args()
    prepare(a.pot_root,a.output,a.seed,a.val_writers)


if __name__ == '__main__':
    main()
