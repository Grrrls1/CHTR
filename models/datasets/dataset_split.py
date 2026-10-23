"""Deterministic character-stratified splits; no dependency on torch."""
import hashlib
import json
import random
from collections import defaultdict

NAMES = ('train', 'val', 'test')


def fingerprint(rows):
    payload = json.dumps(sorted(rows, key=lambda r: r['sample_id']), sort_keys=True,
                         ensure_ascii=False, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()


def create_split(rows, groups, ratios=(.8, .1, .1), seed=42):
    if len(ratios) != 3 or any(not 0 < x < 1 for x in ratios) or abs(sum(ratios)-1) > 1e-8:
        raise ValueError('three positive ratios must sum to one')
    lookup = {r['sample_id']: r for r in rows}
    if len(lookup) != len(rows) or not rows:
        raise ValueError('requires nonempty unique sample IDs')
    flat = [sid for g in groups for sid in g]
    if len(flat) != len(set(flat)) or set(flat) != set(lookup) or any(not g for g in groups):
        raise ValueError('groups must partition candidate IDs')
    by_char = defaultdict(list)
    for i, group in enumerate(groups):
        chars = {lookup[s]['character'] for s in group}
        if any(not isinstance(c, str) or not c.strip() for c in chars):
            raise ValueError('missing character label')
        for c in chars:
            by_char[c].append(i)
    rare = {c for c, indexes in by_char.items() if len(indexes) < 3}
    forced = {i for c in rare for i in by_char[c]}
    assigned = {i: 'train' for i in forced}
    # Multi-character duplicate components stay together, conservatively in train.
    cross = {i for i,g in enumerate(groups) if len({lookup[s]['character'] for s in g}) > 1}
    assigned.update({i: 'train' for i in cross})
    rng = random.Random(seed)
    for c in sorted(by_char):
        indexes = by_char[c]
        available = sorted(i for i in indexes if i not in assigned)
        rng.shuffle(available)
        available.sort(key=lambda i: -len(groups[i]))
        counts = {name: sum(sum(lookup[s]['character'] == c for s in groups[i])
                            for i in indexes if assigned.get(i) == name) for name in NAMES}
        total = sum(sum(lookup[s]['character'] == c for s in groups[i]) for i in indexes)
        if c in rare:
            continue
        for pos, i in enumerate(available):
            missing = [name for name in NAMES if counts[name] == 0]
            if len(available)-pos <= len(missing):
                name = missing[0]
            else:
                name = max(NAMES, key=lambda n: total*ratios[NAMES.index(n)]-counts[n])
            assigned[i] = name
            counts[name] += len(groups[i])
    result = {name+'_ids': sorted(s for i,g in enumerate(groups) if assigned[i] == name for s in g)
              for name in NAMES}
    reports = {}
    sets = {n: set(result[n+'_ids']) for n in NAMES}
    for c, indexes in sorted(by_char.items()):
        counts = {n: sum(s in sets[n] for i in indexes for s in groups[i] if lookup[s]['character']==c) for n in NAMES}
        reason = None if all(counts.values()) else ('fewer_than_three_independent_groups' if c in rare else 'cross_character_group_constraints')
        reports[c] = dict(samples=counts, independent_groups=len(indexes), coverage_exception=reason)
    result.update(schema='chtr-character-split-v1', seed=seed, ratios=list(ratios),
                  dataset_fingerprint=fingerprint(rows), fingerprint_scope='candidate metadata (including stored asset hashes), not full asset integrity',
                  groups=groups, characters=reports,
                  summary={n: dict(samples=len(sets[n]), strokes=sum(r['stroke_count'] for r in rows if r['sample_id'] in sets[n])) for n in NAMES})
    validate_split(result, rows)
    return result


def validate_split(split, rows):
    if split.get('schema') != 'chtr-character-split-v1':
        raise ValueError('unsupported split schema')
    if split['dataset_fingerprint'] != fingerprint(rows):
        raise ValueError('dataset fingerprint mismatch')
    expected = {r['sample_id'] for r in rows}
    if len(expected) != len(rows):
        raise ValueError('duplicate candidate IDs')
    seen = set(); membership = {}
    for name in NAMES:
        ids = split[name+'_ids']
        if not ids or len(ids) != len(set(ids)) or seen.intersection(ids) or not set(ids) <= expected:
            raise ValueError('empty split, duplicate, overlap or unknown IDs')
        seen.update(ids); membership.update({s:name for s in ids})
    if seen != expected:
        raise ValueError('incomplete candidate coverage')
    flat = [s for g in split['groups'] for s in g]
    if len(flat) != len(set(flat)) or set(flat) != expected:
        raise ValueError('invalid group coverage')
    for group in split['groups']:
        if not group or len({membership[s] for s in group}) != 1:
            raise ValueError('group crosses splits')
    labels = {r['sample_id']: r['character'] for r in rows}
    char_groups = defaultdict(list)
    cross_chars = set()
    for group in split['groups']:
        chars = {labels[s] for s in group}
        if len(chars) > 1:
            cross_chars.update(chars)
        for c in chars:
            char_groups[c].append(group)
    by_char = defaultdict(list)
    for row in rows:
        by_char[row['character']].append(row['sample_id'])
    for c, ids in by_char.items():
        count = {n: sum(membership[s] == n for s in ids) for n in NAMES}
        report = split['characters'][c]
        if report['samples'] != count:
            raise ValueError('character report differs from membership')
        relevant = char_groups[c]
        if len(relevant) < 3 and count['train'] != len(ids):
            raise ValueError('rare character must remain in train')
        if len(relevant) >= 3 and c not in cross_chars:
            if not all(count.values()):
                raise ValueError('eligible character missing a split')


def load_split_ids(path, rows, name):
    if name not in NAMES:
        raise ValueError('split must be train, val or test')
    with open(path, encoding='utf-8') as f:
        split = json.load(f)
    validate_split(split, rows)
    return split[name+'_ids']


def read_candidates(path, min_trajectory_iou=None, min_mask_iou=None):
    import lmdb
    with lmdb.open(str(path), readonly=True, lock=False, readahead=False) as env:
        with env.begin() as txn:
            all_ids = json.loads(txn.get(b'__ids__'))
            ready = txn.get(b'__ids_ready__')
            ids = json.loads(ready) if ready else [s for s in all_ids if json.loads(txn.get(s.encode())).get('trajectory_key')]
            rows = []
            for sid in sorted(ids):
                row = json.loads(txn.get(sid.encode()))
                if row.get('sample_id') != sid or not row.get('trajectory_key') or not row.get('masks_key'):
                    raise ValueError(sid+': missing two-stage metadata')
                scores = row.get('stage_iou') or {}
                if any(threshold is not None and (scores.get(key) is None or scores[key] < threshold)
                       for key,threshold in [('trajectory_whole',min_trajectory_iou),('mask_cleanup_whole',min_mask_iou)]):
                    continue
                rows.append(row)
    return rows, dict(total=len(all_ids), ready=len(ids), quality_excluded=len(ids)-len(rows))
