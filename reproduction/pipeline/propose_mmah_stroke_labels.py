"""Propose RHSEDB-24 labels for ordered Make Me A Hanzi strokes; never approve them."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from common import output_dir, read_json, records, require, write_json, write_records, digest


# CCSE category IDs are 1-based; RHSEDB labels are 0-based. Category 6
# (piedian) is ambiguous in matched examples: RHSEDB uses both dian=0 and
# piedian=22. Leave it unresolved unless a same-character RHSEDB anchor agrees.
MAP25 = {
    1: 18, 2: 16, 3: 4, 4: 3, 5: 21, 7: 17, 8: 1, 9: 5, 10: 11,
    11: 13, 12: 13, 13: 15, 14: 12, 15: 7, 16: 6, 17: 9,
    18: 0, 19: 2, 20: 20, 21: 8, 22: 10, 23: 14, 24: 23, 25: 19,
}
AMBIGUOUS_CCSE = 6
LOW_SUPPORT_CCSE = {11, 23}


def anchor_labels(row, categories, mapped):
    """Use same-index RHSEDB labels only if all nonambiguous strokes agree."""
    if row is None or row.get('source_split') != 'train':
        return None
    with np.load(row['npz_path'], allow_pickle=False) as z:
        labels = z['stroke_label'].astype(int).tolist()
    if len(labels) != len(categories):
        return None
    if any(label != expected for label, expected, category in zip(labels, mapped, categories)
           if category != AMBIGUOUS_CCSE):
        return None
    if any(label not in (0, 22) for label, category in zip(labels, categories)
           if category == AMBIGUOUS_CCSE):
        return None
    return labels


def propose(official, v2, rhsedb, piedian_policy='unresolved'):
    if v2 is None:
        count = official['stroke_count']
        return [-1] * count, ['no_ccse_category_source'] * count, None
    require(official['stroke_count'] == v2['stroke_count'], 'MMAH/CCSE stroke count mismatch')
    count = official['stroke_count']
    if v2['category_status'] != 'complete':
        return [-1] * count, ['incomplete_ccse_categories'] * count, None
    categories = v2['stroke_labels']
    require(len(categories) == count, 'CCSE label count mismatch')
    mapped = [MAP25.get(category, -1) for category in categories]
    anchor = anchor_labels(rhsedb, categories, mapped)
    labels, status = [], []
    for i, category in enumerate(categories):
        if category == AMBIGUOUS_CCSE and piedian_policy == 'collapse_to_dian':
            labels.append(0)
            status.append('user_equivalent_dian_piedian_proposal')
        elif category == AMBIGUOUS_CCSE:
            if anchor is None:
                labels.append(-1)
                status.append('ambiguous_dian_or_piedian')
            else:
                labels.append(anchor[i])
                status.append('rhsedb_same_index_proposal')
        else:
            require(mapped[i] >= 0, 'unmapped CCSE category')
            labels.append(mapped[i])
            status.append('taxonomy_low_support_proposal' if category in LOW_SUPPORT_CCSE
                          else 'taxonomy_name_proposal')
    return labels, status, (rhsedb['npz_path'] if anchor is not None else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--makemeahanzi', required=True)
    p.add_argument('--ccse-v2', required=True)
    p.add_argument('--rhsedb', required=True)
    p.add_argument('--source-comparison', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--piedian-policy', choices=('unresolved', 'collapse_to_dian'),
                   default='unresolved')
    a = p.parse_args()
    mmah = records(a.makemeahanzi)
    v2 = {r['character']: r for r in records(a.ccse_v2)}
    rhs = {r['character']: r for r in records(a.rhsedb)}
    comparison = read_json(a.source_comparison)
    require(comparison['shared_characters'] == len(v2) and
            comparison['identical_stroke_paths'] == len(v2) and
            comparison['identical_medians'] == len(v2), 'MMAH/CCSE geometry not verified')
    require(len({r['character'] for r in mmah}) == len(mmah), 'duplicate MMAH character')
    require(set(v2).issubset({r['character'] for r in mmah}), 'CCSE character absent from MMAH')
    out = output_dir(a.output)
    rows, status_counts = [], Counter()
    for r in mmah:
        character = r['character']
        labels, statuses, anchor_path = propose(r, v2.get(character), rhs.get(character), a.piedian_policy)
        status_counts.update(statuses)
        rows.append(dict(character=character, ordered_strokes_path=r['ordered_strokes_path'],
                         ordered_strokes_sha256=r['ordered_strokes_sha256'],
                         stroke_count=r['stroke_count'], stroke_order_source=r['order_source'],
                         stroke_label24_proposed=labels, label_status=statuses,
                         ccse_source_path=(v2[character]['ordered_strokes_path'] if character in v2 else None),
                         rhsedb_anchor_path=anchor_path,
                         category_system='StrokeExtraction-24-proposed',
                         review_status='pending', ready_for_strokeextraction=False))
    write_records(out / 'manifest.jsonl', rows)
    write_json(out / 'mapping.json', dict(
        ccse25_to_rhsedb24={str(k): v for k, v in sorted(MAP25.items())},
        ambiguous_ccse_category=AMBIGUOUS_CCSE,
        piedian_policy=a.piedian_policy,
        user_equivalent_mapping=({'6': 0} if a.piedian_policy == 'collapse_to_dian' else {}),
        low_support_ccse_categories=sorted(LOW_SUPPORT_CCSE),
        note='Proposed taxonomy correspondence, not an approved label conversion.'))
    write_json(out / 'summary.json', dict(
        characters=len(rows), strokes=sum(r['stroke_count'] for r in rows),
        by_label_status=dict(status_counts),
        characters_all_labels_proposed=sum(-1 not in r['stroke_label24_proposed'] for r in rows),
        characters_requiring_review=len(rows),
        source_sha256={name: digest(value) for name, value in [
            ('makemeahanzi', a.makemeahanzi), ('ccse_v2', a.ccse_v2),
            ('rhsedb', a.rhsedb), ('source_comparison', a.source_comparison)]},
        note='Every row remains pending; unresolved labels are -1. Never feed directly to the 24-class reference-bank builder.'))


if __name__ == '__main__':
    main()
