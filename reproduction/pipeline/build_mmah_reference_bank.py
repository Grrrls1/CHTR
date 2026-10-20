"""Build exploratory StrokeExtraction references from ordered MMAH masks and user-mapped labels."""
import argparse
import colorsys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from common import digest, masks_and_labels, output_dir, read_json, records, require, write_json, write_records


COLORS = np.asarray([colorsys.hsv_to_rgb(i / 24, 1, 1) for i in range(24)], np.float32)


def build_one(row, output):
    count = row['stroke_count']
    if count > 30:
        return None, dict(character=row['character'], reason='stroke_count_exceeds_model_limit', strokes=count)
    labels = np.asarray(row['stroke_label24_proposed'], np.int64)
    if np.any(labels < 0):
        return None, dict(character=row['character'], reason='unresolved_category',
                          unresolved=int(np.count_nonzero(labels < 0)))
    path = Path(row['ordered_strokes_path'])
    require(digest(path) == row['ordered_strokes_sha256'], 'source hash changed: ' + str(path))
    with np.load(path, allow_pickle=False) as z:
        require(str(z['character']) == row['character'], 'source character mismatch')
        require(np.array_equal(z['stroke_order'], np.arange(count)), 'source order mismatch')
        masks = z['ordered_stroke_masks'].astype(np.uint8)
    masks_and_labels(masks, labels)
    centers = np.asarray([[np.where(mask)[1].mean(), np.where(mask)[0].mean()]
                          for mask in masks], np.float32)
    color = np.zeros((256, 256, 3), np.float32)
    for mask, label in zip(masks, labels):
        color[mask != 0] = COLORS[label]
    destination = output / ('mmah_ref_u%05X.npz' % ord(row['character']))
    np.savez_compressed(destination, name=np.array(row['character']),
                        reference_single_image=masks, stroke_label=labels,
                        reference_single_centroid=centers,
                        reference_color_image=color.transpose(2, 0, 1),
                        category_system=np.array('StrokeExtraction-24'),
                        stroke_order=np.arange(count, dtype=np.int64),
                        source=np.array('Make Me A Hanzi ordered geometry + CCSE category mapping'))
    return dict(character=row['character'], reference_path=str(destination.resolve()),
                reference_sha256=digest(destination), source_npz_path=str(path.resolve()),
                source_npz_sha256=row['ordered_strokes_sha256'],
                stroke_count=count, source='Make Me A Hanzi + CCSE v2 user-mapped',
                category_system='StrokeExtraction-24',
                mapping_policy='CCSE piedian/dian collapsed to RHSEDB dian=0; hengzhezhezhe to 13',
                order_source='Make Me A Hanzi', review_status='pending',
                bank_status='exploratory_unregistered'), None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--proposals', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    require(a.workers > 0, 'workers must be positive')
    proposal_path = Path(a.proposals).resolve()
    policy = read_json(proposal_path.parent / 'mapping.json')
    require(policy['piedian_policy'] == 'collapse_to_dian' and
            policy['user_equivalent_mapping'] == {'6': 0},
            'User-approved equivalence policy required')
    rows = records(proposal_path)
    require(len({r['character'] for r in rows}) == len(rows), 'duplicate character')
    out = output_dir(a.output)
    built, skipped, errors = [], [], []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [(row, pool.submit(build_one, row, out)) for row in rows]
        for row, future in futures:
            try:
                item, skip = future.result()
                if item is not None:
                    built.append(item)
                else:
                    skipped.append(skip)
            except Exception as exc:
                errors.append(dict(character=row['character'], error=str(exc)))
    write_records(out / 'manifest.jsonl', built)
    write_records(out / 'skipped.jsonl', skipped)
    write_records(out / 'errors.jsonl', errors)
    write_json(out / 'summary.json', dict(input_characters=len(rows), built=len(built),
               skipped=len(skipped), errors=len(errors),
               mapping_policy=policy['piedian_policy'], proposal_manifest_sha256=digest(proposal_path),
               note='Reference-only bank; no target_image or target_single_image. Not a full RHSEDB training set.'))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
