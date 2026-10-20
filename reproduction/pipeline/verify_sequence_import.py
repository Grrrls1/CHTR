"""Verify generated ordered NPZ hashes and upstream CCSE category metadata."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from common import digest, read_json, records, require, write_json


def verify(manifest, graphics, ccse):
    manifest = Path(manifest).resolve()
    rows = records(manifest)
    original = read_json(graphics)
    by_character = {r['character']: r for r in original}
    require(len(by_character) == len(original), 'duplicate source characters')
    category_map = read_json(manifest.parent / 'category_map.json')
    require(category_map['category_system'] == 'CCSE-25', 'category system mismatch')
    for split in ('train', 'val', 'test'):
        source_categories = read_json(Path(ccse) / 'annotations' / ('instances_%s2021.json' % split))['categories']
        require(category_map['categories'] == source_categories, 'category map differs from CCSE ' + split)
    require(len(rows) == len(by_character), 'manifest/source count mismatch')
    counts = Counter()
    for row in rows:
        record = by_character[row['character']]
        path = Path(row['ordered_strokes_path'])
        require(path.is_file() and digest(path) == row['ordered_strokes_sha256'], 'NPZ hash mismatch: ' + str(path))
        with np.load(path, allow_pickle=False) as z:
            n = len(record['strokes'])
            require(z['ordered_stroke_masks'].shape == (n, 256, 256), 'mask shape mismatch: ' + str(path))
            require(np.all(z['ordered_stroke_masks'].reshape(n, -1).sum(1) > 0), 'empty mask: ' + str(path))
            require(str(z['character']) == row['character'] and str(z['category_system']) == 'CCSE-25', 'NPZ metadata mismatch')
            require(np.array_equal(z['stroke_order'], np.arange(n)), 'stroke order mismatch')
            require(len(z['stroke_label']) == n and len(z['raw_partial_stroke_label']) == len(record['category_id']), 'label length mismatch')
            require(np.array_equal(z['raw_partial_stroke_label'], record['category_id']), 'source labels mismatch')
            complete = n == len(record['category_id'])
            expected = record['category_id'] if complete else [-1] * n
            require(np.array_equal(z['stroke_label'], expected), 'ordered label mismatch')
            require(row['category_status'] == ('complete' if complete else 'unknown_due_to_upstream_missing_annotations'), 'status mismatch')
            for index in range(n):
                expected_median = np.asarray([[x / 4, (900 - y) / 4] for x, y in record['medians'][index]], np.float32)
                require(np.array_equal(z['median_%03d' % index], expected_median), 'median mismatch')
            counts[row['category_status']] += 1
    return dict(verified_npz=len(rows), category_map_matches_all_ccse_splits=True,
                category_map_sha256=digest(manifest.parent / 'category_map.json'),
                graphics_sha256=digest(graphics), manifest_sha256=digest(manifest),
                by_category_status=dict(counts))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--graphics', required=True)
    p.add_argument('--ccse', required=True)
    p.add_argument('--report', required=True)
    args = p.parse_args()
    result = verify(args.manifest, args.graphics, args.ccse)
    write_json(args.report, result)
    print(result)


if __name__ == '__main__':
    main()
