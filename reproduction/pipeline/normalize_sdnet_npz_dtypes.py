"""Normalize an SDNet dataset to RHSEDB's native NPZ dtypes atomically."""
import argparse
import os
from pathlib import Path

import numpy as np


MASK_FIELDS = ('reference_single_image', 'target_image', 'target_single_image')


def normalize(path):
    with np.load(path, allow_pickle=False) as source:
        values = {key: source[key] for key in source.files}
    changed = False
    for key in MASK_FIELDS:
        if key in values and values[key].dtype != np.bool_:
            values[key] = values[key].astype(bool)
            changed = True
    if 'reference_color_image' in values and values['reference_color_image'].dtype != np.float32:
        values['reference_color_image'] = values['reference_color_image'].astype(np.float32)
        changed = True
    if 'reference_single_centroid' in values and values['reference_single_centroid'].dtype != np.float32:
        values['reference_single_centroid'] = values['reference_single_centroid'].astype(np.float32)
        changed = True
    if 'stroke_label' in values and values['stroke_label'].dtype != np.int64:
        values['stroke_label'] = values['stroke_label'].astype(np.int64)
        changed = True
    if changed:
        temporary = path.with_suffix('.dtype-normalize.tmp.npz')
        np.savez_compressed(temporary, **values)
        os.replace(temporary, path)
    return changed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True)
    a = p.parse_args()
    root = Path(a.dataset).resolve()
    paths = sorted(path for split in ('train', 'test', 'heldout_test')
                   for path in (root / split).glob('*.npz') if not path.is_symlink())
    changed = sum(normalize(path) for path in paths)
    print('checked={} changed={} symlinks_preserved={}'.format(
        len(paths), changed,
        sum(1 for path in (root / 'train').glob('*.npz') if path.is_symlink())))


if __name__ == '__main__':
    main()
