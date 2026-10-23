"""Create an immutable character-stratified split from a read-only LMDB."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import sys

import lmdb
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.datasets.dataset_split import create_split, read_candidates


def popcount(value):
    return value.bit_count() if hasattr(int, "bit_count") else bin(value).count("1")


class Groups:
    def __init__(self, n):
        self.parent = list(range(n))

    def root(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def join(self, a, b):
        self.parent[self.root(a)] = self.root(b)


class HashTree:
    """BK tree for exact Hamming-radius search on 64-bit difference hashes."""
    def __init__(self):
        self.tree = None

    def insert(self, value, index, radius):
        matches = []
        stack = [self.tree] if self.tree else []
        while stack:
            h, idx, children = stack.pop()
            d = popcount(h ^ value)
            if d <= radius:
                matches.append(idx)
            stack.extend(child for edge,child in children.items() if d-radius <= edge <= d+radius)
        node = self.tree
        if node is None:
            self.tree = (value, index, {})
        else:
            while True:
                d = popcount(node[0] ^ value)
                if d == 0:
                    break
                if d not in node[2]:
                    node[2][d] = (value, index, {})
                    break
                node = node[2][d]
        return matches


def group_candidates(path, rows, radius):
    groups = Groups(len(rows)); exact = {}; tree = HashTree(); near_edges = 0
    with lmdb.open(str(path), readonly=True, lock=False, readahead=False) as env:
        with env.begin() as txn:
            for i,row in enumerate(rows):
                tokens = []
                for field in ('original_image_path',):
                    if row.get(field): tokens.append((field,row[field]))
                source = row.get('provenance') or {}
                if source.get('source_image_path'):
                    tokens.append(('original_image_path',source['source_image_path']))
                images = []
                for field in ('image_key','original_image_key'):
                    if not row.get(field): continue
                    data = txn.get(row[field].encode())
                    if data is None: raise ValueError(row['sample_id']+': missing '+field)
                    # Actual decoded pixels detect duplicates despite encoding differences.
                    im = Image.open(io.BytesIO(data)).convert('L')
                    tokens.append(('pixels', im.size, hashlib.sha256(im.tobytes()).hexdigest()))
                    if field == 'image_key': images.append(im)
                if not images: raise ValueError('missing normalized image')
                for token in tokens:
                    if token in exact: groups.join(i,exact[token])
                    else: exact[token] = i
                thumb = np.asarray(images[0].resize((9,8),Image.Resampling.LANCZOS))
                bits = (thumb[:,1:] > thumb[:,:-1]).reshape(-1)
                h = int.from_bytes(np.packbits(bits).tobytes(),'big')
                for other in tree.insert(h,i,radius):
                    groups.join(i,other); near_edges += 1
                if (i+1) % 10000 == 0: print('grouped',i+1,flush=True)
    result = {}
    for i,row in enumerate(rows): result.setdefault(groups.root(i),[]).append(row['sample_id'])
    return sorted((sorted(g) for g in result.values()),key=lambda g:g[0]), near_edges


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--ratios', type=float, nargs=3, default=[.8,.1,.1])
    p.add_argument('--near-radius', type=int, default=2)
    p.add_argument('--min-trajectory-iou', type=float)
    p.add_argument('--min-mask-iou', type=float)
    args = p.parse_args()
    if args.output.exists(): p.error('output already exists; choose a new versioned path')
    if not 0 <= args.near_radius <= 8: p.error('near-radius must be in [0,8]')
    for value in (args.min_trajectory_iou,args.min_mask_iou):
        if value is not None and not 0 <= value <= 1: p.error('IoU threshold must be in [0,1]')
    rows, selection = read_candidates(args.dataset,args.min_trajectory_iou,args.min_mask_iou)
    groups, edges = group_candidates(args.dataset,rows,args.near_radius)
    split = create_split(rows,groups,args.ratios,args.seed)
    split.update(selection=selection, filters=dict(min_trajectory_iou=args.min_trajectory_iou,min_mask_iou=args.min_mask_iou),
                 strategy='character_stratified_grouped', grouping=dict(normalized_dhash_hamming_radius=args.near_radius,
                 near_edges=edges, exact='decoded grayscale pixels and source image paths',
                 limitations='Perceptual hashing is heuristic, not exhaustive near-duplicate verification. Source collection and unknown writers are not isolated. Cross-character components assigned to train.'),
                 dataset_name=args.dataset.name)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as f:
        json.dump(split,f,ensure_ascii=False,indent=2); f.write('\n')
    print(json.dumps(dict(summary=split['summary'],groups=len(groups),characters=len(split['characters']),
                         exceptions=dict(Counter(r['coverage_exception'] for r in split['characters'].values() if r['coverage_exception']))),ensure_ascii=False),flush=True)


if __name__ == '__main__': main()
