"""Build bank from supplied RHSEDB-compatible NPZ masks and explicit 24-class labels."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from common import records, arrays, masks_and_labels, output_dir, write_records, require, digest, SCHEMA


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True,help='JSONL: character, npz_path, source, category_system=StrokeExtraction-24')
    p.add_argument('--output',required=True)
    a=p.parse_args(); rows=records(a.manifest); out=output_dir(a.output)
    require(len({r['character'] for r in rows})==len(rows),'Only one reference per character')
    for r in rows:
        require(r.get('source') and r.get('category_system')=='StrokeExtraction-24','Explicit source and StrokeExtraction-24 mapping required; raw SVG/25-class IDs unsupported')
        z=arrays(r['npz_path']); masks=z['reference_single_image']; labels=z['stroke_label']
        masks_and_labels(masks,labels)
        centers=np.array([[np.where(m)[1].mean(),np.where(m)[0].mean()] for m in masks])
        dest=out/(hashlib.sha256(r['character'].encode()).hexdigest()[:20]+'.npz')
        np.savez_compressed(str(dest), reference_single_image=masks.astype(np.uint8),stroke_label=labels,reference_single_centroid=centers)
        r.update(reference_path=str(dest),reference_sha256=digest(dest),source_npz_path=r['npz_path'],schema=SCHEMA)
    write_records(out/'manifest.jsonl',rows)


if __name__ == '__main__': main()
