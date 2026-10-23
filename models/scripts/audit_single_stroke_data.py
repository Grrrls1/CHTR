"""Read all (or a fixed spread of) actual assets in the frozen character split."""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.datasets.single_stroke import StrokeCatalog, SingleStrokeDataset, masked_stroke_image


def audit(catalog, limit_per_split=0, max_points=4096):
    report = dict(dataset_fingerprint=catalog.fingerprint, full_scan=not limit_per_split,
                  note='Asset/interface validation, not manual label-quality approval.', splits={})
    for name in ('train','val','test'):
        ds = SingleStrokeDataset(catalog,name,max_points=max_points)
        ids = ds.ids if not limit_per_split else [ds.ids[int(i)] for i in np.linspace(0,len(ds.ids)-1,min(limit_per_split,len(ds.ids)),dtype=int)]
        stats = dict(characters=len(ds.ids), strokes=len(ds), checked_characters=0, checked_strokes=0,
                     min_points=None, max_points=0, max_width_px=0., max_mask_distance_px=0.,
                     points_outside_mask=0, points_checked=0, masked_content_matches=True)
        for sid in ids:
            image,points,offsets,masks = catalog.character_assets(sid)
            for k,mask in enumerate(masks):
                p = points[offsets[k]:offsets[k+1]]
                if len(p)>max_points: raise ValueError(sid+': cap exceeded')
                stats['min_points'] = min(stats['min_points'] or len(p),len(p))
                stats['max_points'] = max(stats['max_points'],len(p))
                stats['max_width_px'] = max(stats['max_width_px'],float(p[:,2].max()))
                # Record rather than silently correct automatic trajectory/mask discrepancies.
                distance = cv2.distanceTransform((~mask).astype(np.uint8),cv2.DIST_L2,3)
                xy = np.rint(p[:,:2]).astype(int)
                d = distance[xy[:,1],xy[:,0]]
                stats['max_mask_distance_px'] = max(stats['max_mask_distance_px'],float(d.max()))
                stats['points_outside_mask'] += int((d>0).sum()); stats['points_checked'] += len(p)
                stroke = masked_stroke_image(image,mask)
                if not np.array_equal(stroke[mask],image[mask]) or not (stroke[~mask]==255).all():
                    raise ValueError('masked original-image construction mismatch')
            stats['checked_characters'] += 1; stats['checked_strokes'] += len(masks)
            if stats['checked_characters'] % 2000 == 0:
                print(name,stats['checked_characters'],'characters checked',flush=True)
        report['splits'][name] = stats
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True);p.add_argument('--split-file',required=True)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--limit-per-split',type=int,default=0)
    p.add_argument('--max-points',type=int,default=4096)
    args=p.parse_args()
    if args.limit_per_split<0:p.error('limit must be nonnegative')
    if args.output.exists():p.error('output exists')
    start=time.monotonic(); catalog=StrokeCatalog(args.dataset,args.split_file)
    try: report=audit(catalog,args.limit_per_split,args.max_points)
    finally: catalog.close()
    report['seconds']=time.monotonic()-start
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
