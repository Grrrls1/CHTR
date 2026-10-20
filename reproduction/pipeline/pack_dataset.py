"""Pack complete, approved characters with writer/style-disjoint deterministic split."""
import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
import shutil
import numpy as np
from common import records, output_dir, require, arrays, digest, states, validate_points, render, iou, gray, png, write_records, write_json, SCHEMA, identifier


def pack_group(rows, out, seed=0, test_fraction=.2, minimum=.95):
    rows=sorted(rows,key=lambda r:r['stroke_index']); first=rows[0]; sid=identifier(first['sample_id'])
    n=first['expected_strokes']
    require([r['stroke_index'] for r in rows]==list(range(n)),'Missing or duplicate semantic strokes')
    require(first.get('character') and first.get('source') and first.get('image_path'),'Missing source provenance')
    group='writer:'+str(first['writer_id']) if first.get('writer_id') else 'style:'+str(first.get('style_id',''))
    require(group!='style:','writer_id/style_id required for leakage-safe split')
    all_points=[]; stroke_offsets=[0]; subpaths=[0]; labels=[]
    for r in rows:
        for key in ('character','source','writer_id','style_id','image_path','expected_strokes'):
            require(r.get(key)==first.get(key),'Inconsistent sample metadata: '+key)
        require(r.get('review_status')=='approved' and r.get('mask_review'),'Unapproved stroke masks')
        require(r.get('trajectory_review_status')=='approved' and r.get('trajectory_review'),'Unapproved trajectory')
        require(r['qc']['qc_status']=='passed' and r['qc']['iou']>=minimum and r['qc']['upstream_iou']>=minimum,'Failed or weakened QC')
        require(digest(r['trajectory_path'])==r['trajectory_sha256'],'Trajectory changed after review')
        require(digest(r['reviewed_path'])==r['reviewed_sha256'],'Region mask changed after review')
        require(digest(r['input_png'])==r['input_sha256'],'Target PNG changed after review')
        reviewed=arrays(r['reviewed_path'])
        require(int(reviewed['stroke_labels'][r['stroke_index']])==r['stroke_label'],'Reviewed label mismatch')
        require(np.array_equal(reviewed['extractnet_strokes'][r['stroke_index']].astype(bool),gray(r['input_png'])<128),'PNG differs from reviewed mask')
        z=arrays(r['trajectory_path']); validate_points(z['points'],z['subpath_offsets'])
        target=gray(r['input_png'])<128
        require(iou(render(z['points'],z['subpath_offsets'])>=128,target)>=minimum,'Recomputed stroke IoU failed')
        require(digest(r['upstream_render_path'])==r['upstream_render_sha256'],'Upstream render changed after QC')
        require(iou(render(z['points'],z['subpath_offsets'])>=128,gray(r['upstream_render_path'])<128)>=minimum,'Recomputed point fidelity failed')
        subpaths.extend((z['subpath_offsets'][1:]+stroke_offsets[-1]).tolist())
        all_points.append(z['points']); stroke_offsets.append(stroke_offsets[-1]+len(z['points'])); labels.append(r['stroke_label'])
    offsets=np.asarray(subpaths,dtype=np.int64); points=states(np.concatenate(all_points),offsets)
    target=gray(first['image_path']); require(target.shape==(256,256),'Character image not 256')
    image=render(points,offsets); score=iou(image>=128,target<128)
    require(score>=minimum,'Whole-character IoU failed: %.6f'%score)
    d=out/sid; d.mkdir()
    npz=d/'sample.npz'; np.savez_compressed(str(npz),points=points,subpath_offsets=offsets,stroke_offsets=np.asarray(stroke_offsets,dtype=np.int64),stroke_types=np.asarray(labels,dtype=np.int64),image_size=np.array(256),schema=np.array(SCHEMA))
    image_path=d/'image.png'; shutil.copyfile(first['image_path'],str(image_path))
    split='test' if int(hashlib.sha256((str(seed)+'|'+group).encode()).hexdigest()[:16],16)/float(16**16)<test_fraction else 'train'
    archived=[]
    for i,r in enumerate(rows):
        mask_path=d/('stroke%03d_mask.png'%i); shutil.copyfile(r['input_png'],str(mask_path))
        upstream_path=d/('stroke%03d_upstream.png'%i); shutil.copyfile(r['upstream_render_path'],str(upstream_path))
        archived.append(dict(r,input_png=str(mask_path),upstream_render_path=str(upstream_path)))
    meta=dict(schema=SCHEMA,sample_id=sid,character=first['character'],split=split,split_group=group,seed=seed,expected_strokes=n,npz_path=str(npz),npz_sha256=digest(npz),image_path=str(image_path),image_sha256=digest(image_path),render_path=png(d/'render.png',255-image),whole_iou=score,threshold=minimum,category_system='StrokeExtraction-24',coordinates='x right, y down; width is pixel diameter; state after arrival',strokes=archived)
    write_json(d/'metadata.json',meta)
    return meta


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--manifest',required=True); p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=0); p.add_argument('--test-fraction',type=float,default=.2); p.add_argument('--iou',type=float,default=.95)
    a=p.parse_args(); require(0<a.test_fraction<1 and .95<=a.iou<=1,'Invalid split/IoU threshold'); out=output_dir(a.output); groups=defaultdict(list)
    for r in records(a.manifest): groups[r['sample_id']].append(r)
    result=[]; rejected=[]
    for sid,rows in groups.items():
        try: result.append(pack_group(rows,out,a.seed,a.test_fraction,a.iou))
        except (ValueError,KeyError,OSError) as e: rejected.append(dict(sample_id=sid,error=str(e)))
    write_records(out/'manifest.jsonl',result); write_records(out/'rejected.jsonl',rejected)
    write_json(out/'splits.json',{s:[r['sample_id'] for r in result if r['split']==s] for s in ('train','test')})
    print('Packed %d; rejected %d'%(len(result),len(rejected)))
    if rejected or not result: raise SystemExit(1)


if __name__=='__main__': main()
