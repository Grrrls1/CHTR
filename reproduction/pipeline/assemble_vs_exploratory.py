"""Assemble passing per-stroke VS trajectories into exploratory six-dimensional character NPZs."""
import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from common import arrays, digest, gray, iou, output_dir, png, preview, records, render, require, states, validate_points, write_json, write_records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True,help='render_and_filter best.jsonl')
    p.add_argument('--output',required=True)
    p.add_argument('--iou',type=float,default=.87)
    a=p.parse_args();require(0<a.iou<=1,'invalid threshold')
    groups=defaultdict(list)
    for row in records(a.manifest):groups[row['sample_id']].append(row)
    out=output_dir(a.output);packed=[];rejected=[];views=[]
    for sample_id,rows in groups.items():
        try:
            rows=sorted(rows,key=lambda row:row['stroke_index']);first=rows[0]
            expected=int(first['expected_strokes'])
            require([row['stroke_index'] for row in rows]==list(range(expected)),
                    'missing or duplicate stroke trajectories')
            xyz=[];subpaths=[0];stroke_offsets=[0];labels=[];stroke_qc=[]
            for row in rows:
                require(row['qc']['qc_status']=='passed' and row['qc']['iou']>=a.iou
                        and row['qc']['upstream_iou']>=a.iou,'stroke failed VS QC')
                z=arrays(row['trajectory_path']);validate_points(z['points'],z['subpath_offsets'],bounds=False)
                base=stroke_offsets[-1];xyz.append(z['points'][:,:3])
                subpaths.extend((z['subpath_offsets'][1:]+base).tolist())
                stroke_offsets.append(base+len(z['points']));labels.append(int(row['stroke_label']))
                stroke_qc.append(dict(stroke_index=row['stroke_index'],candidate_id=row['candidate_id'],
                                      iou=row['qc']['iou'],upstream_iou=row['qc']['upstream_iou'],
                                      native_replay_iou=row['qc']['native_replay_iou']))
            offsets=np.asarray(subpaths,np.int64);points=states(
                np.c_[np.concatenate(xyz).astype(np.float32),np.zeros((stroke_offsets[-1],3),np.float32)],offsets)
            image=render(points,offsets);target=gray(first['image_path'])<128;whole=iou(image>=128,target)
            require(whole>=a.iou,'whole-character six-dimensional render IoU below threshold: %.4f'%whole)
            directory=out/sample_id;directory.mkdir();npz=directory/'sample.npz'
            np.savez_compressed(npz,points=points,subpath_offsets=offsets,
                stroke_offsets=np.asarray(stroke_offsets,np.int64),stroke_types=np.asarray(labels,np.int64),
                image_size=np.array(256),schema=np.array('chtr-vs-exploratory-v1'))
            render_path=png(directory/'render.png',255-image)
            overlay=np.full((256,256,3),255,np.uint8);overlay[target]=[0,0,255];overlay[image>=128]=[255,0,0];overlay[target&(image>=128)]=[0,0,0]
            comparison=png(directory/'comparison.png',overlay)
            item=dict(schema='chtr-vs-exploratory-v1',sample_id=sample_id,character=first['character'],
                      stroke_count=expected,npz_path=str(npz),npz_sha256=digest(npz),
                      image_path=first['image_path'],render_path=render_path,comparison_path=comparison,
                      whole_iou=whole,threshold=a.iou,stroke_qc=stroke_qc,
                      review_status='auto_selected_exploratory',human_review_performed=False)
            write_json(directory/'metadata.json',item);packed.append(item);views.append(item)
        except Exception as exc:rejected.append(dict(sample_id=sample_id,error=str(exc)))
    write_records(out/'manifest.jsonl',packed);write_records(out/'rejected.jsonl',rejected)
    write_json(out/'summary.json',dict(input_samples=len(groups),packed=len(packed),rejected=len(rejected),
        threshold=a.iou,note='Exploratory automatic QC; no human mask or trajectory approval claimed.'))
    preview(out,views,'Six-dimensional reconstruction: red=target, blue=render, black=intersection')
    print('Packed %d; rejected %d'%(len(packed),len(rejected)))


if __name__=='__main__':main()
