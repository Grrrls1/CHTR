"""Full held-out free decoding with fixed-character visualizations."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from models.datasets.single_stroke import StrokeCatalog,SingleStrokeDataset
from models.training.single_stroke import local_path,load_model,signature
from models.training.single_stroke_evaluation import evaluate_split,visualization_ids


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--split',choices=['train','val','test'],default='val')
    p.add_argument('--training-subset',action='store_true')
    p.add_argument('--limit',type=int,default=0,help='0 evaluates the full split; positive values are explicitly partial diagnostics')
    p.add_argument('--visualize-characters',type=int,default=30)
    p.add_argument('--decode-max-points',type=int)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--threshold',type=float,default=.5);p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=4)
    args=p.parse_args()
    if args.limit<0 or args.visualize_characters<0 or args.batch_size<1 or args.output.exists():p.error('invalid limits/batch or output already exists')
    if args.training_subset and args.split!='train':p.error('training-subset requires --split train')
    torch.set_num_threads(args.threads);device=torch.device(args.device)
    model,ck=load_model(local_path(args.checkpoint),device);config=ck['config']
    catalog=StrokeCatalog(local_path(config['dataset']),local_path(config['split_file']))
    try:
        if signature(config,catalog)!=ck['signature']:raise ValueError('checkpoint data/split signature mismatch')
        ds=SingleStrokeDataset(catalog,args.split,config['model']['max_points'])
        indices=range(len(ds))
        if args.training_subset:
            usage=json.loads(local_path(args.checkpoint).with_name('data_usage.json').read_text())
            positions={s:i for i,s in enumerate(ds.ids)};indices=[]
            for sid,k in usage['train_keys']:
                ci=positions[sid]
                if not 0<=k<catalog.rows[sid]['stroke_count']:raise ValueError('invalid saved training key')
                indices.append((ds.ends[ci-1] if ci else 0)+k)
        if args.limit:
            indices=[indices[int(i)] for i in np.linspace(0,len(indices)-1,min(args.limit,len(indices)),dtype=int)]
        fixed=visualization_ids(ds,args.visualize_characters)
        if args.limit or args.training_subset:
            fixed=list(dict.fromkeys(ds.key(i)[0] for i in indices))[:args.visualize_characters]
        eval_config=dict(config,eval_batch_size=args.batch_size,validity_threshold=args.threshold,
                         decode_max_points=args.decode_max_points or config['model']['max_points'])
        report=evaluate_split(model,ds,device,eval_config,args.output,fixed,indices)
        report.update(checkpoint=str(args.checkpoint),step=ck['step'],epoch=ck['epoch'],training_subset=args.training_subset)
        (args.output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(report,ensure_ascii=False),flush=True)
    finally:catalog.close()


if __name__=='__main__':main()
