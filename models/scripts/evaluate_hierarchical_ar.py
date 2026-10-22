"""Evaluate only versioned three-task checkpoints, using the dedicated sampler."""
import argparse
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent))
from models.datasets.hierarchical_ar_dataset import TrajectoryDataset
from models.training.hierarchical_ar_setup import read_checkpoint,create_model,local_path
from models.training.hierarchical_ar_evaluation import evaluate
from models.scripts.train_hierarchical_ar import fixed_coverage


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--dataset');parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--limit',type=int,default=32)
    parser.add_argument('--split',choices=['heldout_images','training_images','all'],default='heldout_images')
    args=parser.parse_args()
    if args.limit<1:raise ValueError('limit must be positive')
    ck=read_checkpoint(local_path(args.checkpoint));config=ck['config'];device=torch.device(args.device)
    torch.set_num_threads(config['threads'])
    dataset=TrajectoryDataset(local_path(args.dataset or config['dataset']))
    try:
        lengths,_=dataset.inspect(config['model']['max_strokes'],config['model']['max_points'])
        if args.split=='all':indices=list(range(len(dataset)))
        else:
            import json
            split=json.loads(local_path(args.checkpoint).with_name('split.json').read_text())
            selected=split['heldout_ids' if args.split=='heldout_images' else 'train_ids']
            lookup={sid:i for i,sid in enumerate(dataset.ids)}
            if any(sid not in lookup for sid in selected):raise ValueError('dataset lacks saved split IDs')
            indices=[lookup[sid] for sid in selected]
        if not indices:raise ValueError('requested split is empty')
        indices=fixed_coverage(indices,lengths,args.limit,dataset.ids)
        model=create_model(config,device,ck)
        report=evaluate(model,dataset,indices,device,config['decode'],Path(args.output),args.split)
        print({k:v for k,v in report.items() if k!='rows'})
    finally:dataset.close()


if __name__=='__main__':main()
