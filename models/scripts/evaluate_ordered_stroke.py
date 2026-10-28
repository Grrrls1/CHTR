"""Explicit stage-one validation/test; never selects a checkpoint from test."""
import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.training.ordered_stroke import load_model, make_catalog, make_dataset, select_indices, local_path
from models.training.ordered_stroke_evaluation import evaluate_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', choices=('train', 'val', 'test'), default='val')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--limit', type=int, default=0, help='0 evaluates the full split, independent of smoke config')
    args = parser.parse_args()
    if args.limit < 0 or (args.batch_size is not None and args.batch_size < 1):
        parser.error('limit must be nonnegative and batch-size positive')
    print('[ordered-eval] loading checkpoint', flush=True)
    model, ck = load_model(args.checkpoint, args.device)
    config = dict(ck['config'])
    torch.set_num_threads(config['threads'])
    if args.batch_size is not None:
        config['eval_batch_size'] = args.batch_size
    print('[ordered-eval] loading catalog and validating fixed split', flush=True)
    catalog = make_catalog(config)
    try:
        dataset = make_dataset(catalog, args.split, config)
        indices = select_indices(dataset, args.limit)
        report = evaluate_split(model, dataset, args.device, config, local_path(args.output), indices,
                                checkpoint=dict(path=str(local_path(args.checkpoint)), epoch=ck['epoch'], step=ck['step']))
        print(json.dumps(report, ensure_ascii=False), flush=True)
    finally:
        catalog.close()


if __name__ == '__main__':
    main()
