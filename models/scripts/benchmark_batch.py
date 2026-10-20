"""Measure real LMDB training throughput and CUDA memory for one batch size."""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from models.datasets.filtered_lmdb_dataset import CHTRFilteredDataset
from models.datasets.lmdb_dataset import collate_strokes
from models.networks.model import StrokeTrajectoryModel
from models.losses.multitask import multitask_loss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--batch-size', type=int, required=True)
    p.add_argument('--batches', type=int, default=16)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    torch.set_num_threads(config.get('threads', 4))
    torch.manual_seed(config['seed'])
    dataset = CHTRFilteredDataset(config['dataset'])
    loader_options = {'prefetch_factor': 1} if args.workers else {}
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_strokes, num_workers=args.workers, pin_memory=True,
                        **loader_options,
                        generator=torch.Generator().manual_seed(config['seed']))
    model = StrokeTrajectoryModel(**config['model']).cuda().train()
    optimizer = torch.optim.AdamW(model.branch_parameter_groups(), lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    result = dict(batch_size=args.batch_size, workers=args.workers, batches=0, samples=0, max_strokes=0, max_points=0)
    measured_samples = 0
    try:
        torch.cuda.reset_peak_memory_stats()
        for i, batch in enumerate(loader):
            if i == 2:
                torch.cuda.synchronize()
                started = time.perf_counter()
            batch = {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            loss, terms = multitask_loss(out, batch, config['loss_weights'], config['state_weights'],
                                     geometry_config=config.get('geometry_loss'))
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite benchmark loss')
            loss.backward()
            for group in optimizer.param_groups:
                norm = torch.nn.utils.clip_grad_norm_(group['params'], config['grad_clip'])
                if not torch.isfinite(norm):
                    raise RuntimeError('nonfinite benchmark gradient')
            optimizer.step()
            result['batches'] += 1
            result['samples'] += len(batch['images'])
            result['max_strokes'] = max(result['max_strokes'], batch['points'].shape[1])
            result['max_points'] = max(result['max_points'], batch['points'].shape[2])
            if i >= 2:
                measured_samples += len(batch['images'])
            del batch, out, loss, terms
            if i+1 >= args.batches:
                break
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        result.update(status='ok', samples_per_second=measured_samples/elapsed,
                      measured_seconds=elapsed, epoch_minutes=len(dataset)*elapsed/measured_samples/60)
    except torch.cuda.OutOfMemoryError:
        result['status'] = 'out_of_memory'
    finally:
        result['peak_allocated_mb'] = torch.cuda.max_memory_allocated()/1024**2
        result['peak_reserved_mb'] = torch.cuda.max_memory_reserved()/1024**2
        Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)
        dataset.close()


if __name__ == '__main__':
    main()
