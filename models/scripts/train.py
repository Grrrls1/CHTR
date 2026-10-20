"""Minimal teacher-forced training with predicted stroke conditions and resume."""
import argparse
import hashlib
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

MODEL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_ROOT.parent))
from models.datasets.lmdb_dataset import collate_strokes
from models.datasets.filtered_lmdb_dataset import CHTRFilteredDataset, MASK_STAGES
from models.networks.model import StrokeTrajectoryModel
from models.losses.multitask import multitask_loss
from models.losses.pixel_geometry import settings as geometry_settings, geometry_metrics
from models.training.early_stopping import EarlyStopping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(MODEL_ROOT/'configs/minimal.json'))
    parser.add_argument('--max-steps', type=int, help='Total optimizer step limit, including resumed steps')
    parser.add_argument('--output-dir')
    parser.add_argument('--device')
    parser.add_argument('--resume', help='Trusted local checkpoint')
    parser.add_argument('--init-weights', help='Trusted checkpoint: weights only, new optimizer/run')
    parser.add_argument('--early-stopping-patience', type=int, help='Non-improving epochs; 0 disables')
    parser.add_argument('--early-stopping-min-delta', type=float, help='Minimum absolute epoch-loss improvement')
    parser.add_argument('--early-stopping-min-epochs', type=int, help='Minimum completed epochs before stopping')
    parser.add_argument('--epochs', type=int, help='Cumulative epoch limit, including resumed epochs')
    parser.add_argument('--min-trajectory-iou', type=float, help='Minimum stage_iou.trajectory_whole')
    parser.add_argument('--min-mask-iou', type=float, help='Minimum whole-character mask IoU')
    parser.add_argument('--mask-iou-stage', choices=MASK_STAGES,
                        help='Use cleaned or extracted whole-character mask IoU')
    args = parser.parse_args()
    if args.resume and args.init_weights:
        parser.error('--resume and --init-weights are mutually exclusive')
    config = json.loads(Path(args.config).read_text())
    for key in ('max_steps', 'output_dir', 'device', 'epochs', 'min_trajectory_iou', 'min_mask_iou', 'mask_iou_stage'):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    stopping_config = dict(config.get('early_stopping') or {})
    for name in ('patience', 'min_delta', 'min_epochs'):
        value = getattr(args, 'early_stopping_' + name)
        if value is not None:
            stopping_config[name] = value
    stopper = EarlyStopping(stopping_config)
    if stopping_config:
        config['early_stopping'] = stopping_config
    checkpoint_every = int(config.get('checkpoint_every', 1))
    if checkpoint_every < 1 or config['epochs'] < 1:
        raise ValueError('checkpoint_every and epochs must be positive')
    if config.get('geometry_loss'):
        geometry_settings(config['geometry_loss'])
    def path(value):
        p = Path(value)
        return p if p.is_absolute() else MODEL_ROOT/p
    output = path(config['output_dir']).resolve()
    output.mkdir(parents=True, exist_ok=True)
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config.get('threads', 2))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = config['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; run scripts/check_environment.py --require-cuda')
        torch.cuda.set_device(device)
        print('GPU:', torch.cuda.get_device_name(device), flush=True)
    dataset = CHTRFilteredDataset(path(config['dataset']), augment=config['augment'],
                                 min_trajectory_iou=config.get('min_trajectory_iou'),
                                 min_mask_iou=config.get('min_mask_iou'),
                                 mask_iou_stage=config.get('mask_iou_stage', 'mask_cleanup_whole'))
    print('dataset_selection=' + json.dumps(dataset.selection_summary), flush=True)
    (output/'dataset_selection.json').write_text(json.dumps(dataset.selection_summary, indent=2)+'\n')
    generator = torch.Generator().manual_seed(seed)
    loader_options = {'prefetch_factor': 1} if config['num_workers'] else {}
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True,
                        collate_fn=collate_strokes, num_workers=config['num_workers'], generator=generator,
                        pin_memory=device.type == 'cuda', **loader_options)
    model = StrokeTrajectoryModel(**config['model']).to(device)
    if args.init_weights:
        initial = torch.load(args.init_weights, map_location=device, weights_only=False)
        model.load_state_dict(initial['model'])
        config['initial_weights'] = str(Path(args.init_weights).resolve())
    parameter_groups = model.branch_parameter_groups() if model.separate_branches else [{'params': list(model.parameters())}]
    optimizer = torch.optim.AdamW(parameter_groups, lr=config['learning_rate'], weight_decay=config['weight_decay'])
    step, start_epoch, skip_batches = 0, 0, 0
    epoch_generator_state = generator.get_state()
    def signature():
        value = {'dataset': dataset.path, 'ids': dataset.ids, 'type_map': dataset.type_map,
                 'batch_size': config['batch_size'], 'num_workers': config['num_workers'],
                 'augment': config['augment'], 'condition_mode': config['condition_mode'],
                 'loss_weights': config['loss_weights'], 'state_weights': config['state_weights'], 'seed': seed}
        if any(dataset.iou_filters[k] is not None for k in ('min_trajectory_iou', 'min_mask_iou')):
            value['iou_filters'] = dataset.iou_filters
        if config.get('geometry_loss'):
            value['geometry_loss'] = config['geometry_loss']
        if config.get('early_stopping'):
            value['early_stopping'] = config['early_stopping']
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    if args.resume:
        if config['num_workers'] != 0 and config['augment']:
            raise ValueError('exact resume with random augmentation requires num_workers=0')
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint['config']['model'] != config['model'] or checkpoint['data_signature'] != signature():
            raise ValueError('resume requires matching architecture, dataset and training data settings')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint['config'].get('initial_weights'):
            config['initial_weights'] = checkpoint['config']['initial_weights']
        step, start_epoch, skip_batches = checkpoint['step'], checkpoint['epoch'], checkpoint['next_batch']
        # Replay the current epoch permutation; restore augmentation/model RNG after skipped batches.
        generator.set_state(checkpoint['epoch_generator_state'].cpu())
        if 'early_stopping' in checkpoint:
            stopper.load_state_dict(checkpoint['early_stopping'])
    (output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    (output/'stroke_type_map.json').write_text(json.dumps(dataset.type_map, ensure_ascii=False, indent=2)+'\n')
    print('device=%s samples=%d parameters=%d condition=%s' %
          (device, len(dataset), sum(p.numel() for p in model.parameters()), config['condition_mode']), flush=True)

    def save(epoch, next_batch):
        temp = output/'last.pt.tmp'
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'config': config, 'data_signature': signature(), 'step': step, 'epoch': epoch,
                    'next_batch': next_batch, 'epoch_generator_state': epoch_generator_state,
                    'early_stopping': stopper.state_dict(),
                    'rng': {'torch': torch.get_rng_state(), 'numpy': np.random.get_state(),
                            'python': random.getstate(),
                            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}, temp)
        temp.replace(output/'last.pt')

    def finish(reason):
        (output/'training_status.json').write_text(json.dumps({
            'reason': reason, 'step': step, 'completed_epochs': stopper.last_epoch,
            'best_epoch_loss': stopper.best, 'bad_epochs': stopper.bad_epochs,
        }, indent=2)+'\n')
        dataset.close()
        print('training stopped: %s; checkpoint: %s' % (reason, output/'last.pt'), flush=True)

    limit = config['max_steps']
    if limit and step >= limit:
        print('checkpoint already reached max_steps', flush=True)
        finish('max_steps')
        return
    if stopper.should_stop:
        finish('early_stopping')
        return
    def restore_rng():
        rng = checkpoint['rng']
        torch.set_rng_state(rng['torch'].cpu())
        np.random.set_state(rng['numpy'])
        random.setstate(rng['python'])
        if rng['cuda'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in rng['cuda']])

    model.train()
    with (output/'metrics.jsonl').open('a', encoding='utf-8') as log:
        for epoch in range(start_epoch, config['epochs']):
            epoch_generator_state = generator.get_state()
            restored = not bool(args.resume and epoch == start_epoch)
            batch_finished = time.monotonic()
            for batch_index, batch in enumerate(loader):
                if epoch == start_epoch and batch_index < skip_batches:
                    batch_finished = time.monotonic()
                    continue
                if not restored:
                    restore_rng()
                    restored = True
                step_started = time.monotonic()
                data_wait = step_started - batch_finished
                batch = {k: v.to(device, non_blocking=device.type == 'cuda') if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                prediction = model(batch, condition_mode=config['condition_mode'])
                loss, terms = multitask_loss(prediction, batch, config['loss_weights'], config['state_weights'],
                                            geometry_config=config.get('geometry_loss'))
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite loss')
                loss.backward()
                # Independent clipping prevents a large geometry gradient from
                # scaling down an otherwise unrelated visual update.
                branch_norms = {group.get('name', 'joint'): torch.nn.utils.clip_grad_norm_(
                    group['params'], config['grad_clip']) for group in optimizer.param_groups}
                grad_norm = torch.stack(list(branch_norms.values())).square().sum().sqrt()
                if not torch.isfinite(grad_norm):
                    raise RuntimeError('nonfinite gradients')
                optimizer.step()
                step += 1
                owners, strokes = prediction['owners'], prediction['strokes']
                pixels = geometry_metrics(prediction['geometry'], batch['points'][owners, strokes],
                                          batch['valid_point'][owners, strokes], batch['images'].shape)
                record = {'step': step, 'epoch': epoch, 'loss': float(loss.detach()), 'grad_norm': float(grad_norm),
                          'step_seconds': time.monotonic() - step_started,
                          'data_wait_seconds': data_wait,
                          'learning_rate': optimizer.param_groups[0]['lr'],
                          **{name+'_grad_norm': float(norm) for name, norm in branch_norms.items()},
                          **{k: float(v) for k, v in pixels.items()},
                          **{k: float(v.detach()) for k, v in terms.items()}}
                if device.type == 'cuda':
                    record['peak_cuda_allocated_mb'] = torch.cuda.max_memory_allocated(device)/1024**2
                    record['peak_cuda_reserved_mb'] = torch.cuda.max_memory_reserved(device)/1024**2
                stopper.add(record['loss'], len(batch['images']))
                log.write(json.dumps(record)+'\n')
                log.flush()
                print(json.dumps(record), flush=True)
                if step % checkpoint_every == 0 or (limit and step >= limit):
                    save(epoch, batch_index+1)
                if limit and step >= limit and batch_index+1 < len(loader):
                    finish('max_steps')
                    return
                batch_finished = time.monotonic()
            if not restored:
                restore_rng()
            if not stopper.samples:
                # Legacy checkpoints saved the last batch of an epoch without
                # monitor accumulators. Advance without inventing an epoch mean.
                stopper.last_epoch = epoch+1
                epoch_generator_state = generator.get_state()
                save(epoch+1, 0)
                skip_batches = 0
                continue
            mean, improved, stop = stopper.finish_epoch(epoch+1)
            epoch_record = dict(epoch=epoch+1, step=step, mean_training_loss=mean,
                                best_loss=stopper.best, bad_epochs=stopper.bad_epochs,
                                improved=improved)
            with (output/'epoch_metrics.jsonl').open('a') as epoch_log:
                epoch_log.write(json.dumps(epoch_record)+'\n')
            print('epoch_summary=' + json.dumps(epoch_record), flush=True)
            # Boundary checkpoint starts the next epoch, with its permutation RNG.
            epoch_generator_state = generator.get_state()
            save(epoch+1, 0)
            if improved:
                shutil.copyfile(output/'last.pt', output/'best.pt')
            if stop or (limit and step >= limit):
                finish('early_stopping' if stop else 'max_steps')
                return
            skip_batches = 0
    finish('max_epochs')


if __name__ == '__main__':
    main()
