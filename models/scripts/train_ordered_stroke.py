"""Independent stage-one teacher-forced training with image-only validation."""
import argparse
import json
from pathlib import Path
import random
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.datasets.ordered_stroke import collate_ordered_strokes
from models.networks.ordered_stroke import OrderedStrokeModel, MODE
from models.losses.ordered_stroke import ordered_stroke_loss
from models.training.ordered_stroke import (local_path, to_device, make_catalog, make_dataset,
                                            select_indices, signature, atomic_checkpoint)
from models.training.ordered_stroke_evaluation import evaluate_split


def validate_config(config):
    if config['training_mode'] != MODE:
        raise ValueError('incorrect training mode')
    for k in ('threads', 'batch_size', 'epochs', 'checkpoint_every', 'eval_every_epochs', 'eval_batch_size', 'decode_max_strokes'):
        if type(config[k]) is not int or config[k] < 1:
            raise ValueError(k+' must be a positive integer')
    for k in ('max_steps', 'train_char_limit', 'eval_char_limit', 'visualize_characters'):
        if type(config[k]) is not int or config[k] < 0:
            raise ValueError(k+' must be nonnegative')
    if config['decode_max_strokes'] > config['model']['max_strokes'] or not 0 < config['validity_threshold'] < 1:
        raise ValueError('invalid decoding settings')
    for k in ('learning_rate', 'grad_clip'):
        if not np.isfinite(config[k]) or config[k] <= 0:
            raise ValueError(k+' must be positive and finite')
    if not np.isfinite(config['weight_decay']) or config['weight_decay'] < 0:
        raise ValueError('weight_decay must be finite and nonnegative')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='models/configs/ordered_stroke.json')
    parser.add_argument('--output-dir')
    parser.add_argument('--device')
    parser.add_argument('--resume')
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--epochs', type=int)
    args = parser.parse_args()
    config = json.loads(local_path(args.config).read_text())
    for k in ('output_dir', 'device', 'max_steps', 'epochs'):
        if getattr(args, k) is not None:
            config[k] = getattr(args, k)
    validate_config(config)
    torch.set_num_threads(config['threads'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if config['device'] == 'auto' else config['device'])
    output = local_path(config['output_dir']).resolve()
    if args.resume:
        if local_path(args.resume).resolve().parent != output:
            raise ValueError('resume must use the same run directory')
    elif output.exists() and any(output.iterdir()):
        raise ValueError('new run requires an empty output directory')
    print('[ordered-train] loading catalog and validating fixed split', flush=True)
    catalog = make_catalog(config)
    try:
        train, val = [make_dataset(catalog, split, config) for split in ('train', 'val')]
        train_indices = select_indices(train, config['train_char_limit'])
        val_indices = select_indices(val, config['eval_char_limit'])
        sig = signature(config, catalog)
        model = OrderedStrokeModel(**config['model']).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        step = epoch = cursor = last_evaluated_epoch = 0
        best = None
        if args.resume:
            ck = torch.load(local_path(args.resume), map_location=device, weights_only=False)
            if ck.get('training_mode') != MODE or ck.get('signature') != sig:
                raise ValueError('resume data/split/model/training settings changed')
            model.load_state_dict(ck['model'])
            optimizer.load_state_dict(ck['optimizer'])
            step, epoch, cursor = ck['step'], ck['epoch'], ck['cursor']
            best, last_evaluated_epoch = ck['best_val_ordered_iou'], ck['last_evaluated_epoch']
            torch.set_rng_state(ck['rng']['torch'].cpu())
            random.setstate(ck['rng']['python'])
            np.random.set_state(ck['rng']['numpy'])
            if ck['rng']['cuda'] is not None and device.type == 'cuda':
                torch.cuda.set_rng_state_all([state.cpu() for state in ck['rng']['cuda']])
        output.mkdir(parents=True, exist_ok=True)
        (output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
        usage = dict(training_mode=MODE, dataset_fingerprint=catalog.fingerprint, signature=sig,
                     train_ids=[train.ids[i] for i in train_indices], val_ids=[val.ids[i] for i in val_indices],
                     full_train=len(train_indices) == len(train), full_val=len(val_indices) == len(val),
                     test_used_for_model_selection=False, geometry_condition='start XYW only; no end point',
                     history='BOS + preceding binary masks, causal; K valid targets + one invalid')
        (output/'data_usage.json').write_text(json.dumps(usage, ensure_ascii=False, indent=2)+'\n')

        def save():
            atomic_checkpoint(output/'last.pt', dict(training_mode=MODE, config=config, signature=sig,
                model=model.state_dict(), optimizer=optimizer.state_dict(), step=step, epoch=epoch, cursor=cursor,
                best_val_ordered_iou=best, last_evaluated_epoch=last_evaluated_epoch,
                rng=dict(torch=torch.get_rng_state(), python=random.getstate(), numpy=np.random.get_state(),
                         cuda=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None)))

        model.train()
        order_epoch, order = None, None
        with (output/'metrics.jsonl').open('a') as log:
            while epoch < config['epochs'] and (not config['max_steps'] or step < config['max_steps']):
                if order_epoch != epoch:
                    order = torch.randperm(len(train_indices), generator=torch.Generator().manual_seed(config['seed']+epoch)).tolist()
                    order_epoch = epoch
                positions = order[cursor:cursor+config['batch_size']]
                batch = to_device(collate_ordered_strokes([train[train_indices[i]] for i in positions]), device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(batch['images'], batch['history_masks'], batch['history_active'])
                loss, terms = ordered_stroke_loss(prediction, batch, config['loss_weights'], config['auxiliary_weight'])
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite loss')
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                if not torch.isfinite(grad_norm):
                    raise ValueError('nonfinite gradients')
                optimizer.step()
                step += 1
                cursor += len(positions)
                if cursor == len(train_indices):
                    epoch += 1
                    cursor = 0
                row = dict(kind='train', step=step, epoch=epoch, cursor=cursor, grad_norm=float(grad_norm),
                           **{k: float(v.detach()) for k, v in terms.items()})
                log.write(json.dumps(row)+'\n')
                log.flush()
                print(json.dumps(row), flush=True)
                del prediction, loss, terms, batch
                if cursor == 0 and epoch % config['eval_every_epochs'] == 0:
                    target = output/f'eval_epoch_{epoch:03d}'
                    if target.exists():
                        # A crash after evaluation but before checkpoint commit can leave
                        # an uncommitted report. Preserve it before deterministic replay.
                        backup = output/f'eval_epoch_{epoch:03d}_uncommitted_{step}'
                        suffix = 0
                        while backup.exists():
                            suffix += 1
                            backup = output/f'eval_epoch_{epoch:03d}_uncommitted_{step}_{suffix}'
                        target.rename(backup)
                    report = evaluate_split(model, val, device, config, target, val_indices,
                                            checkpoint=dict(epoch=epoch, step=step))
                    score = report['free']['ordered_mask_iou']
                    improved = best is None or score > best
                    if improved:
                        best = score
                    last_evaluated_epoch = epoch
                    record = dict(kind='validation', epoch=epoch, step=step, full_split=report['full_split'],
                                  free=report['free'], best_val_ordered_iou=best)
                    log.write(json.dumps(record)+'\n')
                    log.flush()
                    save()
                    if improved:
                        shutil.copyfile(output/'last.pt', output/'best.pt')
                elif step % config['checkpoint_every'] == 0 or cursor == 0:
                    save()
            save()
        status = dict(step=step, epoch=epoch, cursor=cursor, best_val_ordered_iou=best,
                      last_evaluated_epoch=last_evaluated_epoch, device=str(device),
                      parameters=sum(p.numel() for p in model.parameters()),
                      reason='epochs' if epoch >= config['epochs'] else 'max_steps')
        (output/'training_status.json').write_text(json.dumps(status, indent=2)+'\n')
        print(json.dumps(status), flush=True)
    finally:
        catalog.close()


if __name__ == '__main__':
    main()
