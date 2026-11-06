"""Shared stage-two setup, fixed subset selection and checkpoint loading."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from models.datasets.single_stroke import StrokeCatalog, SingleStrokeDataset
from models.networks.single_stroke import SingleStrokeModel, MODE

ROOT=Path(__file__).resolve().parents[2]


def local_path(value):
    p=Path(value)
    return p if p.is_absolute() else ROOT/p


def to_device(batch,device):
    return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}


def make_catalog(config):
    backend = config.get('dataset_backend', 'calligraphy_lmdb')
    if backend == 'casia_pot':
        from models.datasets.casia_training import CasiaCatalog
        return CasiaCatalog(local_path(config['dataset']), local_path(config['split_file']), config.get('render'))
    if backend != 'calligraphy_lmdb':
        raise ValueError('unknown dataset backend: '+backend)
    return StrokeCatalog(local_path(config['dataset']), local_path(config['split_file']))


def make_data(config):
    catalog=make_catalog(config)
    train=SingleStrokeDataset(catalog,'train',config['model']['max_points'],config['augment'])
    val=SingleStrokeDataset(catalog,'val',config['model']['max_points'])
    def count(sid):
        return catalog.stroke_counts[sid] if hasattr(catalog,'stroke_counts') else catalog.rows[sid]['stroke_count']
    selected_chars=train.ids
    limit=config['train_char_limit']
    if limit:
        ordered=sorted(selected_chars,key=lambda s:(count(s),s))
        selected_chars=[ordered[int(i)] for i in np.linspace(0,len(ordered)-1,min(limit,len(ordered)),dtype=int)]
    selected_chars=set(selected_chars)
    indices=[]; lengths=[]
    for ci,sid in enumerate(train.ids):
        if sid not in selected_chars: continue
        start=train.ends[ci-1] if ci else 0
        if not config['train_stroke_limit']:
            indices.extend(range(start,train.ends[ci]))
        else:
            for s in sorted(catalog.rows[sid]['stroke_map'],key=lambda s:s['stroke_index']):
                indices.append(start+s['stroke_index']);lengths.append(s['point_range'][1]-s['point_range'][0])
    if config['train_stroke_limit']:
        # Fixed length coverage for diagnostic subsets, never truncate labels.
        ordered=[i for _,i in sorted(zip(lengths,indices))]
        subset_count=min(config['train_stroke_limit'],len(ordered))
        positions=[len(ordered)//2] if subset_count==1 else np.linspace(0,len(ordered)-1,subset_count,dtype=int)
        indices=[ordered[int(i)] for i in positions]
    # Model selection always covers the full held-out split.
    val_indices=range(len(val))
    if config.get('eval_char_limit', 0):
        limit = config['eval_char_limit']
        if type(limit) is not int or limit < 1:
            raise ValueError('eval_char_limit must be positive when set')
        ordered = sorted(val.ids, key=lambda s:(count(s),s))
        chosen = {ordered[int(i)] for i in np.linspace(0,len(ordered)-1,min(limit,len(ordered)),dtype=int)}
        val_indices = [i for ci,sid in enumerate(val.ids) if sid in chosen
                       for i in range(val.ends[ci-1] if ci else 0,val.ends[ci])]
    return catalog,train,val,indices,val_indices


def signature(config,catalog):
    settings={k:v for k,v in config.items() if k not in ('output_dir','device','threads','epochs','max_steps')}
    payload=dict(config=settings,dataset_fingerprint=catalog.fingerprint,
                 splits={n:catalog.split[n+'_ids'] for n in ('train','val','test')})
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def load_model(checkpoint,device):
    # Checkpoints are trusted local files, never remote/untrusted pickle payloads.
    ck=torch.load(checkpoint,map_location=device,weights_only=False)
    if ck.get('training_mode')!=MODE:raise ValueError('not a single-stroke checkpoint')
    model=SingleStrokeModel(**ck['config']['model']).to(device)
    model.load_state_dict(ck['model'])
    return model,ck
