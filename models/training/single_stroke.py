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


def make_data(config):
    catalog=StrokeCatalog(local_path(config['dataset']),local_path(config['split_file']))
    train=SingleStrokeDataset(catalog,'train',config['model']['max_points'],config['augment'])
    val=SingleStrokeDataset(catalog,'val',config['model']['max_points'])
    selected_chars=train.ids
    limit=config['train_char_limit']
    if limit:
        ordered=sorted(selected_chars,key=lambda s:(catalog.rows[s]['stroke_count'],s))
        selected_chars=[ordered[int(i)] for i in np.linspace(0,len(ordered)-1,min(limit,len(ordered)),dtype=int)]
    selected_chars=set(selected_chars)
    indices=[]; lengths=[]
    for ci,sid in enumerate(train.ids):
        if sid not in selected_chars: continue
        start=train.ends[ci-1] if ci else 0
        for s in sorted(catalog.rows[sid]['stroke_map'],key=lambda s:s['stroke_index']):
            indices.append(start+s['stroke_index']);lengths.append(s['point_range'][1]-s['point_range'][0])
    if config['train_stroke_limit']:
        # Fixed length coverage for diagnostic subsets, never truncate labels.
        ordered=[i for _,i in sorted(zip(lengths,indices))]
        count=min(config['train_stroke_limit'],len(ordered))
        positions=[len(ordered)//2] if count==1 else np.linspace(0,len(ordered)-1,count,dtype=int)
        indices=[ordered[int(i)] for i in positions]
    # Model selection always covers the full held-out split.
    val_indices=range(len(val))
    return catalog,train,val,indices,val_indices


def signature(config,catalog):
    settings={k:v for k,v in config.items() if k not in ('output_dir','device','threads','epochs','max_steps')}
    payload=dict(config=settings,dataset_fingerprint=catalog.fingerprint,
                 splits={n:catalog.split[n+'_ids'] for n in ('train','val','test')})
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def load_model(checkpoint,device):
    # Checkpoints are trusted local files, never remote/untrusted pickle payloads.
    ck=torch.load(checkpoint,map_location=device)
    if ck.get('training_mode')!=MODE:raise ValueError('not a single-stroke checkpoint')
    model=SingleStrokeModel(**ck['config']['model']).to(device)
    model.load_state_dict(ck['model'])
    return model,ck
