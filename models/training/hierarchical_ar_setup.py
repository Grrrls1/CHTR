"""Versioned checkpoint contract for the independent three-task model."""
import hashlib
import json
from pathlib import Path
import torch
from models.networks.hierarchical_ar import HierarchicalAR,MODE,VERSION


def local_path(value):
    p=Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (Path(__file__).resolve().parents[1]/p).resolve()


def read_checkpoint(path):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck.get('model_mode')!=MODE or ck.get('model_version')!=VERSION:
        raise ValueError('not a compatible three-task checkpoint; legacy models cannot be resumed')
    if ck.get('config',{}).get('training_mode')!=MODE:
        raise ValueError('checkpoint training mode differs')
    return ck


def create_model(config,device,checkpoint=None):
    if config.get('training_mode')!=MODE:raise ValueError('requires hierarchical_ar_three_task mode')
    if checkpoint is not None and checkpoint['config']['model']!=config['model']:
        raise ValueError('checkpoint architecture differs')
    model=HierarchicalAR(**config['model']).to(device)
    if checkpoint is not None:model.load_state_dict(checkpoint['model'])
    return model


def training_signature(config,dataset_digest,split):
    excluded={'device','output_dir','epochs','max_steps'}
    settings={k:v for k,v in config.items() if k not in excluded}
    payload=dict(config=settings,dataset_digest=dataset_digest,split=split)
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def write_json(path,value):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temp.replace(path)
