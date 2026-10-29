"""Stage-one setup and trusted local checkpoint handling."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from models.datasets.single_stroke import StrokeCatalog
from models.datasets.ordered_stroke import OrderedStrokeDataset
from models.networks.ordered_stroke import OrderedStrokeModel, MODE
from models.training.single_stroke import local_path, to_device
from models.training.single_stroke import load_model as load_point_model


def select_indices(dataset, limit):
    ordered = sorted(range(len(dataset)), key=lambda i: (
        dataset.catalog.rows[dataset.ids[i]]['stroke_count'], dataset.ids[i]))
    if not limit or limit >= len(ordered):
        return ordered
    return [ordered[int(i)] for i in np.linspace(0, len(ordered)-1, limit, dtype=int)]


def make_catalog(config):
    return StrokeCatalog(local_path(config['dataset']), local_path(config['split_file']))


def make_dataset(catalog, split, config):
    return OrderedStrokeDataset(catalog, split, config['model']['max_strokes'])


def signature(config, catalog):
    excluded = ('output_dir', 'device', 'threads', 'epochs', 'max_steps')
    payload = dict(config={k: v for k, v in config.items() if k not in excluded},
                   fingerprint=catalog.fingerprint, split=catalog.split)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_model(checkpoint, device):
    ck = torch.load(local_path(checkpoint), map_location=device, weights_only=False)
    if ck.get('training_mode') != MODE:
        raise ValueError('not an ordered-stroke checkpoint')
    model = OrderedStrokeModel(**ck['config']['model']).to(device)
    model.load_state_dict(ck['model'])
    return model, ck


def atomic_checkpoint(path, state):
    path = Path(path)
    temp = path.with_name(path.name+'.tmp')
    torch.save(state, temp)
    temp.replace(path)


def load_point_stage(config, device):
    path=config.get('point_checkpoint')
    if not path:
        raise ValueError('point_checkpoint is required: validation and test always run both stages')
    model,ck=load_point_model(local_path(path),device)
    model.eval()
    model.requires_grad_(False)
    config.setdefault('point_decode_max_points',ck['config'].get('decode_max_points',model.max_points))
    config.setdefault('point_validity_threshold',ck['config'].get('validity_threshold',.5))
    config.setdefault('point_batch_size',16)
    if (type(config['point_decode_max_points']) is not int or type(config['point_batch_size']) is not int
            or not 1<=config['point_decode_max_points']<=model.max_points
            or not 0<config['point_validity_threshold']<1 or config['point_batch_size']<1):
        raise ValueError('invalid point-stage decoding configuration')
    return model,dict(path=str(local_path(path)),epoch=ck.get('epoch'),step=ck.get('step'))
