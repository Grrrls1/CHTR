"""Explicit architecture and optional checkpoint loading for free-rollout runs."""
import torch
from models.networks.model import StrokeTrajectoryModel


def create_model(config, device, checkpoint_path=None):
    if not isinstance(config.get('model'), dict):
        raise ValueError('free rollout requires an explicit model configuration')
    if config['model'].get('pretrained', False):
        raise ValueError('free rollout initialization requires pretrained=false')
    model = StrokeTrajectoryModel(**config['model']).to(device)
    checkpoint = None
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if checkpoint['config']['model'] != config['model']:
            raise ValueError('checkpoint architecture differs from configured model')
        model.load_state_dict(checkpoint['model'])
    return model, checkpoint
