"""Hierarchical ordered stroke and point model; no offset prediction head."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def position_2d(h, w, dim, device, dtype):
    if dim % 4:
        raise ValueError('hidden dimension must be divisible by four')
    freq = torch.exp(torch.arange(dim//4, device=device, dtype=dtype) * (-math.log(10000)/(dim//4)))
    y, x = torch.meshgrid(torch.arange(h, device=device, dtype=dtype),
                          torch.arange(w, device=device, dtype=dtype), indexing='ij')
    return torch.cat([(x[..., None]*freq).sin(), (x[..., None]*freq).cos(),
                      (y[..., None]*freq).sin(), (y[..., None]*freq).cos()], -1).reshape(1, h*w, dim)


def block(cin, cout, stride):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
                         nn.GroupNorm(8, cout), nn.GELU(),
                         nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
                         nn.GroupNorm(8, cout), nn.GELU())


class CNNFPN(nn.Module):
    """Download-free CNN baseline; optional torchvision ResNet-18 backbone."""
    def __init__(self, dim, backbone='cnn', pretrained=False):
        super().__init__()
        self.backbone = backbone
        if backbone == 'cnn':
            if pretrained:
                raise ValueError('custom cnn has no pretrained weights')
            self.stem = block(1, 32, 2)
            self.stages = nn.ModuleList([block(32, 64, 2), block(64, 128, 2), block(128, 256, 2)])
            channels = [64, 128, 256]
        elif backbone == 'resnet18':
            from torchvision.models import resnet18, ResNet18_Weights
            net = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
            self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
            self.stages = nn.ModuleList([net.layer1, net.layer2, net.layer3])
            channels = [64, 128, 256]
        else:
            raise ValueError('implemented backbones: cnn, resnet18')
        self.lateral = nn.ModuleList([nn.Conv2d(c, dim, 1) for c in channels])
        self.smooth = nn.ModuleList([nn.Conv2d(dim, dim, 3, padding=1) for _ in channels])

    def forward(self, image):
        if self.backbone == 'resnet18':
            image = image.repeat(1, 3, 1, 1)
            mean = image.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
            std = image.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
            image = (image-mean)/std
        x = self.stem(image)
        feats = []
        for layer in self.stages:
            x = layer(x)
            feats.append(x)
        pyramid = [layer(f) for layer, f in zip(self.lateral, feats)]
        for i in range(len(pyramid)-2, -1, -1):
            pyramid[i] = pyramid[i] + F.interpolate(pyramid[i+1], size=pyramid[i].shape[-2:], mode='bilinear', align_corners=False)
        return [layer(f) for layer, f in zip(self.smooth, pyramid)]


class PointLayer(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])
        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim*4, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory, valid, bias):
        length = x.shape[1]
        causal = torch.ones(length, length, device=x.device, dtype=torch.bool).triu(1)
        q = self.norms[0](x)
        x = x + self.drop(self.self_attn(q, q, q, attn_mask=causal, key_padding_mask=~valid, need_weights=False)[0])
        q = self.norms[1](x)
        # Per-stroke, per-head additive soft mask bias; never hard-mask background.
        attn_bias = bias[:, None, None, :].expand(-1, self.heads, length, -1).reshape(-1, length, memory.shape[1])
        x = x + self.drop(self.cross_attn(q, memory, memory, attn_mask=attn_bias, need_weights=False)[0])
        return x + self.drop(self.ffn(self.norms[2](x)))


class StrokeTrajectoryModel(nn.Module):
    def __init__(self, dim=64, heads=4, stroke_layers=2, point_layers=2,
                 max_strokes=64, max_points=1024, dropout=0.1,
                 backbone='cnn', pretrained=False, mask_alpha=1.0, memory_size=16,
                 separate_branches=False):
        super().__init__()
        if dim % 8 or dim % heads:
            raise ValueError('dim must be divisible by 8 and heads')
        self.dim, self.max_strokes, self.max_points = dim, max_strokes, max_points
        self.mask_alpha, self.memory_size = mask_alpha, memory_size
        self.separate_branches = separate_branches
        self.encoder = CNNFPN(dim, backbone, pretrained)
        self.stroke_bos = nn.Parameter(torch.zeros(1, 1, dim))
        self.stroke_position = nn.Embedding(max_strokes, dim)
        layer = nn.TransformerDecoderLayer(dim, heads, dim*4, dropout, batch_first=True, norm_first=True)
        self.stroke_decoder = nn.TransformerDecoder(layer, stroke_layers, norm=nn.LayerNorm(dim))
        self.mask_query = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.type_head = nn.Linear(dim, 24)
        self.type_embedding = nn.Parameter(torch.randn(24, dim)*0.02)
        self.last_head = nn.Linear(dim, 1)
        self.endpoint_head = nn.Linear(dim, 4)
        self.condition = nn.Linear(dim*2, dim)
        self.point_embedding = nn.Linear(6, dim)
        self.point_bos = nn.Parameter(torch.zeros(1, 1, dim))
        self.point_position = nn.Embedding(max_points, dim)
        self.point_layers = nn.ModuleList([PointLayer(dim, heads, dropout) for _ in range(point_layers)])
        self.point_norm = nn.LayerNorm(dim)
        self.geometry_head = nn.Linear(dim, 3)
        self.state_head = nn.Linear(dim, 3)

    def branch_parameter_groups(self):
        """Disjoint optimizer/clipping groups; category embeddings belong to points."""
        visual_prefixes = ('encoder.', 'stroke_bos', 'stroke_position.', 'stroke_decoder.',
                           'mask_query.', 'type_head.', 'last_head.', 'endpoint_head.')
        visual, point = [], []
        for name, parameter in self.named_parameters():
            (visual if name.startswith(visual_prefixes) else point).append(parameter)
        return [{'params': visual, 'name': 'visual'}, {'params': point, 'name': 'point'}]

    def forward(self, batch, condition_mode='predicted'):
        if condition_mode not in ('predicted', 'ground_truth'):
            raise ValueError('invalid condition mode')
        images = batch['images']
        b, k, length, _ = batch['points'].shape
        if k > self.max_strokes or length > self.max_points:
            raise ValueError('sequence exceeds configured limits; increase limits (no silent truncation)')
        pyramid = self.encoder(images)
        pixels = pyramid[0]
        # Bounded visual memory for long point sequences; both scales retained.
        sizes = [self.memory_size, max(1, self.memory_size//2)]
        memories = []
        for feat, size in zip(pyramid[-2:], sizes):
            feat = F.adaptive_avg_pool2d(feat, (size, size))
            memories.append(feat.flatten(2).transpose(1, 2) + position_2d(size, size, self.dim, feat.device, feat.dtype))
        memory = torch.cat(memories, 1)
        history, zs = [], []
        for i in range(k):
            prefix = torch.cat([self.stroke_bos.expand(b, -1, -1)] + [z[:, None] for z in history], 1)
            prefix = prefix + self.stroke_position(torch.arange(i+1, device=images.device))[None]
            causal = torch.ones(i+1, i+1, device=images.device, dtype=torch.bool).triu(1)
            z = self.stroke_decoder(prefix, memory, tgt_mask=causal)[:, -1]
            history.append(z)
            zs.append(z)
        z = torch.stack(zs, 1)
        mask_logits = torch.einsum('bkd,bdhw->bkhw', self.mask_query(z), pixels)/math.sqrt(self.dim)
        mask_logits = F.interpolate(mask_logits, size=images.shape[-2:], mode='bilinear', align_corners=False)
        type_logits = self.type_head(z)
        last_logits = self.last_head(z).squeeze(-1)
        endpoints = self.endpoint_head(z).sigmoid()
        # Only decode real strokes: padded strokes must never create all-masked attention.
        owners, strokes = batch['valid_stroke'].nonzero(as_tuple=True)
        selected_z = z[owners, strokes]
        if condition_mode == 'ground_truth':
            type_prob = F.one_hot(batch['types'][owners, strokes], 24).to(images.dtype)
            selected_masks = batch['masks'][owners, strokes]
        else:
            type_prob = type_logits[owners, strokes].softmax(-1)
            selected_masks = mask_logits[owners, strokes].sigmoid()
        # Stop every upstream route: visual losses still train the original tensors.
        # The type embedding belongs to the point branch and remains trainable.
        point_memory = memory
        if self.separate_branches:
            selected_z = selected_z.detach()
            type_prob = type_prob.detach()
            selected_masks = selected_masks.detach()
            point_memory = memory.detach()
        cond = self.condition(torch.cat([selected_z, type_prob @ self.type_embedding], -1))
        targets = batch['points'][owners, strokes]
        previous = self.point_embedding(targets[:, :-1])
        x = torch.cat([self.point_bos.expand(len(owners), -1, -1), previous], 1)
        x = x + self.point_position(torch.arange(length, device=images.device))[None] + cond[:, None]
        biases = []
        for size in sizes:
            prob = F.interpolate(selected_masks[:, None], size=(size, size), mode='bilinear', align_corners=False).flatten(1)
            biases.append(self.mask_alpha * prob.clamp_min(1e-4).log())
        bias = torch.cat(biases, 1)
        valid = batch['valid_point'][owners, strokes]
        for layer in self.point_layers:
            x = layer(x, point_memory[owners], valid, bias)
        raw = self.geometry_head(self.point_norm(x))
        geometry = torch.cat([raw[..., :2].sigmoid(), F.softplus(raw[..., 2:])+1e-6], -1)
        states = self.state_head(self.point_norm(x))
        return {'geometry': geometry, 'state_logits': states, 'mask_logits': mask_logits,
                'type_logits': type_logits, 'last_logits': last_logits, 'endpoints': endpoints,
                'owners': owners, 'strokes': strokes}
