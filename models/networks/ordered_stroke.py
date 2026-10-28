"""Independent ordered-instance stage; no end-point or stroke-type head.

Swin-T character features, a shared ResNet18 mask-history encoder, and causal
query decoding follow the reference's broad structure. Backbone variants,
GroupNorm, CLS cross-attention aggregation, FPN and loss details are local choices.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, swin_t
from torchvision.ops import StochasticDepth

from .single_stroke import spatial_position

MODE = 'ordered_stroke_xy_width_v1'


def group_norm(channels):
    return nn.GroupNorm(min(32, channels), channels)


class ResNetFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        net = resnet18(weights=None, norm_layer=group_norm)
        net.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.stages = nn.ModuleList([net.layer1, net.layer2, net.layer3, net.layer4])

    def forward(self, x):
        result = []
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
            result.append(x)
        return result


class CharacterEncoder(nn.Module):
    def __init__(self, dim, heads, encoder_layers, dropout, backbone):
        super().__init__()
        self.backbone_name = backbone
        if backbone == 'swin_t':
            net = swin_t(weights=None)
            for module in net.modules():
                if isinstance(module, StochasticDepth):
                    module.p = 0.
            net.features[0][0] = nn.Conv2d(1, 96, 4, stride=4)
            self.backbone = net.features
            channels = (96, 192, 384, 768)
        elif backbone == 'resnet18':
            self.backbone = ResNetFeatures()
            channels = (64, 128, 256, 512)
        else:
            raise ValueError('character backbone must be swin_t or resnet18')
        self.lateral = nn.ModuleList([nn.Conv2d(c, dim, 1) for c in channels])
        layer = nn.TransformerEncoderLayer(dim, heads, dim*4, dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, encoder_layers, norm=nn.LayerNorm(dim))
        self.smooth = nn.ModuleList([nn.Conv2d(dim, dim, 3, padding=1) for _ in channels])

    def forward(self, images):
        if self.backbone_name == 'swin_t':
            features, x = [], images
            for i, stage in enumerate(self.backbone):
                x = stage(x)
                if i in (1, 3, 5, 7):
                    features.append(x.permute(0, 3, 1, 2))
        else:
            features = self.backbone(images)
        features = [layer(x) for layer, x in zip(self.lateral, features)]
        top = features[-1]
        b, d, h, w = top.shape
        tokens = top.flatten(2).transpose(1, 2) + spatial_position(h, w, d, top)
        features[-1] = self.encoder(tokens).transpose(1, 2).reshape(b, d, h, w)
        for i in range(2, -1, -1):
            features[i] = features[i] + F.interpolate(features[i+1], features[i].shape[-2:], mode='nearest')
        features = [layer(x) for layer, x in zip(self.smooth, features)]
        # Three character memory scales; P2 is reserved for the dense mask head.
        memory = []
        for feature in features[:0:-1]:
            _, d, h, w = feature.shape
            memory.append(feature.flatten(2).transpose(1, 2) + spatial_position(h, w, d, feature))
        return torch.cat(memory, 1), features[0]


class HistoryEncoder(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        # BatchNorm across packed history would leak future masks during training.
        self.backbone = ResNetFeatures()
        self.projections = nn.ModuleList([nn.Conv2d(c, dim, 1) for c in (128, 256, 512)])
        self.cls = nn.Parameter(torch.zeros(3, 1, dim))
        self.attention = nn.ModuleList([nn.MultiheadAttention(dim, heads, dropout=dropout,
                                                            batch_first=True) for _ in range(3)])
        self.norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])
        self.ffn = nn.ModuleList([nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(),
                                                nn.Linear(dim*4, dim)) for _ in range(3)])
        self.fuse = nn.Linear(dim*3, dim)

    def forward(self, masks):
        features = self.backbone(masks[:, None])[-3:]
        result = []
        for i, (feature, projection) in enumerate(zip(features, self.projections)):
            feature = projection(feature)
            b, d, h, w = feature.shape
            tokens = feature.flatten(2).transpose(1, 2) + spatial_position(h, w, d, feature)
            cls = self.cls[i:i+1].expand(b, -1, -1)
            cls = cls + self.attention[i](cls, tokens, tokens, need_weights=False)[0]
            cls = cls + self.ffn[i](self.norm[i](cls))
            result.append(cls[:, 0])
        return self.fuse(torch.cat(result, -1))


class OrderedStrokeModel(nn.Module):
    def __init__(self, dim=64, heads=4, encoder_layers=2, decoder_layers=9,
                 max_strokes=64, dropout=0., character_backbone='swin_t', history_chunk_size=16):
        super().__init__()
        if dim < 4 or dim % 4 or dim % heads or min(encoder_layers, decoder_layers, max_strokes, history_chunk_size) < 1:
            raise ValueError('invalid model dimensions, layers or caps')
        self.dim, self.max_strokes, self.history_chunk_size = dim, max_strokes, history_chunk_size
        self.visual = CharacterEncoder(dim, heads, encoder_layers, dropout, character_backbone)
        self.history_encoder = HistoryEncoder(dim, heads, dropout)
        self.bos = nn.Parameter(torch.zeros(1, 1, dim))
        self.position = nn.Embedding(max_strokes+1, dim)
        self.layers = nn.ModuleList([nn.TransformerDecoderLayer(dim, heads, dim*4, dropout,
                                      batch_first=True, norm_first=True) for _ in range(decoder_layers)])
        self.norm = nn.LayerNorm(dim)
        self.mask_head = nn.Linear(dim, dim)
        self.box_head = nn.Linear(dim, 4)
        self.start_xy_head = nn.Linear(dim, 2)
        self.start_width_head = nn.Linear(dim, 1)
        self.valid_head = nn.Linear(dim, 1)
        nn.init.constant_(self.start_width_head.bias, -3.)

    def encode_history(self, masks, active):
        b, n, h, w = masks.shape
        if active.shape != (b, n) or active.dtype != torch.bool or n > self.max_strokes:
            raise ValueError('invalid history shape or cap')
        if n and (active[:, 1:] & ~active[:, :-1]).any():
            raise ValueError('history padding must be a suffix')
        flat = masks[active]
        values = [self.history_encoder(chunk) for chunk in flat.split(self.history_chunk_size) if len(chunk)]
        result = masks.new_zeros(b, n, self.dim)
        if values:
            result[active] = torch.cat(values)
        return result

    def heads(self, x, pixel_features):
        x = self.norm(x)
        center, size = self.box_head(x).sigmoid().chunk(2, -1)
        boxes = torch.cat([(center-size/2).clamp(0, 1), (center+size/2).clamp(0, 1)], -1)
        return dict(mask_logits=torch.einsum('bnd,bdhw->bnhw', self.mask_head(x), pixel_features),
                    boxes=boxes, starts=torch.cat([self.start_xy_head(x).sigmoid(),
                    F.softplus(self.start_width_head(x))+1e-6], -1),
                    valid_logits=self.valid_head(x).squeeze(-1))

    def decode(self, visual, history, history_active, auxiliary=False):
        memory, pixels = visual
        b, n, _ = history.shape
        if history_active.shape != (b, n) or n > self.max_strokes:
            raise ValueError('invalid encoded history')
        x = torch.cat([self.bos.expand(b, -1, -1), history], 1)
        x = x + self.position(torch.arange(n+1, device=x.device))[None]
        padding = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=x.device), ~history_active], 1)
        causal = torch.ones(n+1, n+1, dtype=torch.bool, device=x.device).triu(1)
        outputs = []
        for i, layer in enumerate(self.layers):
            x = layer(x, memory, tgt_mask=causal, tgt_key_padding_mask=padding)
            if auxiliary or i == len(self.layers)-1:
                outputs.append(self.heads(x, pixels))
        output = outputs[-1]
        if auxiliary:
            output['auxiliary'] = outputs[:-1]
        return output

    def forward(self, images, history_masks, history_active):
        history = self.encode_history(history_masks, history_active)
        return self.decode(self.visual(images), history, history_active, auxiliary=True)

    @torch.no_grad()
    def generate(self, images, max_strokes=None, threshold=.5):
        """Image-only inference; retain probabilities, feed back binary masks."""
        if self.training:
            raise ValueError('call eval() before generation')
        cap = self.max_strokes if max_strokes is None else max_strokes
        if not 1 <= cap <= self.max_strokes or not 0 < threshold < 1:
            raise ValueError('invalid cap or validity threshold')
        if images.shape[1:] != (1, 256, 256):
            raise ValueError('expected Bx1x256x256 normalized grayscale')
        visual = self.visual(images)
        live = torch.arange(len(images), device=images.device)
        history = images.new_zeros(len(images), 0, self.dim)
        collected = [dict(masks=[], boxes=[], starts=[], confidences=[]) for _ in images]
        results = [None] * len(images)
        while len(live):
            current = tuple(v.index_select(0, live) for v in visual)
            active = torch.ones(history.shape[:2], dtype=torch.bool, device=images.device)
            output = self.decode(current, history, active)
            confidence = output['valid_logits'][:, -1].sigmoid()
            masks = F.interpolate(output['mask_logits'][:, -1:], (256, 256),
                                  mode='bilinear', align_corners=False)[:, 0].sigmoid()
            keep = []
            for j, original in enumerate(live.tolist()):
                data = collected[original]
                stopped = bool(confidence[j] < threshold)
                capped = len(data['masks']) >= cap
                if stopped or capped:
                    results[original] = dict(
                        masks=torch.stack(data['masks']) if data['masks'] else images.new_zeros(0, 256, 256),
                        boxes=torch.stack(data['boxes']) if data['boxes'] else images.new_zeros(0, 4),
                        starts=torch.stack(data['starts']) if data['starts'] else images.new_zeros(0, 3),
                        confidences=torch.stack(data['confidences']) if data['confidences'] else images.new_zeros(0),
                        terminated=stopped, truncated=not stopped, stop_reason='invalid' if stopped else 'stroke_cap')
                else:
                    data['masks'].append(masks[j].clone())
                    data['boxes'].append(output['boxes'][j, -1].clone())
                    data['starts'].append(output['starts'][j, -1].clone())
                    data['confidences'].append(confidence[j].clone())
                    keep.append(j)
            if not keep:
                break
            indices = torch.tensor(keep, device=images.device)
            feedback = self.history_encoder((masks[indices] >= .5).to(images.dtype))
            history = torch.cat([history[indices], feedback[:, None]], 1)
            live = live[indices]
        return results
