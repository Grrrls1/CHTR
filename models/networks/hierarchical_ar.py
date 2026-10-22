"""Independent three-task AR model: label-length training, two-level stopping.

Only reusable backbone/attention building blocks are imported from the legacy
implementation. No legacy model, mask/type heads, or legacy losses are created.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .model import CNNFPN, PointLayer, position_2d
from .autoregressive import attention_step

MODE = 'hierarchical_ar_three_task'
VERSION = 1


def _select_caches(caches, indices):
    return [{kind: {key: value.index_select(0, indices) for key, value in cache.items()}
             for kind, cache in layer.items()} for layer in caches]


class HierarchicalAR(nn.Module):
    def __init__(self, dim=128, heads=4, stroke_layers=2, point_layers=2,
                 backbone='resnet18', pretrained=False, memory_size=8,
                 max_strokes=16, max_points=384, dropout=0.):
        super().__init__()
        if pretrained:
            raise ValueError('three-task baseline requires random initialization (pretrained=false)')
        if dropout != 0.:
            raise ValueError('cached full-gradient AR requires dropout=0')
        if dim < 8 or dim % 8 or heads < 1 or dim % heads:
            raise ValueError('dim must be divisible by 8 and heads')
        if min(stroke_layers, point_layers, memory_size, max_strokes, max_points) < 1:
            raise ValueError('layer counts, memory size and decode caps must be positive')
        self.dim, self.memory_size = dim, memory_size
        self.max_strokes, self.max_points = max_strokes, max_points
        self.encoder = CNNFPN(dim, backbone, pretrained=False)
        self.stroke_bos = nn.Parameter(torch.zeros(1,1,dim))
        self.stroke_position = nn.Embedding(max_strokes,dim)
        layer = nn.TransformerDecoderLayer(dim,heads,4*dim,dropout=0.,batch_first=True,norm_first=True)
        self.stroke_decoder = nn.TransformerDecoder(layer,stroke_layers,norm=nn.LayerNorm(dim))
        self.condition = nn.Linear(dim,dim)
        self.stroke_end_head = nn.Linear(dim,1)
        self.point_embedding = nn.Linear(3,dim)
        self.point_bos = nn.Parameter(torch.zeros(1,1,dim))
        self.point_position = nn.Embedding(max_points,dim)
        self.point_layers = nn.ModuleList([PointLayer(dim,heads,0.) for _ in range(point_layers)])
        self.point_norm = nn.LayerNorm(dim)
        self.geometry_head = nn.Linear(dim,3)
        self.point_end_head = nn.Linear(dim,1)

    def encode(self, images):
        if images.ndim != 4 or images.shape[0] < 1 or images.shape[1] != 1 or images.shape[-2] != images.shape[-1]:
            raise ValueError('requires nonempty square images [B,1,H,H]')
        pyramid = self.encoder(images)
        sizes = [self.memory_size,max(1,self.memory_size//2)]
        # With no mask branch, every FPN level must still feed a supervised path.
        # Pool fine/coarse groups into the same two grids (80 tokens by default).
        groups = [pyramid[:-2],pyramid[-2:]]
        memories = []
        for group,size in zip(groups,sizes):
            feat = torch.stack([F.adaptive_avg_pool2d(value,(size,size)) for value in group]).mean(0)
            memories.append(feat.flatten(2).transpose(1,2)
                            + position_2d(size,size,self.dim,feat.device,feat.dtype))
        return torch.cat(memories,1)

    def next_stroke(self, memory, history):
        prefix = torch.cat([self.stroke_bos.expand(len(memory),-1,-1)] + [z[:,None] for z in history],1)
        prefix = prefix + self.stroke_position(torch.arange(len(history)+1,device=memory.device))[None]
        mask = torch.ones(prefix.shape[1],prefix.shape[1],dtype=torch.bool,device=memory.device).triu(1)
        z = self.stroke_decoder(prefix,memory,tgt_mask=mask)[:,-1]
        return z,self.condition(z),self.stroke_end_head(z)[:,0]

    def point_step(self, previous, index, condition, memory, caches):
        x = self.point_bos.expand(len(condition),-1,-1) if previous is None else self.point_embedding(previous[:,None])
        x = x + self.point_position.weight[index][None,None] + condition[:,None]
        for layer,cache in zip(self.point_layers,caches):
            x = x + attention_step(layer.self_attn,layer.norms[0](x),cache.setdefault('self',{}))
            x = x + attention_step(layer.cross_attn,layer.norms[1](x),cache.setdefault('cross',{}),memory)
            x = x + layer.ffn(layer.norms[2](x))
        hidden = self.point_norm(x[:,0])
        raw = self.geometry_head(hidden)
        geometry = torch.cat([raw[:,:2].sigmoid(),F.softplus(raw[:,2:])+1e-6],-1)
        return geometry,self.point_end_head(hidden)[:,0]

    def forward(self, images, lengths):
        """Use ONLY target lengths to organize computation, never target geometry.

        Predicted stops do not gate training. Cache compaction retains full BPTT.
        Each item contains strokes at their original lengths, without padding.
        """
        if len(lengths) != len(images) or any(not row for row in lengths):
            raise ValueError('requires one nonempty length list per image')
        if any(len(row)>self.max_strokes or any(type(n) is not int or not 1<=n<=self.max_points for n in row) for row in lengths):
            raise ValueError('target lengths exceed model caps; labels are never truncated')
        memory = self.encode(images)
        owners = list(range(len(images)))
        history = []
        results = [dict(strokes=[],stroke_end_logits=[]) for _ in owners]
        for stroke in range(max(map(len,lengths))):
            z,condition,stroke_logits = self.next_stroke(memory,history)
            history.append(z)
            active = list(range(len(owners)))
            geometry,logits = ([[] for _ in owners] for _ in range(2))
            caches = [{} for _ in self.point_layers]
            point_memory = memory
            previous = None
            for index in range(max(lengths[owner][stroke] for owner in owners)):
                g,end = self.point_step(previous,index,condition,point_memory,caches)
                for row,slot in enumerate(active):
                    geometry[slot].append(g[row]);logits[slot].append(end[row])
                keep = [row for row,slot in enumerate(active) if index+1 < lengths[owners[slot]][stroke]]
                if not keep:
                    break
                previous = g
                if len(keep) != len(active):
                    selected = torch.tensor(keep,device=images.device)
                    previous = previous.index_select(0,selected)
                    condition = condition.index_select(0,selected)
                    point_memory = point_memory.index_select(0,selected)
                    caches = _select_caches(caches,selected)
                    active = [active[row] for row in keep]
            for row,owner in enumerate(owners):
                results[owner]['strokes'].append(dict(geometry=torch.stack(geometry[row]),point_end_logits=torch.stack(logits[row])))
                results[owner]['stroke_end_logits'].append(stroke_logits[row])
            keep = [row for row,owner in enumerate(owners) if stroke+1 < len(lengths[owner])]
            if not keep:
                break
            if len(keep) != len(owners):
                selected = torch.tensor(keep,device=images.device)
                memory = memory.index_select(0,selected)
                history = [z.index_select(0,selected) for z in history]
                owners = [owners[row] for row in keep]
        for result in results:
            result['stroke_end_logits'] = torch.stack(result['stroke_end_logits'])
        return results

    @torch.inference_mode()
    def generate(self, image, point_end_threshold=.5, stroke_end_threshold=.5,
                 max_strokes=None, max_points=None):
        """Image-only generation. Finish a stroke before consulting its last score."""
        if self.training:
            raise ValueError('generate requires model.eval()')
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError('image-only generation accepts one image at a time')
        if not all(math.isfinite(v) and 0<v<1 for v in (point_end_threshold,stroke_end_threshold)):
            raise ValueError('stop thresholds must be finite and in (0,1)')
        ks = self.max_strokes if max_strokes is None else max_strokes
        ts = self.max_points if max_points is None else max_points
        if not 1<=ks<=self.max_strokes or not 1<=ts<=self.max_points:
            raise ValueError('decode caps exceed model positions')
        memory = self.encode(image)
        history,strokes,stroke_logits = [],[],[]
        completed,reason = False,'max_strokes_reached'
        for _ in range(ks):
            z,condition,last = self.next_stroke(memory,history)
            history.append(z);stroke_logits.append(last[0])
            caches = [{} for _ in self.point_layers]
            previous = None
            geometry,logits = [],[]
            ended = False
            for index in range(ts):
                g,end = self.point_step(previous,index,condition,memory,caches)
                geometry.append(g[0]);logits.append(end[0]);previous=g
                if bool(end.sigmoid()[0] >= point_end_threshold):
                    ended=True;break
            strokes.append(dict(geometry=torch.stack(geometry),point_end_logits=torch.stack(logits),point_ended=ended))
            if not ended:
                reason='max_points_reached';break
            if bool(last.sigmoid()[0] >= stroke_end_threshold):
                completed,reason=True,None;break
        return dict(strokes=strokes,stroke_end_logits=torch.stack(stroke_logits),completed=completed,
                    truncated=not completed,truncation_reason=reason,used_ground_truth=False)


def export_trajectory(result, image_size):
    """Create legacy-compatible six-dimensional events without inventing a stop."""
    geometry = torch.cat([s['geometry'] for s in result['strokes']]).detach().clone()
    geometry[:,:2] *= image_size-1
    geometry[:,2] *= image_size
    states = geometry.new_zeros(len(geometry),3);states[:,0]=1
    offsets=[0]
    for stroke in result['strokes']:
        offsets.append(offsets[-1]+len(stroke['geometry']))
        if stroke.get('point_ended',False):
            states[offsets[-1]-1]=states.new_tensor([0,1,0])
    if result['completed']:
        states[-1]=states.new_tensor([0,0,1])
    return dict(points=torch.cat([geometry,states],-1).cpu().numpy(),
                stroke_offsets=offsets,subpath_offsets=list(offsets))
