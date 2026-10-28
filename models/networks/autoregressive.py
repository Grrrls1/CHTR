"""Image-only greedy decoding using the existing model weights and exact attention caches."""
import torch
from torch.nn import functional as F


def encode_image(model, images):
    pyramid=model.encoder(images)
    pixels=pyramid[0]
    sizes=[model.memory_size,max(1,model.memory_size//2)]
    from .model import position_2d
    memory=torch.cat([F.adaptive_avg_pool2d(feat,(size,size)).flatten(2).transpose(1,2)+
                      position_2d(size,size,model.dim,feat.device,feat.dtype)
                      for feat,size in zip(pyramid[-2:],sizes)],1)
    return pixels,memory,sizes


def next_stroke(model,pixels,memory,sizes,history,image_size):
    prefix=torch.cat([model.stroke_bos.expand(memory.shape[0],-1,-1)]+[z[:,None] for z in history],1)
    prefix=prefix+model.stroke_position(torch.arange(len(history)+1,device=memory.device))[None]
    causal=torch.ones(prefix.shape[1],prefix.shape[1],device=memory.device,dtype=torch.bool).triu(1)
    z=model.stroke_decoder(prefix,memory,tgt_mask=causal)[:,-1]
    mask=torch.einsum('bd,bdhw->bhw',model.mask_query(z),pixels)/model.dim**.5
    mask=F.interpolate(mask[:,None],size=(image_size,image_size),mode='bilinear',align_corners=False)[:,0]
    types=model.type_head(z)
    cond=model.condition(torch.cat([z,types.softmax(-1)@model.type_embedding],-1))
    bias=torch.cat([model.mask_alpha*F.interpolate(mask.sigmoid()[:,None],size=(size,size),
            mode='bilinear',align_corners=False).flatten(1).clamp_min(1e-4).log() for size in sizes],1)
    return z,mask,types,cond,bias


def split_heads(x,heads):
    return x.reshape(x.shape[0],x.shape[1],heads,-1).transpose(1,2)


def attention_step(attn,q,cache,memory=None,bias=None):
    dim=attn.embed_dim;weight=attn.in_proj_weight;offset=attn.in_proj_bias
    def project(x,part):
        return split_heads(F.linear(x,weight[part*dim:(part+1)*dim],
                   offset[part*dim:(part+1)*dim] if offset is not None else None),attn.num_heads)
    query=project(q,0)
    if memory is None:
        key,value=project(q,1),project(q,2)
        if 'key' in cache:
            key=torch.cat([cache['key'],key],2);value=torch.cat([cache['value'],value],2)
        cache.update(key=key,value=value)
    else:
        if 'key' not in cache:cache.update(key=project(memory,1),value=project(memory,2))
        key,value=cache['key'],cache['value']
    scores=bias[:,None,None,:] if bias is not None else None
    if hasattr(F,'scaled_dot_product_attention'):
        out=F.scaled_dot_product_attention(query,key,value,attn_mask=scores,dropout_p=0.)
    else:  # Local torch 1.13; identical scaled attention without the torch 2 API.
        logits=(query@key.transpose(-2,-1))/query.shape[-1]**.5
        if scores is not None:logits=logits+scores
        out=logits.softmax(-1)@value
    return attn.out_proj(out.transpose(1,2).reshape(q.shape[0],1,dim))


def point_step(model,previous,index,cond,memory,bias,caches):
    if index==0:
        x=model.point_bos.expand(cond.shape[0],-1,-1)
    else:
        x=model.point_embedding(previous[:,None])
    x=x+model.point_position.weight[index][None,None]+cond[:,None]
    for layer,cache in zip(model.point_layers,caches):
        x=x+attention_step(layer.self_attn,layer.norms[0](x),cache.setdefault('self',{}))
        x=x+attention_step(layer.cross_attn,layer.norms[1](x),cache.setdefault('cross',{}),memory,bias)
        x=x+layer.ffn(layer.norms[2](x))
    hidden=model.point_norm(x[:,0]);raw=model.geometry_head(hidden)
    geometry=torch.cat([raw[:,:2].sigmoid(),F.softplus(raw[:,2:])+1e-6],-1)
    return geometry,model.state_head(hidden)


@torch.inference_mode()
def generate(model,image,max_strokes=None,max_points=None):
    if model.training:raise ValueError('Image-only decoding requires model.eval()')
    if image.shape[0]!=1:raise ValueError('Greedy decoder currently accepts one image at a time')
    max_strokes=model.max_strokes if max_strokes is None else max_strokes
    max_points=model.max_points if max_points is None else max_points
    if not 1<=max_strokes<=model.max_strokes or not 1<=max_points<=model.max_points:
        raise ValueError('Decode limits must fit the learned position embeddings')
    size=image.shape[-1];pixels,memory,sizes=encode_image(model,image)
    history=[];points=[];offsets=[0];types=[];masks=[];type_probs=[]
    completed=False;reason=None
    for _ in range(max_strokes):
        z,mask,type_logits,cond,bias=next_stroke(model,pixels,memory,sizes,history,size)
        history.append(z);types.append(int(type_logits.argmax(-1).item()));masks.append(mask.sigmoid()[0])
        type_probs.append(type_logits.softmax(-1)[0])
        caches=[{} for _ in model.point_layers];previous=None;event=0
        for index in range(max_points):
            geometry,state=point_step(model,previous,index,cond,memory,bias,caches)
            event=int(state.argmax(-1).item())
            previous=torch.cat([geometry,F.one_hot(torch.tensor([event],device=image.device),3).to(geometry.dtype)],-1)
            points.append(previous[0])
            if event!=0:break
        offsets.append(len(points))
        if event==2:
            completed=True;break
        if event==0:
            reason='max_points_reached';break
    if not completed and reason is None:reason='max_strokes_reached'
    result=torch.stack(points).clone();result[:,:2]*=size-1;result[:,2]*=size
    return dict(points=result.cpu().numpy(),stroke_offsets=offsets,subpath_offsets=list(offsets),
                stroke_types=types,stroke_masks=torch.stack(masks).cpu().numpy(),
                stroke_type_probs=torch.stack(type_probs).cpu().numpy(),
                completed=completed,truncated=not completed,truncation_reason=reason,
                image_size=size,mode='image_only_greedy',used_ground_truth=False)
