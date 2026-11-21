"""Pure CPU protocol, matching, noise, scoring and response primitives."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from models.training.single_stroke_evaluation import raster, iou
from models.training.casia_benchmark_metrics import trajectory_metrics
from models.training.two_stage_metrics import ALIGNED_KEYS
from models.scripts.evaluate_cross_domain_sampled import atomic_bytes, write_json, read_json, write_csv, digest

SCALE=np.array([255,255,256],np.float32)
LEVELS=(.5,1.,2.)
POSITIONS=np.array([0,.1,.25,.5,.75,1.])


def paths_from(record):
    points=np.asarray(record['prediction']['points_xyw_px'],np.float32).reshape(-1,3)
    off=np.asarray(record['prediction']['stroke_offsets'],np.int64)
    if off[0]!=0 or off[-1]!=len(points) or np.any(np.diff(off)<1):raise ValueError('invalid committed offsets')
    return [points[a:b] for a,b in zip(off[:-1],off[1:])]


def pack_prediction(paths,states):
    return dict(points_xyw_px=np.concatenate(paths).tolist() if paths else [],
                stroke_offsets=np.r_[0,np.cumsum([len(p) for p in paths],dtype=np.int64)].tolist(),point_states=states)


def masks_union(masks):
    return np.logical_or.reduce(masks) if len(masks) else np.zeros((256,256),bool)


def geometry(gt,pred,exclude_first=False):
    if exclude_first:gt,pred=gt[1:],pred[1:]
    if not len(gt) or not len(pred):return {k:None for k in ALIGNED_KEYS}
    t=np.linspace(0,1,max(len(gt),len(pred),2))
    rs=lambda a:np.stack([np.interp(t,np.linspace(0,1,len(a)),a[:,j]) for j in range(3)],-1)
    err=rs(pred)-rs(gt)
    return dict(aligned_xy_mae_px=float(np.abs(err[:,:2]).mean()),aligned_width_mae_px=float(np.abs(err[:,2]).mean()),
                aligned_width_bias_px=float(err[:,2].mean()),aligned_endpoint_l2_px=float(np.linalg.norm(pred[-1,:2]-gt[-1,:2])),
                aligned_point_count_mae=abs(len(pred)-len(gt)),aligned_exact_point_count_rate=float(len(pred)==len(gt)))


def score_character(image,gt,gt_masks,paths,states,inst):
    gm=[raster(p) for p in gt];pm=[raster(p) for p in paths]
    n,m=len(gt),len(paths);common=min(n,m)
    metrics=dict(character_trajectory_iou=iou(masks_union(gm),masks_union(pm)),
        ordered_trajectory_iou=sum(iou(a,b) for a,b in zip(gm,pm))/max(n,m,1),
        aligned_strokes=common,predicted_strokes=m,true_strokes=n,missing_strokes=max(n-m,0),extra_strokes=max(m-n,0),
        terminated=sum(int(s['terminated']) for s in states),truncated=sum(int(s['truncated']) for s in states),
        stroke_count_mae=abs(n-m),count_exact=float(n==m),empty_output=int(m==0),unpaired_strokes=abs(n-m),
        character_truncated=int(inst['truncated']),character_terminated=int(inst['terminated']),
        empty_masks=sum(int(not x.any()) for x in inst['masks']),gt_exceeds_stroke_cap=int(n>64),
        gt_strokes_exceeding_point_cap=sum(int(len(p)>400) for p in gt))
    aligned=[geometry(a,b) for a,b in zip(gt,paths)]
    for key in ALIGNED_KEYS:metrics[key]=float(np.mean([a[key] for a in aligned])) if aligned else None
    values=[iou(a,b) for a,b in zip(gt_masks,inst['masks'])]
    metrics.update(ordered_mask_iou=sum(values)/max(n,m,1),strict_sequence_mask50=float(n==m and all(v>=.5 for v in values)))
    metrics.update(trajectory_metrics(image,gt,paths))
    fg=cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]>0
    metrics['image_char_iou']=iou(fg,masks_union(pm))
    return metrics


def matching(pred_masks,gt_masks,threshold=.5):
    """Maximum valid cardinality, then maximum IoU; canonical ordered SciPy tie resolution."""
    m,n=len(pred_masks),len(gt_masks)
    matrix=np.zeros((m,n),np.float64)
    for p in range(m):
        for g in range(n):matrix[p,g]=iou(pred_masks[p],gt_masks[g])
    if not m or not n:return [],matrix
    valid=matrix>=threshold
    reward=valid.astype(float)+np.where(valid,matrix,0)/(max(m,n)+1)
    ps,gs=linear_sum_assignment(-reward)
    pairs=[dict(pred_index=int(p),gt_index=int(g),mask_iou=float(matrix[p,g])) for p,g in zip(ps,gs) if valid[p,g]]
    return sorted(pairs,key=lambda x:x['pred_index']),matrix


def calibrate(residuals):
    a=np.asarray(residuals,np.float64).reshape(-1,3)
    if not len(a):raise ValueError('no geometric matches for val noise calibration')
    centered=a-a.mean(0)
    sx=float(np.sqrt(np.mean(centered[:,:2]**2)));sw=float(np.sqrt(np.mean(centered[:,2]**2)))
    fallback={}
    if sx<1e-6:fallback['xy']='val variance near zero; explicit diagnostic sigma=1px';sx=1.
    if sw<1e-6:fallback['w']='val variance near zero; explicit diagnostic sigma=1px';sw=1.
    corr=np.zeros((3,3),float)
    std=a.std(0)
    for i in range(3):
        for j in range(3):corr[i,j]=float(np.mean(centered[:,i]*centered[:,j])/(std[i]*std[j])) if std[i]*std[j]>0 else 0.
    return dict(matched_instances=len(a),residual_mean_px=a.mean(0).tolist(),residual_std_px=std.tolist(),
        quantiles={str(q):np.quantile(a,q,axis=0).tolist() for q in [.01,.05,.5,.95,.99]},xyw_correlation=corr.tolist(),
        sigma_xy_base_px=sx,sigma_w_base_px=sw,mu_xy_base_px2=sx*sx,mu_w_base_px2=sw*sw,fallback=fallback,
        noise_mean=0.,variance_ddof=0,source='S_TRANSFER; fixed BOOK val100 geometric matches')


def standard_noise(seed,sid,pred_index,draw):
    vals=[]
    for component in ['x','y','w']:
        token=json.dumps([seed,sid,int(pred_index),int(draw),component],separators=(',',':')).encode()
        derived=int.from_bytes(hashlib.sha256(token).digest()[:16],'little')
        vals.append(float(np.random.default_rng(derived).standard_normal()))
    return np.array(vals,np.float64)


def first_point(gt,pred,mode,k,draw,cal,seed,sid,pid):
    gt=np.asarray(gt,np.float64);pred=np.asarray(pred,np.float64)
    z=standard_noise(seed,sid,pid,draw)
    requested=np.zeros(3,float)
    if mode=='PRED':actual=pred.copy();requested=pred-gt
    else:
        if mode in ['N10','N11']:requested[:2]=z[:2]*k*cal['sigma_xy_base_px']
        if mode in ['N01','N11']:requested[2]=z[2]*k*cal['sigma_w_base_px']
        actual=gt+requested
        actual[:2]=np.clip(actual[:2],0,255);actual[2]=max(actual[2],1e-3)
    return actual.astype(np.float32),dict(z=z.tolist(),requested_noise_px=requested.tolist(),actual_noise_px=(actual-gt).tolist(),
        xy_clipped=bool(np.any(actual[:2]!=gt[:2]+requested[:2])),w_floored=bool(actual[2]!=gt[2]+requested[2]))


def arc64(path):
    a=np.asarray(path,np.float64)
    if not len(a):raise ValueError('empty stroke is not a valid generated first-point path')
    distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(a[:,:2],axis=0),axis=1))]
    length=float(distance[-1])
    if length<=1e-12:return np.repeat(a[:1],64,axis=0),length,True
    # Zero-length segments have no geometric extent; choose their final sample explicitly.
    keep=np.r_[np.diff(distance)>0,True];s=distance[keep]/length
    out=np.stack([np.interp(np.linspace(0,1,64),s,a[keep,j]) for j in range(3)],-1)
    out[0]=a[0];out[-1]=a[-1]
    return out,length,False


def response(gt,clean,pred,delta):
    g,gl,gd=arc64(gt);c,cl,cd=arc64(clean);p,pl,pd=arc64(pred)
    at=lambda a:np.stack([np.interp(POSITIONS,np.linspace(0,1,64),a[:,j]) for j in range(3)],-1)
    ga,ca,pa=map(at,[g,c,p]);delta=np.asarray(delta,float)
    return dict(s=POSITIONS.tolist(),xy_clean_sq=np.sum((pa[:,:2]-ca[:,:2])**2,-1).tolist(),
        w_clean_sq=((pa[:,2]-ca[:,2])**2).tolist(),xy_gt_sq=np.sum((pa[:,:2]-ga[:,:2])**2,-1).tolist(),
        w_gt_sq=((pa[:,2]-ga[:,2])**2).tolist(),clean_xy_gt_sq=np.sum((ca[:,:2]-ga[:,:2])**2,-1).tolist(),
        clean_w_gt_sq=((ca[:,2]-ga[:,2])**2).tolist(),actual_delta_xy_sq=float(np.sum(delta[:2]**2)),
        actual_delta_w_sq=float(delta[2]**2),gt_length_px=gl,clean_length_px=cl,pred_length_px=pl,
        degenerate_gt=gd,degenerate_clean=cd,degenerate_pred=pd)


def instance_metrics(gt,pred,clean,delta,state,clean_state=None):
    clean_state=state if clean_state is None else clean_state
    result=dict(full=geometry(gt,pred),post_start=geometry(gt,pred,True),
        trajectory_iou=iou(raster(gt),raster(pred)),predicted_points=len(pred),gt_points=len(gt),clean_points=len(clean),
        clean_point_truncated=int(clean_state['truncated']),clean_point_terminated=int(clean_state['terminated']),
        post_start_valid=int(len(gt)>1 and len(pred)>1),predicted_first_only=int(len(pred)==1),gt_first_only=int(len(gt)==1),
        point_truncated=int(state['truncated']),point_terminated=int(state['terminated']),
        first_xy_l2_px=float(np.linalg.norm(pred[0,:2]-gt[0,:2])),first_w_error_px=float(pred[0,2]-gt[0,2]),
        arc=response(gt,clean,pred,delta))
    return result


def mean_defined(values):
    a=[x for x in values if x is not None]
    return float(np.mean(a)) if a else None


def single_character_metrics(instances):
    out={'matched_instances':len(instances),
         'post_start_any_valid_instances':sum(r['post_start_valid']>0 for r in instances),
         'post_start_valid_draw_weight':sum(r['post_start_valid'] for r in instances),
         'predicted_first_only_draw_weight':sum(r['predicted_first_only'] for r in instances)}
    for key in ['trajectory_iou','first_xy_l2_px','first_w_error_px','predicted_points','gt_points','clean_points','clean_point_truncated','clean_point_terminated','post_start_valid','predicted_first_only','gt_first_only','point_truncated','point_terminated']:
        out[key]=mean_defined([r[key] for r in instances])
    for scope in ['full','post_start']:
        for key in ALIGNED_KEYS:out[scope+'_'+key]=mean_defined([r[scope][key] for r in instances])
    for key in ['gt_length_px','clean_length_px','pred_length_px','actual_delta_xy_sq','actual_delta_w_sq']:
        out[key]=mean_defined([r['arc'][key] for r in instances])
    for key in ['degenerate_gt','degenerate_clean','degenerate_pred']:
        out[key+'_count']=sum(int(r['arc'][key]) for r in instances)
    for key in ['xy_clean_sq','w_clean_sq','xy_gt_sq','w_gt_sq','clean_xy_gt_sq','clean_w_gt_sq']:
        out[key]=np.mean([r['arc'][key] for r in instances],axis=0).tolist() if instances else [None]*len(POSITIONS)
    return out


def conditions():
    out=[]
    for point in ['BOOK','TRANSFER']:
        for mask in ['GT','PRED']:
            for mode in ['N00','PRED']:
                out.append(dict(experiment='E1',domain='BOOK',point=point,mask=mask,mode=mode,k=0.,draws=1,
                    condition_id=f'E1_{point}_{mask}_{mode}',oracle=not(mask=='PRED' and mode=='PRED')))
        out.append(dict(experiment='GT_REF',domain='BOOK',point=point,mask='GT',mode='N00',k=0.,draws=1,
                        condition_id=f'GT_REF_{point}',oracle=True))
    for domain in ['BOOK','CASIA']:
        for stroke in ['BOOK','TRANSFER']:
            for point in ['BOOK','TRANSFER']:
                out.append(dict(experiment='E2',domain=domain,stroke=stroke,point=point,mask='PRED',mode='PRED',k=0.,draws=1,
                                condition_id=f'E2_{domain}_{stroke}_{point}',oracle=False))
    for point in ['BOOK','TRANSFER']:
        for mask in ['GT','PRED']:
            for mode in ['N10','N01','N11']:
                for k in LEVELS:
                    out.append(dict(experiment='E1',domain='BOOK',point=point,mask=mask,mode=mode,k=k,draws=5,
                                    condition_id=f'E1_{point}_{mask}_{mode}_k{k:g}',oracle=True))
    assert len(out)==54 and sum(r['draws'] for r in out)==198
    return out
