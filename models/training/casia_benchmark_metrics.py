"""Local 256px realization of PEN-Net section 3; not claimed official-code parity.

AIoU: Otsu input mask vs one-pixel prediction, maximized over repeated square
3x3 dilation. Exhaust all distinct masks through canvas saturation, rather than
stop at the first plateau. LDTW: min-total Euclidean DTW cost / its path length.
"""
import cv2
import numpy as np
from models.training.casia_metrics import dtw_xy

PROTOCOL = dict(name='pen_net_equations_local256_v1', source='https://arxiv.org/abs/2210.15879',
                official_implementation_parity=False, ground_truth='Otsu threshold of input image',
                centerline='rounded XY; cv2 LINE_8 width=1; disconnected strokes; singleton pixel',
                dilation='3x3 square, zero border; max over iterations 0..canvas saturation',
                aggregate='arithmetic mean across characters; point diagnostic mean across strokes',
                empty_prediction_aiou=0., empty_prediction_ldtw=None,
                dtw='Euclidean XY; min-total-cost path; tie diagonal,vertical,horizontal; no resampling',
                ldtw='DTW cost divided by selected optimal path length, canvas pixel units')


def centerline_mask(paths, shape=(256,256)):
    mask=np.zeros(shape,np.uint8)
    for path in paths:
        xy=np.asarray(path)[:, :2]
        if not len(xy):
            continue
        if not np.isfinite(xy).all():
            raise ValueError('nonfinite prediction')
        xy=np.rint(xy).astype(np.int32)
        cv2.polylines(mask,[xy],False,1,1,cv2.LINE_8)
        for x,y in (xy[0],xy[-1]):
            if 0<=x<shape[1] and 0<=y<shape[0]:
                mask[y,x]=1
    return mask.astype(bool)


def adaptive_iou(image, paths):
    if image.dtype != np.uint8 or image.ndim != 2:
        raise ValueError('AIoU requires uint8 grayscale input')
    _, binary = cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    gt=binary.astype(bool)
    pred=centerline_mask(paths,image.shape)
    if not pred.any() or not gt.any():
        return dict(aiou=0., dilation_iterations=0)
    # Chessboard distance <= k equals k applications of the square 3x3 kernel.
    distance=cv2.distanceTransform((~pred).astype(np.uint8),cv2.DIST_C,3).astype(np.int32)
    all_counts=np.bincount(distance.ravel())
    gt_counts=np.bincount(distance[gt],minlength=len(all_counts))
    intersection=np.cumsum(gt_counts,dtype=np.int64)
    union=int(gt.sum())+np.cumsum(all_counts,dtype=np.int64)-intersection
    scores=intersection/np.maximum(union,1)
    best=int(np.argmax(scores))
    return dict(aiou=float(scores[best]),dilation_iterations=best)


def trajectory_metrics(image, reference, prediction):
    result=adaptive_iou(image,prediction)
    gt=np.concatenate(reference)[:,:2]
    pred=np.concatenate(prediction)[:,:2] if prediction else np.empty((0,2))
    result.update(dtw_xy(gt,pred))
    result['ldtw_px']=result['dtw_path_mean_px']
    return result
