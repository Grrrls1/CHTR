"""Stable 24-class colors shared by GT, teacher-forced and image-only masks."""
import html
import json
from pathlib import Path

import cv2
import numpy as np

# RGB, converted explicitly for OpenCV BGR image files.
TYPE_COLORS=((31,119,180),(214,39,40),(44,160,44),(255,127,14),(148,103,189),
             (23,190,207),(227,119,194),(140,86,75),(188,189,34),(127,127,127),
             (174,199,232),(255,152,150),(152,223,138),(255,187,120),(197,176,213),
             (158,218,229),(247,182,210),(196,156,148),(219,219,141),(199,199,199),
             (0,80,70),(100,40,120),(170,70,0),(0,90,160))


def color_masks(masks,types,image,threshold=.5,annotate=True):
    masks=np.asarray(masks,dtype=np.float32);types=np.asarray(types,dtype=np.int64)
    image=np.asarray(image,dtype=np.uint8)
    if masks.ndim!=3 or masks.shape[0]!=len(types) or masks.shape[1:]!=image.shape:
        raise ValueError('Mask/type/image dimensions disagree')
    if not np.isfinite(masks).all() or np.any((masks<0)|(masks>1)):
        raise ValueError('Expected mask probabilities in [0,1]')
    if np.any((types<0)|(types>=len(TYPE_COLORS))):raise ValueError('Unknown stroke type')
    active=masks>=threshold
    colors=np.asarray(TYPE_COLORS,dtype=np.float32)[types,::-1]
    count=active.sum(0)
    total=np.einsum('khw,kc->hwc',active.astype(np.float32),colors)
    blend=total/np.maximum(count[...,None],1)
    background=np.repeat((.25*image+.75*255)[...,None],3,axis=-1)
    result=background.copy();visible=count>0
    result[visible]=.85*blend[visible]+.15*background[visible]
    result=np.rint(result).astype(np.uint8)
    if annotate:
        for index,mask in enumerate(active):
            yy,xx=np.where(mask)
            if len(xx)==0:continue
            closest=int(np.argmin((xx-xx.mean())**2+(yy-yy.mean())**2))
            point=(max(0,int(xx[closest])-3),min(image.shape[0]-3,int(yy[closest])+3))
            cv2.putText(result,str(index+1),point,cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),3,cv2.LINE_AA)
            cv2.putText(result,str(index+1),point,cv2.FONT_HERSHEY_SIMPLEX,.42,(0,0,0),1,cv2.LINE_AA)
    return result


def type_entries(types,masks,names,probabilities=None):
    entries=[]
    for index,label in enumerate(types):
        label=int(label)
        item=dict(stroke_number=index+1,type_id=label,type_name=names[label],
                  rgb=list(TYPE_COLORS[label]),mask_pixels=int((masks[index]>=.5).sum()))
        if probabilities is not None:item['type_confidence']=float(probabilities[index,label])
        entries.append(item)
    return entries


def save_masks(directory,image,names,label_masks,label_types,teacher_masks,teacher_probs,free_masks,free_probs):
    teacher_types=np.asarray(teacher_probs).argmax(-1);free_types=np.asarray(free_probs).argmax(-1)
    sets=(('label_masks',label_masks,label_types,None),('teacher_masks',teacher_masks,teacher_types,teacher_probs),
          ('free_masks',free_masks,free_types,free_probs))
    details={}
    for name,masks,types,probs in sets:
        cv2.imwrite(str(Path(directory)/(name+'.png')),color_masks(masks,types,image))
        details[name]=type_entries(types,masks,names,probs)
    (Path(directory)/'mask_types.json').write_text(json.dumps(details,ensure_ascii=False,indent=2)+'\n')
    return details


def write_comparison_sheet(folder,results):
    if not results:return
    chosen=sorted(set((0,(len(results)-1)//2,len(results)-1)))
    views=(('input','Input'),('label','Label trajectory'),('label_masks','GT masks / GT types'),
           ('teacher_forced','Teacher trajectory'),('teacher_masks','Teacher predicted masks'),
           ('free','Image-only trajectory'),('free_masks','Free predicted masks'))
    rows=[]
    for index in chosen:
        item=results[index];directory=folder/item['sample_id'];tiles=[]
        if not all((directory/(name+'.png')).exists() for name,_ in views):return
        for name,title in views:
            image=cv2.imread(str(directory/(name+'.png')))
            tile=cv2.copyMakeBorder(image,32,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
            cv2.putText(tile,title,(6,22),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,0,0),1,cv2.LINE_AA)
            tiles.append(tile)
        row=np.hstack(tiles);caption=np.full((28,row.shape[1],3),255,np.uint8)
        text='GT strokes=%d, free strokes=%d; teacher IoU=%.3f, free IoU=%.3f'%(item['gt_strokes'],item['predicted_strokes'],item['teacher_forced_iou'],item['image_only_iou'])
        cv2.putText(caption,text,(8,19),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1,cv2.LINE_AA)
        rows.append(np.vstack([row,caption]))
    width=rows[0].shape[1];legend=np.full((100,width,3),255,np.uint8)
    cv2.putText(legend,'Mask colors = TYPE IDs (Chinese names in HTML legend); numbers on masks = stroke order. Overlaps blend colors.',(8,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1,cv2.LINE_AA)
    for i,color in enumerate(TYPE_COLORS):
        x=(i%8)*(width//8)+8;y=27+(i//8)*24
        cv2.rectangle(legend,(x,y),(x+16,y+16),tuple(reversed(color)),-1)
        cv2.putText(legend,'Type ID %d'%i,(x+22,y+13),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1,cv2.LINE_AA)
    sheet=np.vstack(rows+[legend])
    cv2.imwrite(str(folder/'comparison_sheet_with_masks.png'),sheet)
    cv2.imwrite(str(folder/'comparison_sheet.png'),sheet)


def write_gallery(folder,results,names):
    folder=Path(folder)
    legend=[dict(type_id=i,type_name=names[i],rgb=list(TYPE_COLORS[i])) for i in range(24)]
    (folder/'mask_type_legend.json').write_text(json.dumps(legend,ensure_ascii=False,indent=2)+'\n')
    swatches=''.join('<span class="swatch"><i style="background:rgb(%d,%d,%d)"></i>%d %s</span>'%
                     (*TYPE_COLORS[i],i,html.escape(names[i])) for i in range(24))
    cards=[]
    views=(('input','Input'),('label','Trajectory label'),('label_masks','GT masks / GT types'),
           ('teacher_forced','Teacher-forced trajectory / GT boundaries'),
           ('teacher_masks','Predicted masks / predicted types / GT stroke slots'),
           ('free','Image-only trajectory'),('free_masks','Image-only predicted masks / predicted types'),
           ('overlay','Red=input, blue=predicted trajectory'))
    for row in results:
        sid=row['sample_id'];figures=[]
        for name,title in views:
            if (folder/sid/(name+'.png')).exists():
                figures.append('<figure><img width="256" src="%s/%s.png"><figcaption>%s</figcaption></figure>'%(sid,name,title))
        table='';f=folder/sid/'mask_types.json'
        if f.exists():
            info=json.loads(f.read_text())
            for key,title in (('label_masks','GT'),('teacher_masks','Teacher predicted'),('free_masks','Free predicted')):
                labels='; '.join('%d: %s [ID %d]%s'%(s['stroke_number'],s['type_name'],s['type_id'],
                    (' %.1f%%'%(100*s['type_confidence'])) if 'type_confidence' in s else '') for s in info[key])
                table+='<p><b>'+title+': </b>'+html.escape(labels)+'</p>'
        cards.append('<article><h2>'+html.escape(row['character']+' '+sid)+'</h2><div class="views">'+''.join(figures)+
                     '</div>'+table+'<details><summary>Metrics</summary><pre>'+html.escape(json.dumps(row,ensure_ascii=False,indent=2))+'</pre></details></article>')
    document='<meta charset="utf-8"><style>body{font-family:sans-serif;margin:20px}figure{display:inline-block;margin:6px;width:256px;vertical-align:top}article{border-bottom:1px solid #aaa;padding:12px 0}.swatch{display:inline-flex;align-items:center;margin:5px}.swatch i{width:18px;height:18px;display:inline-block;margin-right:5px;border:1px solid #777}figcaption{font-size:13px}</style><h1>Training-image prediction comparison</h1><p>Mask colors follow stroke TYPE IDs, not stroke-instance IDs. Numbers are 1-based stroke order. Same type has the same color; overlaps blend colors. Mask threshold: 0.5; input is a faint gray background.</p><p>Teacher view uses GT stroke slots; free view uses the original generated stroke count. Both predicted-mask views use PREDICTED types. Teacher trajectories use GT history/boundaries. Free trajectories use no GT history. Truncated output is not complete.</p><div>'+swatches+'</div>'+''.join(cards)
    temporary=folder/'index.mask_types.tmp.html';temporary.write_text(document,encoding='utf-8');temporary.replace(folder/'index.html')
    write_comparison_sheet(folder,results)
