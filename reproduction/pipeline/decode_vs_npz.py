"""Decode original VS state, retaining unclipped Bézier endpoints and subpaths."""
import argparse
from pathlib import Path
import numpy as np
import cv2
from common import arrays, read_json, require, states, validate_points, write_json, digest, SCHEMA


def replay_native(source, config):
    """Independent NumPy replay of draw + TF1 crop_and_resize (including zero extrapolation).

    This verifies state/axis/width handling separately from the continuous point renderer.
    It is not used to pass the six-dimensional point-rendering fidelity gate.
    """
    z=arrays(source); data=z['strokes_data']; size=int(z['image_size']); raster=int(config['raster_size'])
    canvas=np.zeros((size,size),np.float32); index=0
    for cursor,length in zip(np.asarray(z['init_cursors']).reshape(-1,2),np.atleast_1d(z['round_length'])):
        cursor=cursor.astype(np.float64)*size; width=float(z['init_width']); scale=1.; previous=float(raster)
        for _ in range(int(length)):
            flag,c0,c1,d0,d1,radius,scaling=data[index]; index+=1
            window=float(np.clip(scale*previous,config['min_window_size'],size))
            a=np.array([.5,.5]); b=(np.array([d0,d1])+1)/2; c=a+(b-a)*[c0,c1]
            coords=[np.trunc(v*(2*raster-1)+.5).astype(int) for v in (a,c,b)]
            r0=int(1+width*raster//2); r1=int(1+radius*raster//2)
            local=np.zeros((raster*2,raster*2),np.float32)
            for i in range(100):
                t=i/100.; xy=np.trunc((1-t)**2*coords[0]+2*t*(1-t)*coords[1]+t*t*coords[2]).astype(int)
                cv2.circle(local,(int(xy[1]),int(xy[0])),int((1-t)*r0+t*r1),1.,-1)
            patch=cv2.resize(local,(raster,raster))
            lo=np.floor(cursor-window/2).astype(int); hi=np.ceil(cursor+window/2).astype(int); shape=hi-lo
            center=((lo+hi)/2-(cursor-window/2))/window*raster
            extent=raster*shape/window
            x=np.linspace(center[0]-(extent[0]-1)/2,center[0]+(extent[0]-1)/2,shape[0])
            y=np.linspace(center[1]-(extent[1]-1)/2,center[1]+(extent[1]-1)/2,shape[1])
            xx,yy=np.meshgrid(x,y); good=(xx>=0)&(xx<=raster-1)&(yy>=0)&(yy<=raster-1)
            xx=np.clip(xx,0,raster-1); yy=np.clip(yy,0,raster-1)
            x0=np.floor(xx).astype(int); y0=np.floor(yy).astype(int); x1=np.minimum(x0+1,raster-1); y1=np.minimum(y0+1,raster-1)
            dx=xx-x0; dy=yy-y0
            pasted=((1-dy)*((1-dx)*patch[y0,x0]+dx*patch[y0,x1])+dy*((1-dx)*patch[y1,x0]+dx*patch[y1,x1]))*good
            left,top=np.maximum(lo,0); right,bottom=np.minimum(hi,size)
            if flag==0 and right>left and bottom>top:
                canvas[top:bottom,left:right]+=pasted[top-lo[1]:bottom-lo[1],left-lo[0]:right-lo[0]]
            end=cursor+np.array([d1,d0])*window/2
            next_window=np.clip(scaling*window,config['min_window_size'],size)
            width=radius*window/next_window; scale=scaling; previous=window; cursor=np.clip(end,0,size-1)
    return np.round(np.clip(canvas,0,1)*255).astype(np.uint8)


def decode(source, config, spacing=0.5):
    z = arrays(source)
    data = z['strokes_data']
    lengths = np.atleast_1d(z['round_length'])
    cursors = np.asarray(z['init_cursors']).reshape(-1, 2)
    size = int(z['image_size'])
    raster = int(config['raster_size'])
    minimum = float(config['min_window_size'])
    require(config.get('cursor_type') == 'next', 'Only cursor_type=next is implemented')
    require(not config.get('init_cursor_on_undrawn_pixel', False), 'This decoder expects width reset at every round')
    require(size == 256 and raster == 128 and 0 < minimum <= size, 'Expected 256 canvas, original 128 renderer')
    require(0 < spacing <= 4, 'spacing must be in (0,4]')
    require(data.ndim == 2 and data.shape[1] == 7 and np.isfinite(data).all(), 'Bad strokes_data')
    require(np.issubdtype(lengths.dtype, np.integer) and np.all(lengths > 0) and lengths.sum() == len(data), 'Bad round lengths')
    require(len(cursors) == len(lengths) and np.isfinite(cursors).all() and np.all((cursors >= 0) & (cursors < 1)), 'Bad cursors')
    require(np.isin(data[:, 0], [0, 1]).all() and np.all(data[:, 5:] > 0), 'Bad flags, width, or scaling')
    require(np.all((data[:, 1:3] >= 0) & (data[:, 1:3] <= 1)) and np.all(np.abs(data[:, 3:5]) <= 1), 'Bad control or displacement')
    require(np.isfinite(z['init_width']).all() and float(z['init_width']) > 0, 'Bad initial width')
    paths, curves, current = [], [], []
    row = 0
    for cursor, length in zip(cursors, lengths):
        if current:
            paths.append(np.concatenate(current)); current = []
        cursor = cursor.astype(np.float64) * size
        width = float(z['init_width'])
        scale, prev_window = 1.0, float(raster)
        for _ in range(int(length)):
            flag, c0, c1, d0, d1, radius, scaling = data[row]; row += 1
            window = float(np.clip(scale * prev_window, minimum, size))
            end = cursor + np.array([d1, d0]) * window / 2
            control = cursor + (end - cursor) * [c1, c0]
            # utils.draw first quantizes radius on a 2*raster canvas.
            # Its diameter after downsampling/pasting is radius_integer * W/raster.
            w0 = int(1 + width * raster // 2) * window / raster
            w1 = int(1 + radius * raster // 2) * window / raster
            if flag == 0:
                if current and not np.allclose(current[-1][-1, :2], cursor, atol=1e-5):
                    paths.append(np.concatenate(current)); current = []
                length_est = np.linalg.norm(control - cursor) + np.linalg.norm(end - control)
                t = np.linspace(0, 1, max(2, int(np.ceil(length_est / spacing)) + 1))[:, None]
                xy = (1-t)**2 * cursor + 2*t*(1-t)*control + t**2*end
                xyz = np.c_[xy, ((1-t)*w0+t*w1)]
                current.append(xyz)
                curves.append(np.r_[cursor, control, end, w0, w1, window])
            elif current:
                paths.append(np.concatenate(current)); current = []
            next_window = np.clip(scaling * window, minimum, size)
            width = radius * window / next_window
            scale, prev_window = scaling, window
            cursor = np.clip(end, 0, size - 1)
    if current:
        paths.append(np.concatenate(current))
    require(paths, 'No drawing segments')
    offsets = np.r_[0, np.cumsum([len(p) for p in paths])].astype(np.int64)
    points = states(np.c_[np.concatenate(paths), np.zeros((int(offsets[-1]), 3))], offsets)
    validate_points(points, offsets, bounds=False)
    return dict(points=points, subpath_offsets=offsets, curves=np.asarray(curves), image_size=np.array(size), schema=np.array(SCHEMA))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--file', required=True)
    p.add_argument('--model-config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--spacing', type=float, default=0.5)
    a = p.parse_args()
    require(not Path(a.output).exists(), 'Output exists')
    z = decode(a.file, read_json(a.model_config), a.spacing)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output, **z)
    write_json(a.output + '.json', dict(schema=SCHEMA, source=str(Path(a.file).resolve()), source_sha256=digest(a.file), model_config=read_json(a.model_config), spacing=a.spacing, width_unit='pixel diameter', state_semantics='action after arriving at point', raster_equivalence='must be measured; continuous circles approximate upstream quantized raster'))


if __name__ == '__main__':
    main()
