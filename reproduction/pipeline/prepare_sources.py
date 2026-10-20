"""Normalize supplied raster-source JSONL to a 256 canvas; preserve provenance."""
import argparse
import hashlib
import struct
from pathlib import Path
import cv2
import numpy as np
from common import records, write_records, output_dir, gray, png, require, unique, digest, SCHEMA, identifier


def expand_sources(rows, staging):
    """Accept raster records, font records, or CASIA GNT streams (little endian)."""
    result=[]
    for record in rows:
        r=dict(record)
        if r.get('gnt_path'):
            source=Path(r['gnt_path']); source_hash=digest(source)
            require(r.get('writer_id') and r.get('source'),'GNT writer_id and source must be supplied')
            count=0
            with source.open('rb') as f:
                while True:
                    start=f.tell(); length_bytes=f.read(4)
                    if not length_bytes: break
                    require(len(length_bytes)==4,'Truncated GNT length')
                    length=struct.unpack('<I',length_bytes)[0]
                    header=f.read(6); require(len(header)==6,'Truncated GNT header')
                    character=header[:2].decode(r.get('encoding','gb18030'))
                    w,h=struct.unpack('<HH',header[2:]); require(w>0 and h>0 and length==10+w*h,'Invalid GNT sample size')
                    pixels=f.read(w*h); require(len(pixels)==w*h,'Truncated GNT pixels')
                    sid=hashlib.sha256((source_hash+'|'+str(start)).encode()).hexdigest()[:20]
                    path=png(staging/(sid+'.png'),np.frombuffer(pixels,np.uint8).reshape(h,w))
                    result.append(dict(r,sample_id=sid,image_path=path,character=character,gnt_offset=start,gnt_sha256=source_hash))
                    count+=1
            require(count>0,'Empty GNT stream')
        elif r.get('font_path'):
            from PIL import Image, ImageDraw, ImageFont
            require(r.get('character') and r.get('style_id') and r.get('source'),'Font character/style_id/source required')
            size=int(r.get('font_size',192)); require(1<=size<=2048,'Invalid font_size')
            font=ImageFont.truetype(r['font_path'],size)
            width,height=font.getsize(r['character']); require(width>0 and height>0,'Empty font glyph')
            image=Image.new('L',(width+32,height+32),255); ImageDraw.Draw(image).text((16,16),r['character'],font=font,fill=0)
            sid=r.get('sample_id') or hashlib.sha256((digest(r['font_path'])+'|'+r['character']+'|'+str(size)).encode()).hexdigest()[:20]
            identifier(sid)
            path=png(staging/(sid+'.png'),np.asarray(image))
            result.append(dict(r,image_path=path,font_sha256=digest(r['font_path'])))
        else:
            result.append(r)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, help='JSONL: image_path, character, writer_id or style_id, source; optional sample_id, polarity')
    p.add_argument('--output', required=True)
    p.add_argument('--margin', type=int, default=8)
    a = p.parse_args()
    require(0 <= a.margin < 128, 'Invalid margin')
    out = output_dir(a.output)
    staging=out/'source_rasters'; staging.mkdir()
    rows = expand_sources(records(a.manifest),staging)
    for r in rows:
        provenance = r.get('writer_id') or r.get('style_id') or r.get('provenance_id')
        require(r.get('character') and r.get('source') and provenance, 'character/source and writer_id, style_id or provenance_id required')
        r.setdefault('sample_id', hashlib.sha256((r['source'] + '|' + r['character'] + '|' + str(provenance) + '|' + digest(r['image_path'])).encode()).hexdigest()[:20])
    unique(rows)
    for r in rows:
        im = gray(r['image_path'])
        polarity = r.get('polarity', 'black_on_white')
        require(polarity in ('black_on_white', 'white_on_black'), 'Explicit valid polarity required')
        if polarity == 'white_on_black': im = 255-im
        require(np.any(im < 128), 'Blank source image')
        h, w = im.shape
        scale = (256 - 2*a.margin)/float(max(h, w))
        nw, nh = max(1, round(w*scale)), max(1, round(h*scale))
        left, top = (256-nw)//2, (256-nh)//2
        canvas = np.full((256,256),255,np.uint8)
        canvas[top:top+nh,left:left+nw] = cv2.resize(im,(nw,nh),interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        r.update(source_image_path=r['image_path'], source_sha256=digest(r['image_path']), image_path=png(out/(r['sample_id']+'.png'),canvas), transform=dict(original_size=[w,h], resized_size=[nw,nh], nominal_scale=scale, scale_xy=[nw/w,nh/h], padding=[left,top], pixel_mapping='(x+0.5)*scale_xy-0.5+padding'), canvas=256, schema=SCHEMA)
    write_records(out/'manifest.jsonl',rows)


if __name__ == '__main__': main()
