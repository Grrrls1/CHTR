"""CASIA's documented little-endian isolated-character POT format."""
from pathlib import Path
import struct

import numpy as np


def decode_label(tag):
    if tag <= 0x7f:
        return chr(tag)
    # Actual official POT files also store ASCII in the high byte (e.g. 0x2100).
    if tag & 0xff == 0 and 0 < tag >> 8 <= 0x7f:
        return chr(tag >> 8)
    if tag > 0xffff:
        raise ValueError('Expected a two-byte GB label')
    return tag.to_bytes(2, 'big').decode('gbk')


def iter_pot(path):
    """Yield validated samples with raw point order and original stroke breaks."""
    path = Path(path)
    with path.open('rb') as stream:
        index = 0
        while True:
            offset = stream.tell()
            header = stream.read(8)
            if not header:
                return
            if len(header) != 8:
                raise ValueError(f'{path}:{offset}: truncated POT header')
            size, tag, count = struct.unpack('<HIH', header)
            if size < 12 or (size-8) % 4 or count < 1:
                raise ValueError(f'{path}:{offset}: invalid POT size/stroke count')
            body = stream.read(size-8)
            if len(body) != size-8:
                raise ValueError(f'{path}:{offset}: truncated POT body')
            pairs = np.frombuffer(body, dtype='<i2').reshape(-1, 2)
            strokes, current, ended = [], [], False
            for i, (x, y) in enumerate(pairs):
                if x == -1 and y == 0:
                    if not current:
                        raise ValueError(f'{path}:{offset}: empty stroke')
                    strokes.append(np.asarray(current, np.float32))
                    current = []
                elif x == -1 and y == -1:
                    if current or i != len(pairs)-1:
                        raise ValueError(f'{path}:{offset}: malformed character end')
                    ended = True
                elif x < 0 or y < 0 or ended:
                    raise ValueError(f'{path}:{offset}: invalid coordinate/end marker')
                else:
                    current.append((int(x), int(y)))
            if not ended or len(strokes) != count:
                raise ValueError(f'{path}:{offset}: stroke count or end marker mismatch')
            try:
                char, label_error = decode_label(tag), None
            except (UnicodeError, ValueError) as error:
                char, label_error = None, str(error)
            yield dict(index=index, byte_offset=offset, tag_code=tag, char=char,
                       label_error=label_error, strokes=strokes, writer_id=path.stem,
                       pot_file=str(path.resolve()))
            index += 1


def normalized_strokes(strokes, size=256, margin=8):
    """One shared isotropic bounding-box transform; preserve every original point."""
    points = np.concatenate(strokes)
    lo, hi = points.min(0), points.max(0)
    extent = hi-lo
    scale = (size-1-2*margin)/max(float(extent.max()), 1.)
    translation = (size-1-extent*scale)/2-lo*scale
    return [p*scale+translation for p in strokes], dict(scale=scale, translation=translation.tolist(),
                                                       raw_bbox=np.r_[lo, hi].tolist())
