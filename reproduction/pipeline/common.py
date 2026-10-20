"""Versioned local CHTR-style schema. No TensorFlow/PyTorch dependency."""
import hashlib
import html
import json
import os
from pathlib import Path
import re
import tempfile
from datetime import datetime, timezone

import cv2
import numpy as np

SCHEMA = 'chtr-local-v1'
ROOT = Path(__file__).resolve().parents[3]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=str(path.parent), delete=False, encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        temp = f.name
    os.replace(temp, str(path))


def iter_records(path):
    path = Path(path).resolve()
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            for key, val in list(row.items()):
                if (key.endswith('_path') or key in ('input_png', 'npz', 'model_config')) and isinstance(val, str):
                    row[key] = str((path.parent / val).resolve())
            yield row


def records(path):
    return list(iter_records(path))


def write_records(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=str(path.parent), delete=False, encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        temp = f.name
    os.replace(temp, str(path))


def digest(path):
    h = hashlib.sha256()
    with open(str(path), 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identifier(value):
    require(isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_-]+', value), 'ID must use ASCII letters, digits, _ or -')
    return value


def unique(rows, key='sample_id'):
    ids = [identifier(row[key]) for row in rows]
    require(len(ids) == len(set(ids)), 'Duplicate ' + key)


def output_dir(path):
    path = Path(path).resolve()
    require(not path.exists() or not any(path.iterdir()), 'Output must be new or empty: ' + str(path))
    path.mkdir(parents=True, exist_ok=True)
    return path


def gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    require(image is not None, 'Cannot read image: ' + str(path))
    if image.ndim == 3:
        if image.shape[2] == 4:
            alpha = image[:, :, 3:4].astype(float) / 255
            image = np.round(image[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    require(image.dtype == np.uint8, 'Only uint8 raster images supported')
    return image


def png(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(cv2.imwrite(str(path), image), 'Cannot write ' + str(path))
    return str(path.resolve())


def arrays(path):
    with np.load(str(path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def masks_and_labels(masks, labels):
    require(masks.ndim == 3 and masks.shape[1:] == (256, 256), 'Masks must be (N,256,256)')
    require(0 < len(masks) <= 30, 'Supported stroke count: 1..30')
    require(np.isfinite(masks).all() and np.isin(masks, [0, 1]).all(), 'Masks must be binary, finite')
    require(np.all(masks.reshape(len(masks), -1).sum(1) > 0), 'Empty stroke mask')
    require(labels.shape == (len(masks),) and np.issubdtype(labels.dtype, np.integer), 'Integer stroke labels required')
    require(np.all((labels >= 0) & (labels < 24)), 'Labels must use upstream 24-class IDs (0..23)')


def states(points, offsets):
    points = np.asarray(points, dtype=np.float32).copy()
    points[:, 3:] = [1, 0, 0]
    for end in offsets[1:]:
        points[int(end) - 1, 3:] = [0, 1, 0]
    points[-1, 3:] = [0, 0, 1]
    return points


def validate_points(points, offsets, size=256, bounds=True):
    require(points.ndim == 2 and points.shape[1] == 6 and len(points) > 0, 'Expected nonempty points[T,6]')
    require(np.isfinite(points).all() and np.all(points[:, 2] > 0), 'Nonfinite points or nonpositive diameter')
    offsets = np.asarray(offsets)
    require(offsets.ndim == 1 and np.issubdtype(offsets.dtype, np.integer), 'Integer offsets required')
    require(len(offsets) >= 2 and offsets[0] == 0 and offsets[-1] == len(points) and np.all(np.diff(offsets) > 0), 'Invalid subpath offsets')
    require(np.array_equal(points[:, 3:], states(points, offsets)[:, 3:]), 'Invalid pen states')
    if bounds:
        require(np.all((points[:, :2] >= 0) & (points[:, :2] <= size - 1)), 'Out-of-canvas points: review required; do not silently clip')


def render(points, offsets, size=256, supersample=4):
    validate_points(points, offsets, size, bounds=False)
    canvas = np.zeros((size * supersample, size * supersample), np.uint8)
    for start, end in zip(offsets[:-1], offsets[1:]):
        path = points[start:end, :3]
        for i, a in enumerate(path):
            b = path[min(i + 1, len(path) - 1)]
            count = max(1, int(np.ceil(np.linalg.norm(b[:2] - a[:2]) * supersample)))
            for t in np.linspace(0, 1, count + 1):
                p = a * (1 - t) + b * t
                cv2.circle(canvas, tuple(np.rint(p[:2] * supersample).astype(int)), max(1, int(round(p[2] * supersample / 2))), 255, -1)
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def iou(a, b):
    require(a.shape == b.shape, 'IoU canvas mismatch')
    a, b = a.astype(bool), b.astype(bool)
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def audit(reviewer, decision, source):
    require(bool(reviewer.strip()), 'Reviewer name required')
    return dict(reviewer=reviewer, decision=decision, source_sha256=digest(source), time=datetime.now(timezone.utc).isoformat(), schema=SCHEMA)


def interactive_decisions(template_path):
    """Terminal editor used after opening the generated local preview in a browser."""
    template=read_json(template_path)
    for key,original in template.items():
        print('\n'+key+'\n'+json.dumps(original,ensure_ascii=False,indent=2))
        print('Paste one JSON object with reviewer, approved/rejected status and edits; blank leaves pending.')
        line=input('decision> ').strip()
        if line:
            decision=json.loads(line)
            decision.setdefault('source_sha256',original['source_sha256'])
            template[key]=decision
    return {k:v for k,v in template.items() if v.get('status') in ('approved','rejected')}


def preview(directory, rows, title, limit=200):
    """Bounded self-contained preview; full data remains in the stage manifest."""
    import base64
    cards = []
    from itertools import islice
    require(limit >= 0, 'Preview limit must be nonnegative')
    for row in islice(rows, limit):
        images = []
        for key in ('image_path', 'input_png', 'render_path', 'overlay_path', 'preview_path'):
            if row.get(key) and Path(row[key]).is_file():
                uri = 'data:image/png;base64,' + base64.b64encode(Path(row[key]).read_bytes()).decode('ascii')
                images.append('<figure><img width="256" src="' + uri + '"><figcaption>' + key + '</figcaption></figure>')
        if row.get('trajectory_path'):
            z = arrays(row['trajectory_path'])
            parts = []
            for i, (a, b) in enumerate(zip(z['subpath_offsets'][:-1], z['subpath_offsets'][1:])):
                p = z['points'][a:b]
                xy = ' '.join('%.2f,%.2f' % tuple(v) for v in p[:, :2])
                parts.append('<polyline points="%s" fill="none" stroke="hsl(%d,80%%,40%%)" marker-end="url(#arrow)"/><text x="%.2f" y="%.2f">%d</text>' % (xy, i * 71 % 360, p[0, 0], p[0, 1], i))
            images.append('<svg width="256" height="256" viewBox="0 0 256 256"><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="red"/></marker></defs>' + ''.join(parts) + '</svg>')
        cards.append('<article>' + ''.join(images) + '<pre>' + html.escape(json.dumps(row, ensure_ascii=False, indent=2)) + '</pre></article>')
    Path(directory, 'index.html').write_text('<meta charset="utf-8"><title>' + html.escape(title) + '</title><style>body{font-family:sans-serif}article{border:1px solid #ccc;padding:12px}figure{display:inline-block;margin:5px}pre{white-space:pre-wrap}</style><h1>' + html.escape(title) + '</h1><p>Preview limited to %d samples; see manifest for all records.</p>' % limit + ''.join(cards), encoding='utf-8')
