"""Standard-library-only lookup of exported handwriting trajectories; no robot I/O."""
import copy
import json
import math
from pathlib import Path


class TrajectoryDictionary:
    def __init__(self, directory):
        directory = Path(directory)
        self.path = directory if directory.is_file() else directory/'dictionary.json'
        self.root = self.path.resolve().parent
        self.index = json.loads(self.path.read_text(encoding='utf-8'))
        if self.index.get('schema_version') != 1:
            raise ValueError('Unsupported trajectory dictionary schema')

    def available_characters(self):
        return sorted(c for c, entry in self.index['characters'].items() if entry['default_variant'] is not None)

    def lookup(self, character, variant=None, normalized=True, allow_incomplete=False):
        if len(character) != 1:
            raise ValueError('lookup expects one character')
        entry = self.index['characters'].get(character)
        if entry is None:
            raise KeyError('Character absent from source dataset: '+character)
        variant = entry['default_variant'] if variant is None else variant
        item = next((v for v in entry['variants'] if v['id'] == variant), None)
        if item is None:
            raise ValueError('No completed usable variant yet: '+character)
        if item['status'] != 'converted' and not (allow_incomplete and item['status'] == 'needs_review'):
            raise ValueError('Variant is pending, failed, empty or truncated: '+character)
        path = (self.root/item['trajectory_json']).resolve()
        if self.root not in path.parents:
            raise ValueError('Trajectory path escapes library')
        data = json.loads(path.read_text(encoding='utf-8'))
        if data['character'] != character or data['variant_id'] != variant:
            raise ValueError('Dictionary/trajectory identity mismatch')
        points, offsets = data['points_xyw_px'], data['stroke_offsets']
        if not offsets or offsets[0] != 0 or offsets[-1] != len(points) or any(
                type(i) is not int for i in offsets) or any(a >= b for a, b in zip(offsets, offsets[1:])):
            raise ValueError('Invalid stroke offsets')
        if any(len(p) != 3 or not all(math.isfinite(v) for v in p) or p[2] <= 0 for p in points):
            raise ValueError('Invalid XYW points')
        scale = [255., 255., 256.] if normalized else [1., 1., 1.]
        strokes = [[[p[j]/scale[j] for j in range(3)] for p in points[a:b]]
                   for a, b in zip(offsets, offsets[1:])]
        metadata = copy.deepcopy({k: v for k, v in data.items() if k not in ('points_xyw_px', 'stroke_offsets')})
        return dict(metadata, strokes=strokes,
                    coordinates='normalized_xyw' if normalized else 'pixel_xyw',
                    axes='x right, y down', width_semantics='image-space stroke diameter, not force or pressure',
                    pen_up_between_strokes=True)

    def lookup_text(self, text, normalized=True):
        if not text:
            raise ValueError('Text must not be empty')
        # Resolve all characters before returning a plan; never silently substitute missing glyphs.
        return dict(text=text, characters=[self.lookup(c, normalized=normalized) for c in text],
                    coordinate_frame='character-local image plane; no robot/world calibration applied',
                    contains_robot_motion_commands=False)
