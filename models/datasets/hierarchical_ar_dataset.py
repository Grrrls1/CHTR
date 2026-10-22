"""Read image and original trajectory only; no mask/type annotations required."""
import hashlib
import io
import json
from pathlib import Path
import cv2
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    def __init__(self,path):
        self.path=str(Path(path).resolve());self.env=None
        with lmdb.open(self.path,readonly=True,lock=False,readahead=False) as env:
            with env.begin() as txn:
                all_ids=json.loads(txn.get(b'__ids__'))
                ready=txn.get(b'__ids_ready__')
                self.ids=json.loads(ready) if ready is not None else [sid for sid in all_ids if json.loads(txn.get(sid.encode())).get('trajectory_key')]
        if not self.ids or len(set(self.ids))!=len(self.ids):
            raise ValueError('requires a nonempty ready dataset with unique IDs')

    def _env(self):
        if self.env is None:self.env=lmdb.open(self.path,readonly=True,lock=False,readahead=False)
        return self.env

    def close(self):
        if self.env is not None:self.env.close();self.env=None

    def __len__(self):return len(self.ids)

    def _assets(self,txn,index):
        row=json.loads(txn.get(self.ids[index].encode()))
        assets=[txn.get(row[key].encode()) for key in ('image_key','trajectory_key')]
        if any(value is None for value in assets):raise ValueError(self.ids[index]+': missing image/trajectory')
        return row,*assets

    def inspect(self,max_strokes,max_points):
        """Validate every actual offset array and fingerprint inputs for safe resume."""
        lengths=[];digest=hashlib.sha256()
        with self._env().begin() as txn:
            for index,sid in enumerate(self.ids):
                row,image,trajectory=self._assets(txn,index)
                with np.load(io.BytesIO(trajectory),allow_pickle=False) as z:
                    off=z['stroke_offsets'];n=len(z['points']);counts=np.diff(off)
                    if off.ndim!=1 or off.dtype.kind not in 'iu' or len(off)<2 or off[0]!=0 or off[-1]!=n or (counts<1).any():
                        raise ValueError(sid+': invalid offsets')
                    if len(counts)>max_strokes or counts.max()>max_points:
                        raise ValueError(sid+': labels exceed caps; no silent truncation')
                    lengths.append(counts.astype(int).tolist())
                for value in (sid.encode(),image,trajectory):
                    digest.update(len(value).to_bytes(8,'little'));digest.update(value)
        return lengths,digest.hexdigest()

    def __getitem__(self,index):
        with self._env().begin() as txn:row,image,trajectory=self._assets(txn,index)
        image=cv2.imdecode(np.frombuffer(image,np.uint8),cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape[0]!=image.shape[1]:raise ValueError('requires square image')
        size=image.shape[0]
        with np.load(io.BytesIO(trajectory),allow_pickle=False) as z:
            points=z['points'][:,:3].astype(np.float32,copy=True);off=z['stroke_offsets'].copy()
            if 'image_size' in z and int(z['image_size'])!=size:raise ValueError('trajectory canvas differs')
        if not np.isfinite(points).all() or (points[:,2]<=0).any() or (points[:,:2]<0).any() or (points[:,:2]>size-1).any():
            raise ValueError('invalid trajectory geometry')
        if off.dtype.kind not in 'iu' or len(off)<2 or off[0]!=0 or off[-1]!=len(points) or (np.diff(off)<1).any():
            raise ValueError('invalid trajectory boundaries')
        points[:,:2]/=size-1;points[:,2]/=size
        return dict(image=torch.from_numpy(image.astype(np.float32)[None]/255.),points=torch.from_numpy(points),
                    stroke_offsets=torch.from_numpy(off.astype(np.int64)),image_size=size,
                    sample_id=self.ids[index],character=row.get('character',''))
