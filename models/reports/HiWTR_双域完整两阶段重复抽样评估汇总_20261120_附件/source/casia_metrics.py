"""Explicit CASIA evaluation metrics; no implicit alignment or sequence resampling."""
import numpy as np


def _dtw(a, b):
    """Exact unconstrained DTW, Euclidean local cost and min-total-cost path."""
    n, m = len(a), len(b)
    previous = np.full(m+1, np.inf)
    previous[0] = 0.
    previous_lengths = np.zeros(m+1, np.int64)
    for i in range(n):
        current = np.full(m+1, np.inf)
        lengths = np.zeros(m+1, np.int64)
        for j in range(m):
            # Tie order: diagonal, vertical, horizontal.
            best = previous[j]
            steps = previous_lengths[j]
            if previous[j+1] < best:
                best, steps = previous[j+1], previous_lengths[j+1]
            if current[j] < best:
                best, steps = current[j], lengths[j]
            dx, dy = a[i, 0]-b[j, 0], a[i, 1]-b[j, 1]
            current[j+1] = best+np.sqrt(dx*dx+dy*dy)
            lengths[j+1] = steps+1
        previous, previous_lengths = current, lengths
    return previous[m], previous_lengths[m]


try:
    from numba import njit
    _dtw = njit(cache=True)(_dtw)
    DTW_BACKEND = 'numba exact CPU'
except ImportError:
    DTW_BACKEND = 'Python exact CPU (install numba for full evaluation)'


def dtw_xy(reference, prediction):
    if not len(reference) or not len(prediction):
        return dict(dtw_sum_px=None, dtw_path_mean_px=None, dtw_path_length=0,
                    dtw_status='undefined_empty_sequence')
    if not np.isfinite(reference).all() or not np.isfinite(prediction).all():
        raise ValueError('Nonfinite DTW coordinates')
    cost, length = _dtw(np.ascontiguousarray(reference[:, :2], dtype=np.float64),
                        np.ascontiguousarray(prediction[:, :2], dtype=np.float64))
    return dict(dtw_sum_px=float(cost), dtw_path_mean_px=float(cost/length),
                dtw_path_length=int(length), dtw_status='ok')


class FeatureMoments:
    """Streaming float64 mean/covariance, bounded memory independent of sample count."""
    def __init__(self):
        self.count = 0
        self.mean = None
        self.m2 = None

    def update(self, features):
        features = np.asarray(features, dtype=np.float64)
        if not len(features):
            return
        mean = features.mean(0)
        centered = features-mean
        m2 = centered.T @ centered
        if self.count == 0:
            self.mean, self.m2, self.count = mean, m2, len(features)
            return
        delta = mean-self.mean
        total = self.count+len(features)
        self.m2 += m2+np.outer(delta, delta)*(self.count*len(features)/total)
        self.mean += delta*(len(features)/total)
        self.count = total

    def covariance(self):
        if self.count < 2:
            raise ValueError('FID requires at least two samples per distribution')
        return self.m2/(self.count-1)
