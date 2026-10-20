"""Epoch-level training-loss stopping, with checkpointable partial-epoch means."""
import math


class EarlyStopping:
    def __init__(self, config=None):
        config = config or {}
        self.patience = config.get('patience', 0)
        self.min_delta = float(config.get('min_delta', 0.0))
        self.min_epochs = config.get('min_epochs', 1)
        if (type(self.patience) is not int or self.patience < 0 or
                type(self.min_epochs) is not int or self.min_epochs < 1 or
                not math.isfinite(self.min_delta) or self.min_delta < 0):
            raise ValueError('early_stopping requires patience>=0, min_epochs>=1, min_delta>=0')
        self.best = None
        self.bad_epochs = 0
        self.last_epoch = 0
        self.loss_sum = 0.0
        self.samples = 0

    def add(self, loss, batch_size):
        if not math.isfinite(loss) or batch_size <= 0:
            raise ValueError('Expected finite loss and positive batch size')
        self.loss_sum += loss * batch_size
        self.samples += batch_size

    def finish_epoch(self, epoch):
        if not self.samples:
            raise ValueError('Cannot monitor an empty epoch')
        mean = self.loss_sum / self.samples
        improved = self.best is None or mean < self.best - self.min_delta
        if improved:
            self.best = mean
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        self.last_epoch = epoch
        self.loss_sum, self.samples = 0.0, 0
        return mean, improved, self.should_stop

    @property
    def should_stop(self):
        return bool(self.patience and self.last_epoch >= self.min_epochs and
                    self.bad_epochs >= self.patience)

    def state_dict(self):
        return dict(best=self.best, bad_epochs=self.bad_epochs, last_epoch=self.last_epoch,
                    loss_sum=self.loss_sum, samples=self.samples)

    def load_state_dict(self, state):
        for name in self.state_dict():
            setattr(self, name, state[name])
