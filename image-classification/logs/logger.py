import os
from collections import defaultdict, deque
from functools import partial

import torch


class Meter:
    def __init__(self, window_size=20, delimiter="\n"):
        self.total = 0
        self.count = 0
        self.deque = deque(maxlen=window_size)
        self.delimiter = delimiter

    def __str__(self):
        return f"Global Metrics ({self.count} Elements){self.delimiter}Global Average : {self.global_average}{self.delimiter}Window Metrics ({self.deque.maxlen} Elements){self.delimiter}Average : {self.average}{self.delimiter}Median : {self.median}{self.delimiter}Max Value : {self.max}{self.delimiter}"

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def reset(self):
        self.total = 0
        self.count = 0
        self.deque = deque(maxlen=self.deque.maxlen)

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def average(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_average(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)


class Logger:
    def __init__(
        self,
        logger_dir,
        log_freq,
        max_iter,
        rank=0,
        is_train=True,
        file_prefix="",
        delimiter="\n",
    ):
        self.meters = defaultdict(
            partial(Meter, window_size=log_freq, delimiter=delimiter)
        )
        self.max_iter = max_iter
        self.log_freq = log_freq
        self.delimiter = delimiter
        self.logger_dir = os.path.join(logger_dir, f"rank_{rank}")
        self.logger_file = (
            file_prefix + "_train.txt" if is_train else file_prefix + "_eval.txt"
        )

        if not os.path.exists(self.logger_dir):
            os.makedirs(self.logger_dir)

    def __getitem__(self, key):
        if key in self.meters:
            return self.meters[key]
        else:
            raise KeyError("No such key exists")

    def update(self, epoch, step, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue

            if isinstance(v, tuple):
                v, n = v
            else:
                n = 1

            if isinstance(v, torch.Tensor):
                v = v.item()

            assert isinstance(v, (int, float))
            self.meters[k].update(v, n)

        if step % self.log_freq == 0 or step == self.max_iter:
            self.log_file(epoch, step)

    def log_file(self, epoch, step):
        with open(os.path.join(self.logger_dir, self.logger_file), "a") as log:
            log.write(
                f"Logging Metrics for Epoch {epoch} at Batch {step}{self.delimiter}"
            )
            for key, value in self.meters.items():
                log.write(f"Metric : {key}{self.delimiter}")
                log.write(str(value) + self.delimiter)

    def reset(self):
        for item in self.meters.values():
            item.reset()
