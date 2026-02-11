import numpy as np
import torch
import os
from utils.config import DataConfig

class BatchIterator:
    def __init__(
        self,
        data_config: DataConfig,
        split: str,
        device: torch.device = torch.device('cpu'),
        device_type: str = 'cpu',
        sample_mode: str = 'unordered', # 'unordered' or 'permuted'
    ):
        self.split = split
        self.data_config = data_config
        self.device = device
        self.device_type = device_type
        self.data_dir = os.path.join('data', data_config.dataset)
        path = os.path.join(self.data_dir, f'{split}.bin')
        self.data = np.memmap(path, dtype=np.uint16, mode='r')
        self.max_start = len(self.data) - data_config.block_size
        self.ptr = 0
        self.rng: np.random.Generator = np.random.default_rng(42)
        self.sample_mode = sample_mode
        if sample_mode == 'permuted':
            self._reshuffle()
        elif sample_mode == 'unordered':
            pass
        else:
            raise ValueError(f"Unknown sample_mode '{sample_mode}'")

    def _reshuffle(self):
        self.order = self.rng.permutation(self.max_start)
        self.ptr = 0

    def next(self):
        if self.sample_mode == 'permuted' and self.ptr + self.data_config.batch_size > self.max_start:
            self._reshuffle()
        if self.sample_mode == 'permuted':
            ix = self.order[self.ptr:self.ptr + self.data_config.batch_size]
            self.ptr += self.data_config.batch_size
        elif self.sample_mode == 'unordered':
            ix = self.rng.integers(0, self.max_start, size=(self.data_config.batch_size,))
        else:
            raise ValueError(f"Unknown sample_mode '{self.sample_mode}'")
        x = torch.stack([torch.from_numpy(self.data[i:i+self.data_config.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i+1:i+1+self.data_config.block_size].astype(np.int64)) for i in ix])
        if self.device_type == 'cuda':
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y
    
    def get_batch(self):
        return self.next()