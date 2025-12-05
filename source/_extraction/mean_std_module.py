from typing import Any
import torch
import lightning as pl
import mrcfile
import os
import h5py
import numpy as np
import tifffile

from source.reconstruction.extractor import Extractor


class MeanStdModule(pl.LightningModule):
    """
    Class for calculating mean and std of dataset.
    """
    def __init__(self, *args, **kwargs):
        super(MeanStdModule, self).__init__()
        self.save_hyperparameters(logger=False, ignore=['extractor'])
        self.mean = None
        self.std = None
        self.min = None
        self.max = None

        self._mean = 0
        self._std = 0
        self._count = 0
        self._min = 0
        self._max = 0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.mean(dim=-3)
        if self.mean is None:
            self._mean += x.mean(dim=(-1, -2)).sum(dim=0)
        else:
            self._std += ((x - self.mean) ** 2).mean(dim=(-1, -2)).sum(dim=0)

        self._count += x.shape[0]
        if self.min is None:
            self._min = min(x.min().item(), self._min)
        if self.max is None:
            self._max = max(self._max, x.max().item())
        return self._mean / self._count, self._std / self._count, self._min, self._max
    
    def test_step(
        self, batch: dict[str, torch.Tensor], *args: Any, **kwargs: Any
    ) -> dict[str, list[torch.Tensor]]:
        self.predict_step(batch=batch, args=args, kwargs=kwargs)

    def predict_step(
        self, batch: dict[str, torch.Tensor], *args: Any, **kwargs: Any
    ) -> dict[str, list[torch.Tensor]]:
        self.forward(batch['images'])

    def on_predict_epoch_end(self) -> None:
        if self.mean is None:
            self.mean = (self._mean / self._count).item()
        self.std = (torch.sqrt(self._std / self._count)).item()
        self.min = self._min
        self.max = self._max

        self._mean = 0
        self._std = 0
        self._count = 0
        self._min = 0
        self._max = 0

    def on_test_epoch_end(self) -> None:
        self.on_predict_epoch_end()
