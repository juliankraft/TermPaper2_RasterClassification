import xarray as xr
import lightning as L
from lightning.pytorch.callbacks import BasePredictionWriter
from pathlib import Path
import numpy as np
import os

from torch.utils.data import DataLoader
from torch import Tensor
from typing import Any, Sequence, cast

from src.model import LightningResNet


class PredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir: str | Path):
        super().__init__(write_interval='batch')

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def on_predict_start(
            self,
            trainer: L.Trainer,
            pl_module: LightningResNet) -> None:

        predict_dataloader = cast(DataLoader, trainer.predict_dataloaders)
        ds = cast(xr.Dataset, predict_dataloader.dataset.ds)  # type: ignore
        num_classes = pl_module.num_classes

        self.mask = ds.mask
        self.da = xr.full_like(
            ds['label'].load().expand_dims(cls=np.arange(num_classes), axis=-1),
            dtype=np.float16,
            fill_value=np.nan)
        self.da.name = 'label_prob'

    def write_on_batch_end(
            self,
            trainer: L.Trainer,
            pl_module: L.LightningModule,
            prediction: tuple[Tensor, dict[str, Tensor]],
            batch_indices: Sequence[int] | None,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int) -> None:

        predicted_label, xy_i = prediction

        predicted_label = predicted_label.cpu()
        xi = xy_i['xi'].cpu()
        yi = xy_i['yi'].cpu()

        for p, x, y in zip(predicted_label, xi, yi):
            self.da[{'x': x, 'y': y}] = p

    def on_predict_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:

        da = self.da

        cls_pred = xr.full_like(da.isel(cls=0), fill_value=np.nan)
        cls_pred.values = np.argmax(da.values, axis=-1).astype('float16')
        cls_pred = cls_pred.where(da.isel(cls=0).notnull())

        ds = xr.Dataset({
            'label_prob': da,
            'label_pred': cls_pred,
            'training_mask': self.mask
        })
        save_path = self.make_predition_path(self.output_dir)
        ds.to_zarr(save_path, mode='w')

    @staticmethod
    def make_predition_path(output_dir: str | Path) -> str:
        return os.path.join(output_dir, 'preds.zarr')
