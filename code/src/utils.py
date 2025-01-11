import xarray as xr
import lightning as L
from lightning.pytorch.callbacks import BasePredictionWriter
from pathlib import Path
import numpy as np
import torch
import os

from torch.utils.data import DataLoader
from torch import Tensor
from typing import Any, Sequence, cast

from src.model import LightningResNet


class PredictionWriter(BasePredictionWriter):
    def __init__(
            self,
            output_dir: str | Path,
            label_type: str):
        super().__init__(write_interval='batch')

        self.output_dir = output_dir
        self.label_type = label_type
        os.makedirs(output_dir, exist_ok=True)

    def on_predict_start(
            self,
            trainer: L.Trainer,
            pl_module: LightningResNet) -> None:

        print('running on_predict_start', flush=True)  # Debugging
        print('initializing predict dataloaders', flush=True)  # Debugging
        predict_dataloader = cast(DataLoader, trainer.predict_dataloaders)
        print('getting dataset', flush=True)  # Debugging
        ds = cast(xr.Dataset, predict_dataloader.dataset.ds)  # type: ignore
        num_classes = pl_module.num_classes

        print('getting mask', flush=True)  # Debugging
        self.mask = ds.mask

        print('initializing da', flush=True)  # Debugging
        label_attr = getattr(ds, self.label_type)
        self.da = xr.full_like(
            label_attr.load().expand_dims(cls=np.arange(num_classes), axis=-1),
            dtype=np.float16,
            fill_value=np.nan)

        print('initializing cls_pred', flush=True)  # Debugging
        self.cls_pred = xr.full_like(
            label_attr.load(),
            dtype=np.float16,
            fill_value=np.nan)

        print('done with on_predict_start', flush=True)  # Debugging

    def write_on_batch_end(
            self,
            trainer: L.Trainer,
            pl_module: L.LightningModule,
            prediction: tuple[Tensor, dict[str, Tensor]],
            batch_indices: Sequence[int] | None,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int) -> None:

        print('running write_on_batch_end', flush=True)  # Debugging

        predicted_label, batch = prediction

        predicted_label = predicted_label.cpu()
        xi_patch: Tensor = torch.stack(batch['patch_coord']['xi'], -1).cpu()
        yi_patch: Tensor = torch.stack(batch['patch_coord']['yi'], -1).cpu()

        for p, x, y in zip(predicted_label, xi_patch, yi_patch):
            self.da[{'x': slice(*x), 'y': slice(*y)}] = p
            predicted_class = np.argmax(p, axis=-1).type(torch.float16).numpy()
            nan_mask = np.isnan(p).any(axis=-1)
            predicted_class[nan_mask] = np.nan
            self.cls_pred[{'x': slice(*x), 'y': slice(*y)}] = predicted_class

    def on_predict_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:

        print('running on_predict_end', flush=True)  # Debugging
        da = self.da

        ds = xr.Dataset({
            'label_prob': da,
            'label_pred': self.cls_pred,
            'training_mask': self.mask
        })
        print('saving prediction', flush=True)
        save_path = self.make_predition_path(self.output_dir)
        ds.to_zarr(save_path, mode='w')

    @staticmethod
    def make_predition_path(output_dir: str | Path) -> str:
        return os.path.join(output_dir, 'preds.zarr')
