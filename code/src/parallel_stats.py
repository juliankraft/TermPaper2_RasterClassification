# Adapted from:
# - https://github.com/bask0/drought_ml/blob/master/preprocessing/compute_stats.py
# - https://github.com/bask0/drought_ml/blob/master/preprocessing/chunk_parallel.py


import multiprocessing.pool as mpp
from multiprocessing import Pool
import tqdm
import numpy as np
import xarray as xr
import dask
from os import PathLike
import torch

from typing import Callable

dask.config.set(scheduler='synchronous')


def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)


mpp.Pool.istarmap = istarmap


def chunk_parcall(
        path: str | PathLike,
        var: str | list[str],
        num_processes: int,
        disable_progress_bar: bool,
        fun: Callable,
        use_mask: bool,
        mask: xr.DataArray | None = None,
        xrcombine: bool = False,
        desc: str = 'Processing |',
        args: tuple = ()):
    """Apply function parallelized over data chunks.

    Args:
        path: path to .zarr data cube.
        var: variable from cube.
        num_processes: number of processes.
        fun: A function with signature:
            if use_mask:
                path, var, mask, lat_slice, lon_slice
            else:
                path, var, lat_slice, lon_slice
        use_mask: if True, only non-masked chunks are processed.
        mask: if passed, this mask is used if 'use_mask=True'.
        xrcombine: if True, the results are combined into one dataset. Else, the
            chunks will be returned as a list (default).
        desc: the tqdm description to desplay before the variable name. Default is
            'Processing |', which displays as 'Processing | my_var'.
        args: tuple passed to `fun(..., *args)` as last positional agruments.
        subset: if passed, compute 'fun' on subset. The argument is passed to dataset.isel(**subset), for example,
            `time=0` would mean that 'fun' is applied to dataset.isel(time=0).

    Returns:
        A list of return values from `fun`.

    """
    cube = xr.open_zarr(path)

    if use_mask:
        if mask is None:
            mask = cube.mask.load()
        else:
            mask = mask.load()

    da = cube[var]

    lat_chunk_bounds = np.lib.stride_tricks.sliding_window_view(
        np.concatenate((np.zeros(1, dtype=int), np.cumsum(da.chunksizes['x']))), 2)
    lon_chunk_bounds = np.lib.stride_tricks.sliding_window_view(
        np.concatenate((np.zeros(1, dtype=int), np.cumsum(da.chunksizes['y']))), 2)

    iterable = []

    for lat_chunk_bound in lat_chunk_bounds:
        for lon_chunk_bound in lon_chunk_bounds:
            lat_slice = slice(*lat_chunk_bound)
            lon_slice = slice(*lon_chunk_bound)

            if use_mask:
                if mask.isel(x=lat_slice, y=lon_slice).sum() > 0:
                    iterable.append((path, var, mask, lat_slice, lon_slice, *args))
            else:
                iterable.append((path, var, lat_slice, lon_slice, *args))

    results = []

    if disable_progress_bar:
        with Pool(num_processes) as pool:
            for r in pool.istarmap(fun, iterable):
                results.append(r)
    else:

        if isinstance(var, list):
            desc_var = ', '.join(var)
        else:
            desc_var = var

        with Pool(num_processes) as pool:
            for r in tqdm.tqdm(
                    pool.istarmap(fun, iterable),
                    total=len(iterable),
                    desc=f'{desc} {desc_var:<25}',
                    ncols=160):
                results.append(r)

    if xrcombine:
        results = xr.combine_by_coords(results)
        sel_dims = {dim: 0 for dim in set(da.dims) - {'x', 'y'}}
        results = results.broadcast_like(da.isel(**sel_dims))
        if results.lat[0] < results.lat[1]:
            results = results.reindex(x=list(reversed(results.lat)))

    return results


def batch_stats(
        path: str,
        var: str,
        mask: xr.DataArray,
        lat_subset: slice,
        lon_subset: slice) -> None:

    da = xr.open_zarr(path)[var.lower()].isel(x=lat_subset, y=lon_subset).load()
    da = da.where(mask).load()
    # counts = da.notnull().sum().compute().item()
    # sums = da.sum().compute().item()
    # sq_sums = (da ** 2).sum().compute().item()

    counts = da.notnull().sum(['x', 'y']).compute()
    sums = da.sum(['x', 'y']).compute()
    sq_sums = (da ** 2).sum(['x', 'y']).compute()

    return counts, sums, sq_sums


def par_stats(
        path: str | PathLike,
        variables: list[str],
        mask: xr.DataArray | None = None,
        num_processes: int = 1,
        disable_progress_bar: bool = False):

    print(f'Computing stats - num_processes {num_processes}...') # Debugging

    stats = {}

    for var in variables:

        results = chunk_parcall(
            path=path,
            var=var,
            num_processes=num_processes,
            disable_progress_bar=disable_progress_bar,
            fun=batch_stats,
            use_mask=True,
            mask=mask,
            desc='Computing stats |'
        )

        if len(results) == 0:
            raise RuntimeError(
                'no stats returned.'
            )

        n = 0.
        s = 0.
        s2 = 0.

        for n_, s_, s2_ in results:
            n += n_
            s += s_
            s2 += s2_

        mn = s / n
        sd = ((s2 / n) - (s / n) ** 2) ** 0.5

        stats[var] = {'mean': mn, 'std': sd}

    return stats


def batch_class_count(
        path: str,
        var: str,
        mask: xr.DataArray,
        lat_subset: slice,
        lon_subset: slice) -> None:

    da = xr.open_zarr(path)[var.lower()].isel(x=lat_subset, y=lon_subset).load()
    da = da.where(mask).load()

    # label_count = ds['label'].groupby(ds['label']).count().compute()

    unique, count = np.unique(da.values, return_counts=True)

    return unique, count


def par_class_weights(
        path: str | PathLike,
        variable: str,
        num_classes: int,
        mask: xr.DataArray | None = None,
        num_processes: int = 1,
        disable_progress_bar: bool = False):

    print(f'Computing class weights - num_processes {num_processes}...') # Debugging

    results = chunk_parcall(
        path=path,
        var=variable,
        num_processes=num_processes,
        disable_progress_bar=disable_progress_bar,
        fun=batch_class_count,
        use_mask=True,
        mask=mask,
        desc='Computing stats |'
    )

    if len(results) == 0:
        raise RuntimeError(
            'no stats returned.'
        )

    val_counts = {}

    for values, counts in results:
        for value_, count_ in zip(values, counts):
            value = value_.item()
            count = count_.item()
            if np.isnan(value):
                continue
            if value not in val_counts:
                val_counts[value] = count
            else:
                val_counts[value] += count

    total_count = np.sum(list(val_counts.values()))

    class_weights = []
    for i in range(num_classes):
        if i not in val_counts:
            class_weights.append(1.0)
        else:
            class_weights.append((total_count - val_counts[i]) / total_count)

    return torch.tensor(class_weights, dtype=torch.float32)
