import xarray as xr
import rioxarray as rxr
import shutil
import os
from math import ceil
from tqdm import tqdm

print("Libraries imported...    ")


class ChunkWriter():
    def __init__(self, tiff_path: str, save_path: str, chunk_size: int = 100):
        self.da = rxr.open_rasterio(tiff_path)
        self.save_path = save_path
        self.chunk_size = chunk_size

        self.num_y = len(self.da.y)

        self.current_chunk = 0

    def __iter__(self):
        if os.path.exists(self.save_path):
            shutil.rmtree(self.save_path)

        return self

    def __next__(self):
        self.write_chunk(self.current_chunk)

        if (self.current_chunk * self.chunk_size) > self.num_y:
            raise StopIteration

        self.current_chunk += 1

    def __len__(self) -> int:
        return ceil(self.num_y / self.chunk_size)

    def write_chunk(self, chunk: int):

        da = self.da.isel(y=slice(chunk * self.chunk_size, (chunk + 1) * self.chunk_size)).load()

        ds = xr.Dataset(
            {
                'rs': da.sel(band=[1, 2, 3, 4]),
                'mask': da.sel(band=5),
                'category': da.sel(band=6),
            }
        )

        ds['sealed'] = da.sel(band=7)
        ds['sealed'] = ds['sealed'].where(ds['sealed'] != 1, 0)
        ds['sealed'] = ds['sealed'].where(ds['sealed'] != 2, 1)
        ds['sealed'] = ds['sealed'].where(ds['sealed'] != 3, 2)

        ds['sealed_simple'] = ds['sealed'].where(ds['sealed'] != 2, 1)

        ds = ds.transpose('band', 'x', 'y')

        category_dict = {
            0: "ConstructionSite",
            1: "Building",
            2: "BuildingDistortion",
            3: "GreenAreas",
            4: "RoadAsphalt",
            5: "Forest",
            6: "WaterBasin",
            7: "Path",
            8: "MeadowPasture",
            9: "SealedObjects"
        }

        sealed_dict = {
            0: "unsealed",
            1: "sealed",
            2: "unknown"
        }

        # Dataset attributes
        ds.attrs = {'creator': 'Julian Kraft'}

        # Variable attributes
        ds['rs'].attrs.update({'source': 'SwissImage RS'})
        ds['category'].attrs.update({'classes': ', '.join([f'{k}={v}' for k, v in category_dict.items()])})
        ds['sealed'].attrs.update({'classes': ', '.join([f'{k}={v}' for k, v in sealed_dict.items()])})
        ds['sealed_simple'].attrs.update({
            'classes': ', '.join([f'{k}={v}' for k, v in sealed_dict.items() if k != 2])
            })

        ds['rs'] = ds.rs.chunk({'band': 4, 'x': self.chunk_size, 'y': self.chunk_size})
        ds['category'] = ds.category.chunk({'x': self.chunk_size, 'y': self.chunk_size})
        ds['sealed'] = ds.sealed.chunk({'x': self.chunk_size, 'y': self.chunk_size})
        ds['sealed_simple'] = ds.sealed_simple.chunk({'x': self.chunk_size, 'y': self.chunk_size})
        ds['mask'] = ds.mask.chunk({'x': -1, 'y': -1})

        encoding = {}

        for variable in ds.data_vars:
            encoding[variable] = {'compressor': None}

            if variable == 'rs':
                ds[variable] = ds[variable].astype('uint16')

            else:
                ds[variable] = ds[variable].where(ds[variable] != -9999, 255).astype('uint8')

            ds[variable].attrs = {}

        if chunk == 0:
            kwargs = {}
        else:
            for variable in ds.data_vars:
                ds[variable].attrs = {}
            kwargs = {'append_dim': 'y'}

        ds.to_zarr(self.save_path, mode='a', **kwargs)

    def write(self, dev_mode: bool = False):
        for i, _ in enumerate(tqdm(self, ncols=80, desc='Writing chunks')):
            if dev_mode and (i > 1):
                raise StopIteration


if __name__ == '__main__':
    print("Classes defined...    ")

    # datatset path
    tiff_path = '/cfs/earth/scratch/kraftjul/sa2/data/dataset_category_sealing/ds_categorys_sealing.tif'
    save_path = '/cfs/earth/scratch/kraftjul/sa2/data/ds_categorys_sealing.zarr'

    # # testset psth
    # tiff_path='/cfs/earth/scratch/kraftjul/sa2/data/Sample_CombinedData_32signed/Sample_CombinedData32signed.tif'
    # save_path='/cfs/earth/scratch/kraftjul/sa2/data/sample_combined.zarr'

    chunkwriter = ChunkWriter(
        tiff_path=tiff_path,
        save_path=save_path,
        chunk_size=500)

    print("ChunkWriter initialized...    ")

    chunkwriter.write()

    print("ChunkWriter written...    ")
