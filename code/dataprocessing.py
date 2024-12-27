import xarray as xr
import rioxarray as rxr
import shutil
import os
from math import ceil
from tqdm import tqdm


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
                'label': da.sel(band=6),
            }
        )

        ds = ds.transpose('band', 'x', 'y')

        category_dict_reversed = {
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

        # Dataset attributes
        ds.attrs = {'creator': 'Julian Kraft'}

        # Variable attributes
        ds['rs'].attrs.update({'source': 'SwissImage RS'})
        ds['label'].attrs.update({'classes': ', '.join([f'{k}={v}' for k, v in category_dict_reversed.items()])})

        ds['rs'] = ds.rs.chunk({'band': 4, 'x': self.chunk_size, 'y': self.chunk_size})
        ds['label'] = ds.label.chunk({'x': self.chunk_size, 'y': self.chunk_size})
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

    tiff_path = '/cfs/earth/scratch/kraftjul/sa2/data/CombinedData_32signed/CombinedData32signed.tif'
    save_path = '/cfs/earth/scratch/kraftjul/sa2/data/combined.zarr'

    if not os.path.exists(save_path):
        raise FileNotFoundError(f'Tiff file {save_path} does not exist')

    chunkwriter = ChunkWriter(
        tiff_path=tiff_path,
        save_path=save_path,
        chunk_size=500)
    chunkwriter.write()
