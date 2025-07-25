
import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features
from shapely.geometry import box
from pathlib import Path

from product_path import ProductPath


class RasterData:
    '''
    Clase para pasar la data geográfica (y metadata) del patch de una función a otra.
    Esta en conjunto con el np.ndarray del patch constituyen toda la información
    extraída en su creación.
    '''

    def __init__(
            self,
            raster_reader: rasterio.io.DatasetReader,
            window: rasterio.windows.Window
    ):
        self.bounds = box(*raster_reader.window_bounds(window))
        self.transform = raster_reader.window_transform(window)
        self.shape = (256, 256)
        self.bands = [
            "B01", "B02", "B03", "B04", "B05",
            "B06", "B07", "B08", "B09", "B11", "B12", "B8A",
            ]

    def set_dates(self, dates):
        self.dates = dates
        return self

    def set_process_nums(self, process_nums):
        self.process_nums = process_nums
        return self

    def __repr__(self):
        return (
            f"Object attributes: bounds, transform, shape, bands, dates, process_nums.\n"
            f"Tensor_shape: ({len(self.dates)}, 12, 256, 256)\n"
            f"Dates: \n\t"
            + "\n\t".join([str(date) for date in self.dates])
                    )

    def __str__(self):
        return (
            f"Object attributes: bounds, transform, shape, bands, dates, process_nums.\n"
            f"Tensor_shape: ({len(self.dates)}, 12, 256, 256)\n"
            f"Dates: \n\t"
            + "\n\t".join([str(date) for date in self.dates])
                    )


class Patch:
    '''
    Sattelite Image Time Series.
    Clase correspondiente al tensor 4D y su data.
    '''

    def __init__(
        self,
        tile_name: str,
        patch_n: int,
    ):
        self.tile_name = tile_name
        self.patch_n = patch_n
        self.id = self.get_id()

    def get_annotation_raster(self, labels_gdf: gpd.GeoDataFrame) -> np.ndarray:
        assert hasattr(self, "annotations"), "annotations raster has not been created"
        return self.annotations

    def get_tensor(self, labels_gdf: gpd.GeoDataFrame) -> np.ndarray:
        assert hasattr(self, "tensor"), "tensor has not been created"
        return self.tensor

    def get_metadata_row(self) -> dict:
        assert hasattr(self, "annotations"), "annotation raster has not been created"
        metadata_row = {
            "id": self.id,
            "tile_name": self.tile_name,
            "patch_n": self.patch_n,
            "parcel_cover": (self.annotations > 0).sum() / self.annotations.size,
            "dates_S2": {
                i: pd.to_datetime(date).strftime("%Y%m%d")
                for i, date in enumerate(self.data.dates)
            },
            "process_num_S2": {
                i: processn
                for i, processn in enumerate(self.data.process_nums)
            },
            "geometry": self.data.bounds,
        }
        return metadata_row

    def save_tensor(self, path: Path):
        assert hasattr(self, "tensor"), "tensor has not been created"
        np.save(path, self.tensor,)

    def save_annotations(self, path: Path):
        assert hasattr(self, "annotations"), "annotation raster has not been created"
        np.save(path, self.annotations,)

    def create_tensor(
        self,
        products_paths: Path,
        patch_size: int = 256,
        padding: int = 1,
        black_patch_threshold: float = 0.2,
    ):
        '''
        Recibe un iterable con los paths de todos los productos del tile y el número de patch deseado.
        '''
        products_array = []
        dates = []
        processnums = []

        #Se ordenan los path de productos por fecha
        sorted_paths = (
                pd.DataFrame(
                    [(ProductPath(path).get_date(),  path) for path in products_paths],
                    columns=["date", "path"]
                    )
                .sort_values(by="date")
                .path
                )

        first_iter_of_patch = True
        for path in sorted_paths:
            product_path = ProductPath(path)
            bands_array = []
            for band_path in product_path.get_band_paths():
                with rasterio.open(band_path) as src:
                    window = self.get_patch_window(
                            src, padding=padding, patch_size=patch_size,
                    )
                    band_raster = src.read(1, window=window)
                    if first_iter_of_patch:
                        self.data = RasterData(src, window)
                        first_iter_of_patch = False
                bands_array.append(band_raster)
            if len(bands_array) != 12:
                print(f"PRODUCTO NO TIENE LAS 12 BANDAS: {product_path} ")
                continue
            bands_stack = np.stack(bands_array, axis=0)  # (bands, N, N)

            black_patch_condition = (
                (bands_stack == 0).sum() / bands_stack.size
                < black_patch_threshold
            )
            if not black_patch_condition: continue

            products_array.append(bands_stack)
            dates.append(product_path.get_date())
            processnums.append(product_path.get_processn())

        tensor_final = np.stack(products_array, axis=0)
        self.data.set_dates(dates)
        self.data.set_process_nums(processnums)

        self.tensor = tensor_final
        return self

    def create_annotation_raster(self, labels_gdf: gpd.GeoDataFrame):
        assert hasattr(self, "tensor"), "tensor has not been created"

        tensor_bounds = self.data.bounds
        sel_parcels = labels_gdf[labels_gdf.intersects(tensor_bounds)]
        shapes = list(zip(sel_parcels.polygon, sel_parcels.crop_class))
        self.annotations = features.rasterize(
            shapes,
            out_shape=self.data.shape,
            fill=0,
            transform=self.data.transform,
            all_touched=False,  # Esto lo tengo que revisar bien
            dtype=None
        )
        return self

    def get_id(self):
        array_size = 10980
        tiles = [
            "31TBF",
            "29TNF",
            "30UXU",
            "32TPP",
            "32UMC",
            "29UNU",
            "33TVN",
            "31UFU",
            "35TMH",
            "32VNH",
        ]
        tile_map = {tile: i for i, tile in enumerate(tiles)}
        patchesxtile = (array_size//256 + 1)**2
        id = tile_map[self.tile_name] * 10**4 + self.patch_n

        assert self.patch_n < patchesxtile, "número de patch no válido"
        assert self.tile_name in tiles, "tile no válido"
        return id

    def patch_coors(self, n: int, patch_size=256, array_size=10980, padding=1):
        '''
        Retorna las coordenadas del punto superior izquierdo del patch n-ésimo.
        Supone que todos los productos son de igual tamaño.
        '''
        patch_size1 = patch_size + padding
        lim = patch_size1 *(array_size // patch_size1 +1 )
        x = (n * patch_size1) % lim
        if ((n+1) * patch_size1) % lim == 0:# Condición último patch
            x = array_size - (patch_size)

        y= ((n * patch_size1) // lim) * patch_size1
        if y + patch_size > array_size:# Condición último patch
            y = array_size - (patch_size)
        return (x, y)

    def get_patch_window(
        self,
        raster_reader: rasterio.io.DatasetReader,
        patch_size=256, padding=1,
    ) -> rasterio.windows.Window:
        '''
        Retorna el patch n-ésimo a partir de un Dataset Reader de rasterio
        '''
        array_size = raster_reader.shape[0]

        x_patch, y_patch = self.patch_coors(self.patch_n, patch_size, array_size, padding)
        window = rasterio.windows.Window.from_slices(
                cols=slice(x_patch, x_patch + patch_size),
                rows=slice(y_patch, y_patch + patch_size),
            )
        return window

    def create_patch_data(
        self,
        raster_reader: rasterio.io.DatasetReader,
        patch_size=256, padding=1,
    ):
        '''
        Extrae bounds y transformation asociados al patch a partir del
        raster_reader de alguna de las bandas de un producto del tile completo.

        En el flujo habitual de procesamiento self.data se genera
        en self.create_patch_tensor(), pero para aplicaciones de visualización
        puede ser útil generar el patch data directamente.
        '''
        window = self.get_patch_window(
                raster_reader,  padding=padding, patch_size=patch_size,
        )
        self.data = RasterData(raster_reader, window)
        return self
