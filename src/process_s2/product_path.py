import pandas as pd
from pathlib import Path
import numpy as np


class ProductPath:
    def __init__(self, product_path: Path):
        self.product_path = product_path

    def get_band_paths(self):
        '''
        Obtiene una lista con los paths de las bandas a partir del path de producto.
        '''
        return np.sort(list(self.product_path.glob(
            f"*_*_B*"
            )))

    def get_band_paths_and_codes(self):
        '''
        Obtiene una lista con los pares (path, codes) de las bandas a partir del
        path de producto.

        Es útil para la creación alternativa de los tensores con xarray.
        '''
        paths =  np.sort(list(self.product_path.glob(
            f"*_*_B*"
            )))
        bands = [str(path).split("/")[-1].split("_")[-2] for path in paths]
        return zip(paths, bands)

    def get_date(self):
        '''
        Entrega el datetime asociado a un producto Sentinel-2 a partir del path a su directorio.
        '''
        return pd.to_datetime(str(self.product_path).split("/")[-1].split("_")[2][:8])

    def get_processn(self):
        '''
        Entrega el número de baseline processing number asociado a un producto 
        Sentinel-2 a partir del path a su directorio.
        '''
        return str(self.product_path).split("/")[-1].split("_")[3]

    def get_rgb_band_paths(self):
        '''
        Obtiene una lista con los paths de las bandas RGB a partir del path de producto.

        Útil para visualizaciones
        '''
        return np.sort(list(self.product_path.glob(
            f"*_*_B0[234]*"
            )))

    def get_rgb_band_paths_and_codes(self):
        '''
        Obtiene una lista con los pares (path, codes) de las bandas a partir del
        path de producto.

        Es útil para la creación alternativa de los tensores con xarray.
        '''
        paths =  np.sort(list(self.product_path.glob(
            f"*_*_B0[234]*"
            )))
        bands = [str(path).split("/")[-1].split("_")[-2] for path in paths]
        return zip(paths, bands)
