import os
from dotenv import load_dotenv
from pathlib import Path
import time
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
import json

from patch import Patch
from product_path import ProductPath


class RequiredPaths:
    def __init__(
        self,
        in_s2: Path,  # dirección de los productos a procesar
        in_labels: Path,  # dirección del archivo .gpkg con las parcelas
        in_class_mapping: Path,  # dirección del csv con las clases de cultivos
        out_metadata: Path,  # dirección de la metadata producida en el procesamiento.
        out_s2: Path,  # dirección de los tensores de imágenes 4D producidos.
        out_annotations: Path,  # dirección de las matrices 2D con los labels producidos.
    ):
        self.in_s2 = in_s2
        self.in_labels = in_labels
        self.in_class_mapping = in_class_mapping
        self.out_metadata = out_metadata
        self.out_s2 = out_s2
        self.out_annotations = out_annotations
        assert in_labels.exists(), f"No existe archivo con labels {in_labels}"
        assert in_class_mapping.exists(), f"No existe diccionario de clases de cultivo {in_class_mapping}"
        if not out_s2.exists(): os.makedirs(out_s2)
        if not out_annotations.exists(): os.makedirs(out_annotations)


def get_crs(products_paths):
    '''
    Loopea en todos los productos buscando crs.
    Se asegura de que el crs exista y sea consistente.

    Según lo explorado no todas los productos tienen crs, pero basta con que 
    alguno lo tenga y que los que tengan, tengan el mismo.
    '''
    crs_arr = []
    for path in products_paths:
        product_path = ProductPath(path)
        for band_path in product_path.get_band_paths():
            crs = rasterio.open(band_path).crs.to_epsg()
            if crs is not None:
                crs_arr.append(crs)
    crs_arr = np.array(crs_arr)
    assert len(crs_arr) > 0, "No se encontró crs."
    assert (crs_arr == crs_arr[0]).all(), "crs no es consistente."
    return f"EPSG:{crs_arr[0]}"


def update_metadata_file(new_rows, path, crs):
    '''
    Actualiza el archivo de metadata existente, si no existe crea uno.
    '''
    metadata_gdf = (
        gpd.GeoDataFrame(new_rows, geometry="geometry", crs=crs)
        .set_index("id")
    )
    if path.exists():
        old_metadata_gdf = (
            gpd.read_file(path).set_crs(crs, allow_override=True)
            .astype({"id": int})
            .set_index("id")
        )
        metadata_gdf = pd.concat([
            metadata_gdf,
            old_metadata_gdf,
            ]).reset_index().drop_duplicates(subset="id").set_index("id")
    with open(path, "w") as text_file:
        text_file.write(metadata_gdf.to_json())


def process_missing_patches(
    tile_name: str,
    patches_list: list,
    paths: RequiredPaths,
    patch_size: int,
    grid_padding: int,
    verbose: bool = False,
):
    '''
    Main body of the processing script.
    It generates all the processing for a given tile given
    all the necessary input, and output paths.
    '''
    start = time.time()
    print(f"Formateando tile {tile_name}...")

    if verbose: print(f"\tReconociendo crs...")
    sentinel_crs = get_crs([
        p for p in paths.in_s2.rglob(f"S2?_MSIL2A_*{tile_name}*")
        if p.is_dir() and p.parent.parent.name == tile_name
    ])

    # Parcelas en tile
    class_mapping = ( # definición mapeo hcat4_code -> crop label class
        pd.read_csv(paths.in_class_mapping, index_col=0)
        .iloc[:, 0]
        .to_dict()
    )
    labels_gdf = (
        gpd.read_file(
            paths.in_labels,
            where=f"name='{tile_name}'",
        )
        .to_crs(sentinel_crs)
        .assign(polygon=lambda df: df.geometry.map(lambda x: x.geoms[0]))
        .assign(crop_class=lambda df: df.hcat4_code.map(class_mapping))
    )

    # Crear tensor con bandas y tiempo, en el patch n-ésimo
    count = 0
    metadata_rows = []
    array_size = 10980
    patch_size = 256
    final_n = (array_size//patch_size + 1)**2
    for patch_n in patches_list:
        patch = Patch(tile_name, patch_n)
        if verbose: print(f"\tFormateando patch {patch.patch_n} (id={patch.get_id()})...")

        patch.create_tensor(
            products_paths=[
                p for p in paths.in_s2.rglob(f"S2?_MSIL2A_*{tile_name}*")
                if p.is_dir() and p.parent.parent.name == tile_name
            ],
            patch_size=patch_size,
            padding=grid_padding,
        )
        patch.create_annotation_raster(labels_gdf)

        metadata_rows.append(
            patch.get_metadata_row()
        )
        # Guardar metadata cada 5 patches (por si hay detención forzosa)
        count = (count + 1) % 5
        if count == 0:
            if verbose: print("Tiempo de ejecución acumulado: ", round((time.time() - start)/60, 2), "[m]")
            # Almacenar metadata
            update_metadata_file(
                new_rows=metadata_rows,
                path=paths.out_metadata,
                crs=sentinel_crs)

    # Almacenar metadata faltante al terminar el tile
    update_metadata_file(
        new_rows=metadata_rows,
        path=paths.out_metadata,
        crs=sentinel_crs)

    end = time.time()
    print("The time of execution of above program is :",
          (end-start)/60, "m")


def main():
    load_dotenv()
    in_path = Path(os.getenv("INPUT_DATA_PATH")) # dirección del directorio con los datos a procesar.
    out_path = Path(os.getenv("OUTPUT_DATA_PATH")) # dirección del directorio donde se almacenarán los datos procesados siguiendo el formato de https://huggingface.co/datasets/IGNF/PASTIS-HD/tree/main.
    paths = RequiredPaths(
        in_s2 = in_path / "products",
        in_labels = in_path / "gsa_2022_selectedtiles.gpkg",
        in_class_mapping = in_path / "class_mapping.csv",
        out_metadata = out_path / "metadata.geojson", 
        out_s2 = out_path / "DATA_S2", 
        out_annotations = out_path / "ANNOTATIONS", 
    )

    with open("patches_faltantes.json", 'r') as file:
        missing_patches = json.load(file)

    for tile_name in missing_patches.keys():
        patches_list = [int(patch_n) for patch_n in missing_patches[tile_name]]
        process_missing_patches(
            tile_name=tile_name,
            patches_list=patches_list,
            paths=paths,
            patch_size=int(os.getenv("PATCH_SIZE")),
            grid_padding=int(os.getenv("GRID_PADDING")),
            verbose=bool(os.getenv("VERBOSE")),
        )


if __name__ == "__main__":
    main()
