import logging
import os

import pandas as pd

from autorad.config.type_definitions import PathLike

log = logging.getLogger(__name__)


def get_paths_with_separate_folder_per_case(
    data_dir: PathLike,
    image_stem: str = "image",
    mask_stem: str = "segmentation",
    relative: bool = False,
) -> pd.DataFrame:
    ids, images, masks = [], [], []
    for id_ in os.listdir(data_dir):
        id_dir = os.path.join(data_dir, id_)
        if not os.path.isdir(id_dir):
            continue
        image_path = os.path.join(id_dir, f"{image_stem}.nii.gz")
        mask_path = os.path.join(id_dir, f"{mask_stem}.nii.gz")
        if not os.path.exists(image_path):
            log.error(f"Image for ID={id_} does not exist ({image_path})")
            continue
        if not os.path.exists(mask_path):
            log.error(f"Mask for ID={id_} does not exist ({mask_path})")
            continue
        if relative:
            image_path = os.path.relpath(image_path, data_dir)
            mask_path = os.path.relpath(mask_path, data_dir)

        ids.append(id_)
        images.append(image_path)
        masks.append(mask_path)
    return pd.DataFrame(
        {"ID": ids, "image_path": images, "segmentation_path": masks}
    )
