from __future__ import annotations

from pathlib import Path

import git
import pandas as pd
from monai.apps.datasets import DecathlonDataset, MedNISTDataset

from classrad.config import config
from classrad.config.type_definitions import PathLike
from classrad.data.dataset import ImageDataset


def load_mednist_dataset(root_dir: PathLike) -> ImageDataset:
    """
    Downlad, load MONAI MedNISTDataset
    and convert it to ImageDataset.
    """
    mednist_dataset = MedNISTDataset(
        root_dir=root_dir, section="training", download=True
    )
    data_list = mednist_dataset.data
    image_paths = [elem["image"] for elem in data_list]
    labels = [elem["label"] for elem in data_list]
    class_names = [elem["class_name"] for elem in data_list]
    ids = range(len(image_paths))
    df = pd.DataFrame(
        {
            "id": ids,
            "image_path": image_paths,
            "label": labels,
            "class_name": class_names,
        }
    )
    return ImageDataset(
        df=df, image_colname="image_path", mask_colname="mask_path"
    )


def convert_decathlon_dataset(decathlon_dataset: DecathlonDataset):
    """
    Convert the MONAI DecathlonDataset into a
    classrad.data.dataset.ImageDataset.
    """
    data_list = decathlon_dataset.data
    image_paths = [elem["image"] for elem in data_list]
    mask_paths = [elem["label"] for elem in data_list]
    ids = range(len(image_paths))
    df = pd.DataFrame(
        {
            "id": ids,
            "image_path": image_paths,
            "mask_path": mask_paths,
        }
    )
    return ImageDataset(
        df=df, image_colname="image_path", mask_colname="mask_path"
    )


def load_prostatex(root_dir: PathLike | None = None) -> ImageDataset:
    if root_dir is None:
        root_dir = Path(config.TEST_DATA_DIR) / "datasets" / "PROSTATEx"
    if not Path(root_dir).exists() or Path(root_dir).is_file():
        git.Repo.clone_from(
            "https://github.com/rcuocolo/PROSTATEx_masks/", root_dir
        )


if __name__ == "__main__":
    load_prostatex()
