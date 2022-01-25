import pandas as pd
from monai.apps.datasets import MedNISTDataset, DecathlonDataset
from classrad.data.dataset import ImageDataset


def import_mednist_dataset(mednist_dataset: MedNISTDataset):
    """
    Convert the MONAI MedNISTDataset into a
    classrad.data.dataset.ImageDataset.
    """
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
    return ImageDataset().from_dataframe(
        dataframe=df, image_colname="image_path", mask_colname="mask_path"
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
    return ImageDataset().from_dataframe(
        df=df, image_colname="image_path", mask_colname="mask_path"
    )
