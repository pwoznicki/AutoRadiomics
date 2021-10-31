import monai.data.Dataset as MONAIDataset
from monai.apps import download_url, download_and_extract
from monai.config import print_config
from monai.data import DataLoader, Dataset, CacheDataset
from monai.losses import BendingEnergyLoss, MultiScaleLoss, DiceLoss
from monai.metrics import DiceMetric
from monai.networks.blocks import Warp
from monai.networks.nets import LocalNet
from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    RandAffined,
    Resized,
    ScaleIntensityRanged,
    EnsureTyped,
)


class DistanceExtractor:
    def __init__(extraction_params):
        files

    def execute(mask1_path, mask2_path):
        files = [{"mask1": mask1_path, "mask2": mask2_path}]
