from pathlib import Path

import numpy as np
from monai.data import DataLoader, Dataset
from monai.transforms import AddChanneld, Compose, LoadImaged, Spacingd
from scipy.ndimage.measurements import center_of_mass


class DistanceExtractor:
    def __init__(self, df, out_path, mask1_colname, mask2_colname):
        self.df = df
        self.out_path = out_path
        self.mask1_colname = mask1_colname
        self.mask2_colname = mask2_colname
        self.ds = None
        self.loader = None

    def calculate_distance(self, arr1, arr2):
        """
        Calculates Euclidean distance between centers of mass from two binary
        arrays.
        """
        center1 = np.array(center_of_mass(arr1))
        center2 = np.array(center_of_mass(arr2))
        dist = np.linalg.norm(center1 - center2)
        return dist

    def prepare_data(self):
        files = []
        self.df[self.mask1_colname].fillna("no path", inplace=True)
        self.df[self.mask2_colname].fillna("no path", inplace=True)

        for idx, row in self.df.iterrows():
            mask1_path = row[self.mask1_colname]
            mask2_path = row[self.mask2_colname]
            if not Path(mask1_path).exists():
                print(f"Path does not exist [{mask1_path}]")
            elif not Path(mask2_path).exists():
                print(f"Path does not exist [{mask2_path}]")
            else:
                files.append(
                    {"mask1": mask1_path, "mask2": mask2_path, "index": idx}
                )
        transforms = Compose(
            LoadImaged(keys=["mask1", "mask2"]),
            AddChanneld(keys=["mask1", "mask2"]),
            Spacingd(keys=["mask1", "mask2"], pixdim=[1, 1, 1], diagonal=True),
        )
        self.ds = Dataset(data=files, transform=transforms)
        self.loader = DataLoader(self.ds, batch_size=1)

    def run_calculation(self):
        self.df["original_distance_from_primary"] = -100
        for data in self.loader:
            arr1 = data["mask1"].numpy()[0]
            arr2 = data["mask2"].numpy()[0]
            idx = data["index"].numpy()[0]
            print(idx)

            dist = self.calculate_distance(arr1, arr2)
            self.df.loc[idx, "original_distance_from_primary"] = dist
        return self

    def save_df(self):
        self.df.to_csv(self.out_path)

    def execute(self):
        self.prepare_data()
        self.run_calculation()
        self.save_df()
