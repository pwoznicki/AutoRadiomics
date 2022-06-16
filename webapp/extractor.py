import logging
from multiprocessing import Pool

import pandas as pd

from autorad.feature_extraction.extractor import FeatureExtractor
from autorad.utils.utils import time_it

log = logging.getLogger(__name__)


class StreamlitFeatureExtractor(FeatureExtractor):
    def __init__(
        self,
        dataset,
        feature_set="pyradiomics",
        extraction_params="Baessler_CT.yaml",
        n_jobs=None,
        progressbar=None,
    ):
        self.progressbar = progressbar
        super().__init__(dataset, feature_set, extraction_params, n_jobs)

    @time_it
    def get_features(self) -> pd.DataFrame:
        """
        Run extraction for all cases.
        """
        image_paths = self.dataset.image_paths
        mask_paths = self.dataset.mask_paths
        lst_of_feature_dicts = []
        for i, (image_path, mask_path) in enumerate(
            zip(image_paths, mask_paths)
        ):
            feature_dict = self.get_features_for_single_case(
                image_path, mask_path
            )
            lst_of_feature_dicts.append(feature_dict)
            fraction_complete = (i + 1) / len(image_paths)
            if self.progressbar:
                self.progressbar.progress(fraction_complete)

        feature_df = pd.DataFrame(lst_of_feature_dicts)
        return feature_df

    @time_it
    def get_features_parallel(self) -> pd.DataFrame:
        image_paths = self.dataset.image_paths
        mask_paths = self.dataset.mask_paths
        n = len(image_paths)
        p = Pool(self.n_jobs)
        lst_of_feature_dicts = []
        try:
            if self.progressbar:
                for i, result in enumerate(
                    p.imap(
                        self.get_features_for_single_case,
                        zip(image_paths, mask_paths),
                    )
                ):
                    lst_of_feature_dicts.append(result)
                    fraction_complete = (i + 1) / n
                    self.progressbar.progress(fraction_complete)
            else:
                lst_of_feature_dicts = p.map(
                    self.get_features_for_single_case,
                    zip(image_paths, mask_paths),
                )
        except Exception:
            raise RuntimeError("Multiprocessing failed! :/")
        feature_df = pd.DataFrame(lst_of_feature_dicts)
        return feature_df
