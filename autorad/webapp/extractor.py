from __future__ import annotations

import logging
from multiprocessing import Pool

import pandas as pd
import streamlit as st

from autorad.config.type_definitions import PathLike
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
    ):
        super().__init__(dataset, feature_set, extraction_params, n_jobs)

    @time_it
    def get_features(self):
        """
        Run extraction for all cases.
        """
        progressbar = st.progress(0)
        lst_of_feature_dicts = []
        for i, (image_path, mask_path) in enumerate(
            zip(self.dataset.image_paths, self.dataset.mask_paths)
        ):
            feature_dict = self.get_features_for_single_case(
                image_path, mask_path
            )
            lst_of_feature_dicts.append(feature_dict)
            fraction_complete = (i + 1) / len(self.dataset.image_paths)
            progressbar.progress(fraction_complete)
        feature_df = pd.DataFrame(lst_of_feature_dicts)
        return feature_df

    def get_features_for_single_case_from_zipped_paths(
        self, paths: tuple[PathLike, PathLike]
    ) -> dict | None:
        image_path, mask_path = paths
        return super().get_features_for_single_case(image_path, mask_path)

    @time_it
    def get_features_parallel(self):
        mask_paths = [path_ for path_ in self.dataset.mask_paths]
        image_paths = [path_ for path_ in self.dataset.image_paths]

        progressbar = st.progress(0)
        n = len(image_paths)
        p = Pool(self.n_jobs)
        lst_of_feature_dicts = []
        for i, result in enumerate(
            p.imap(
                self.get_features_for_single_case_from_zipped_paths,
                zip(image_paths, mask_paths),
            )
        ):
            lst_of_feature_dicts.append(result)
            fraction_complete = (i + 1) / n
            progressbar.progress(fraction_complete)
        feature_df = pd.DataFrame(lst_of_feature_dicts)
        return feature_df
