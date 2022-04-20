import logging
from multiprocessing import Pool

import pandas as pd

from classrad.feature_extraction.extractor import FeatureExtractor
from classrad.utils.utils import time_it

log = logging.getLogger(__name__)


class StreamlitFeatureExtractor(FeatureExtractor):
    def __init__(
        self,
        dataset,
        out_path,
        feature_set="pyradiomics",
        extraction_params="Baessler_CT.yaml",
        verbose=False,
    ):
        super().__init__(
            dataset, out_path, feature_set, extraction_params, verbose
        )

    @time_it
    def extract_features(self, num_threads=1, progressbar=None):
        # Get the feature extractor
        self._initialize_extractor()
        log.info("StreamlitFeatureExtractor initialized")

        # Get the feature values
        log.info("Extracting features")
        if num_threads > 1:
            feature_df = self.get_features_parallel(
                num_threads=num_threads, progressbar=progressbar
            )
        else:
            feature_df = self.get_features(progressbar=progressbar)
        feature_df.to_csv(self.out_path, index=False)
        return feature_df

    def get_features(self, progressbar=None):
        """
        Get the feature values
        """
        feature_df_rows = []
        df = self.dataset.get_df()
        rows = list(df.iterrows())
        for i, (_, row) in enumerate(rows):
            feature_series = self._get_features_for_single_case(row)
            if feature_series is not None:
                feature_df_rows.append(feature_series)
            if progressbar:
                fraction_complete = (i + 1) / len(rows)
                progressbar.progress(fraction_complete)
        feature_df = pd.concat(feature_df_rows, axis=1).T
        return feature_df

    def get_features_parallel(self, num_threads, progressbar=None):
        df = self.dataset.get_df()
        try:
            _, df_rows = zip(*df.iterrows())
            n = len(self.dataset.df)
            p = Pool(num_threads)
            if progressbar:
                results = []
                for i, result in enumerate(
                    p.imap(self._get_features_for_single_case, df_rows)
                ):
                    results.append(result)
                    fraction_complete = (i + 1) / n
                    progressbar.progress(fraction_complete)
            else:
                results = p.map(self._get_features_for_single_case, df_rows)
            feature_df = pd.concat(results, axis=1).T
        except Exception:
            print("Multiprocessing failed! :/")
        return feature_df
