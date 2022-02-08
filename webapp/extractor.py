from multiprocessing import Pool
import pandas as pd
from classrad.feature_extraction.extractor import FeatureExtractor
from classrad.utils.utils import time_it


class StreamlitFeatureExtractor(FeatureExtractor):
    @time_it
    def extract_features(
        self, feature_set=None, verbose=None, progressbar=None
    ):
        # Get the feature extractor
        self.logger.info("Initializing feature extractor")
        self.initialize_extractor()

        self.logger.info("Initializing feature dataframe")
        self.initialize_feature_df()

        # Get the feature values
        self.logger.info("Extracting features")
        if self.num_threads > 1:
            self.get_features_parallel(progressbar=progressbar)
        else:
            self.get_features(progressbar=progressbar)
        self.save_feature_df()

    def get_features(self, progressbar=None):
        """
        Get the feature values
        """
        rows = list(self.dataset.df.iterrows())
        for i, (index, row) in enumerate(rows):
            id_ = row["lesion_ID"]
            print(f"Lesion ID = {id_}")
            feature_series = self.add_features_for_single_case(row)
            self.feature_df = self.feature_df.append(
                feature_series, ignore_index=True
            )
            if progressbar is not None:
                fraction_complete = (i + 1) / len(rows)
                progressbar.progress(fraction_complete)
        return self

    def get_features_parallel(self, progressbar=None):
        try:
            _, df_rows = zip(*self.dataset.df.iterrows())
            n = len(self.dataset.df)
            p = Pool(self.num_threads)
            if progressbar is not None:
                results = []
                for i, result in enumerate(
                    p.imap(self.add_features_for_single_case, df_rows)
                ):
                    results.append(result)
                    fraction_complete = (i + 1) / n
                    progressbar.progress(fraction_complete)
            else:
                results = p.map(self.add_features_for_single_case, df_rows)
            self.feature_df = pd.concat(results, axis=1).T
        except Exception:
            print("Multiprocessing failed! :/")
        return self
