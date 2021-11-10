from Radiomics.feature_extraction.extractor import FeatureExtractor
from multiprocessing import Pool
import pandas as pd
from Radiomics.utils.utils import time_it


class StreamlitFeatureExtractor(FeatureExtractor):
    @time_it
    def extract_features(self, feature_set=None, verbose=None, progressbar=None):
        """
        Extract features from a set of images.
        """
        if feature_set is not None:
            self.feature_set = feature_set
        if verbose is not None:
            self.verbose = verbose

        # Get the feature extractor
        self.logger.info("Initializing feature extractor")
        self.get_feature_extractor()

        self.logger.info("Initializing feature dataframe")
        self.initialize_feature_df()

        # Get the feature values
        self.logger.info("Extracting features")
        if self.num_threads is not None:
            self.get_features_parallel(progressbar=progressbar)
        else:
            self.get_features(progressbar=progressbar)
        self.save_feature_df()
        return self.feature_df

    def get_features(self, progressbar=None):
        """
        Get the feature values
        """
        rows = list(self.df.iterrows())
        for i, (index, row) in enumerate(rows):
            feature_series = self.add_features_for_single_case(row)
            self.feature_df = self.feature_df.append(feature_series, ignore_index=True)
            if progressbar:
                fraction_complete = (i + 1) / len(rows)
                progressbar.progress(fraction_complete)
        return self

    def get_features_parallel(self, progressbar=None):
        try:
            _, df_rows = zip(*self.df.iterrows())
            n = len(self.df)
            p = Pool(self.num_threads)
            if progressbar:
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
