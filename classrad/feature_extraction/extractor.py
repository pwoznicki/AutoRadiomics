"""
Write a FeatureExtractor class on top of pyradiomics featureextractor,
which will extract features, given paths from a pandas df.
"""

import logging
import sys
from multiprocessing import Pool

import pandas as pd
from radiomics import featureextractor
from classrad.utils.utils import time_it
from tqdm import tqdm


class FeatureExtractor:
    """
    Class to extract features from a set of images.
    """

    def __init__(
        self,
        df,
        out_path,
        feature_set="pyradiomics",
        extraction_params=None,
        image_col="image_path",
        mask_col="mask_path",
        verbose=False,
        num_threads=None,
    ):
        """
        Initialize the Feature Extractor.
        :param df: pandas df with columns 'image_path' and 'mask_path'
        :param out_path: path to save the output csv
        :param extractor_params: path to the pyradiomics extractor params
        :param image_col: name of df column containing paths to images
        :param mask_col: name of df column containing paths to masks
        :param verbose: whether to print progress
        """
        self.df = df
        self.out_path = out_path
        self.feature_set = feature_set
        self.extraction_params = extraction_params
        self.image_col = image_col
        self.mask_col = mask_col
        self.verbose = verbose
        self.num_threads = num_threads
        self.feature_df = None
        self.extractor = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
        self.logger.info("FeatureExtractor initialized")

    def extract_features(self, feature_set=None, verbose=None):
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
            self.get_features_parallel()
        else:
            self.get_features()
        self.save_feature_df()

    def get_feature_extractor(self):
        """
        Get the feature extractor.
        """
        if self.feature_set == "pyradiomics":
            self.extractor = featureextractor.RadiomicsFeatureExtractor(
                self.extraction_params
            )
        else:
            raise ValueError("Feature set not supported")
        return self

    def add_features_for_single_case(self, case):
        """
        Run extraction for one case and append results to feature_df
        Args:
            case: pd.Series describing a row of df
        """
        image_path = case[self.image_col]
        mask_path = case[self.mask_col]
        feature_vector = self.extractor.execute(image_path, mask_path)
        feature_series = pd.concat([case, pd.Series(feature_vector)])
        return feature_series

    def get_feature_names(self, case):
        image_path = case[self.image_col]
        mask_path = case[self.mask_col]
        feature_vector = self.extractor.execute(image_path, mask_path)
        feature_names = feature_vector.keys()
        return feature_names

    def initialize_feature_df(self):
        if self.feature_df is None:
            first_df_row = self.df.iloc[0]
            feature_names = self.get_feature_names(first_df_row)
            self.feature_df = self.df.copy()
            self.feature_df = self.feature_df.reindex(columns=feature_names)
        else:
            self.logger.info("Dataframe already has content!")
        return self

    @time_it
    def get_features(self):
        """
        Get the feature values.
        """
        rows = self.df.iterrows()
        for index, row in tqdm(rows):
            feature_series = self.add_features_for_single_case(row)
            self.feature_df = self.feature_df.append(feature_series, ignore_index=True)
        return self

    def save_feature_df(self):
        self.feature_df.to_csv(self.out_path, index=False)

    @time_it
    def get_features_parallel(self):
        try:
            _, df_rows = zip(*self.df.iterrows())
            p = Pool(self.num_threads)
            results = p.map(self.add_features_for_single_case, df_rows)
            self.feature_df = pd.concat(results, axis=1).T
        except Exception:
            print("Multiprocessing failed! :/")
        return self
