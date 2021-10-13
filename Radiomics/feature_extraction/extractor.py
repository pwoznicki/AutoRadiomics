
"""
Write a FeatureExtractor class on top of pyradiomics featureextractor, which will extract features, given paths from a pandas df.
"""

import pandas as pd
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
from radiomics import getTestCase
import pkgutil
import logging
import sys
from tqdm import tqdm

class FeatureExtractor:
    """
    Class to extract features from a set of images.
    """
    def __init__(self, 
            df,
            out_path,
            feature_set='pyradiomics',
            extraction_params='default_params.yaml',
            image_col='image_path',
            mask_col='mask_path',
            verbose=False):
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
        self.extraction_params = pkgutil.get_data(__name__, extraction_params)
        self.image_col = image_col
        self.mask_col = mask_col
        self.verbose = verbose
        self.feature_df = df.copy()
        self.extractor = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
        self.logger.info('FeatureExtractor initialized')

    def extract_features(self, feature_set=None, verbose=None):
        """
        Extract features from a set of images.
        """
        if feature_set is not None:
            self.feature_set = feature_set
        if verbose is not None:
            self.verbose = verbose

        # Get the feature extractor
        self.logger.info('Initializing feature extractor')
        self.get_feature_extractor()

        # Get the feature names
        #self.logger.info('Getting feature names')
        #feature_names = self.get_feature_names(extractor)

        # Get the feature values
        self.logger.info('Extracting features')
        self.get_features()

        # Write the features to a csv file
        #self.logger.info('Writing features to csv file')
        #self.write_features_to_csv(features, output_path)

    def get_feature_extractor(self):
        """
        Get the feature extractor.
        """
        if self.feature_set == 'pyradiomics':
            self.extractor = featureextractor.RadiomicsFeatureExtractor(
                    self.extraction_params
            )
        else:
            raise ValueError('Feature set not supported')
        return self

    def get_features(self):
        """
        Get the feature values.
        """
        features = []
        for index, row in tqdm(list(self.df.iterrows())):
            image_path = row[self.image_col]
            mask_path = row[self.mask_col]
            feature_vector = self.extractor.execute(image_path, mask_path)
            vector_vals = list(feature_vector.values())
            features.append(vector_vals)
        colnames = list(feature_vector.keys())
        #features = np.swapaxes(np.array(features), 0, 1).tolist()
        self.feature_df[colnames] = pd.DataFrame(features, index=self.feature_df.index)
        self.feature_df.to_csv(self.out_path, index=False)

    # def write_features_to_csv(self, feature_values, feature_names, output_path):
    #     """
    #     Write the features to a csv file.
    #     """
    #     features.to_csv(output_path)