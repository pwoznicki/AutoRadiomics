from classrad.feature_extraction.extractor import FeatureExtractor


def test_get_feature_names():
    extractor = FeatureExtractor()
    feature_names = extractor.get_feature_names()
    return feature_names
