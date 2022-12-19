import logging

from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE

from autorad.config import config

log = logging.getLogger(__name__)


def create_oversampling_model(method: str, random_state: int = config.SEED):
    if method is None:
        return None
    if method == "ADASYN":
        return ADASYN(random_state=random_state)
    elif method == "SMOTE":
        return SMOTE(random_state=random_state)
    elif method == "BorderlineSMOTE":
        return BorderlineSMOTE(random_state=random_state, kind="borderline1")
    raise ValueError(f"Unknown oversampling method: {method}")


class OversamplerWrapper:
    def __init__(self, oversampler, random_state=config.SEED):
        self.oversampler = oversampler
        self.oversampler.__init__(random_state=random_state)

    def fit(self, X, y):
        return self.oversampler.fit(X, y)

    def fit_transform(self, X, y):
        return self.oversampler.fit_resample(X, y)

    def transform(self, X):
        log.debug(f"{self.oversampler} does nothing on .transform()...")
        return X
