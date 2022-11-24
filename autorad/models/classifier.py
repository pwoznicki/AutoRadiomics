import mlflow
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

from autorad.config import config
from autorad.training import optuna_params


class MLClassifier(ClassifierMixin):
    """
    Class to provide a homogenous interface for all models.
    """

    def __init__(
        self,
        model,
        name: str,
        params: dict = {},
    ):
        self.model = model
        self.name = name
        self.params = params
        self._param_fn = None
        self.available_models = [
            "Random Forest",
            "AdaBoost",
            "Logistic Regression",
            "SVM",
            "Gaussian Process Classifier",
            "XGBoost",
        ]

    def __repr__(self):
        return f"{self.name}"

    @classmethod
    def from_sklearn(cls, name: str, params: dict = {}):
        if name == "Random Forest":
            model = RandomForestClassifier(**params)
        elif name == "AdaBoost":
            model = AdaBoostClassifier(**params)
        elif name == "Logistic Regression":
            model = LogisticRegression(
                max_iter=1000,
                **params,
            )
        elif name == "SVM":
            model = SVC(
                probability=True,
                max_iter=1000,
                **params,
            )
        elif name == "Gaussian Process Classifier":
            model = GaussianProcessClassifier(**params)
        elif name == "XGBoost":
            model = XGBClassifier(
                verbosity=0,
                silent=True,
                use_label_encoder=False,
                **params,
            )
        else:
            raise ValueError("Classifier name not recognized.")

        return cls(model, name, params)

    @classmethod
    def from_keras(cls, model, name, **params):
        """
        Args:
            keras_model (scikeras.KerasClassifier):
                keras model, wrapped for sklearn
            name (str): name, used to reference the model
        """
        return cls(model, name, **params)

    @classmethod
    def initialize_default_sklearn_models(cls):
        """
        Initialize a list of all available models.
        """
        models = []
        for model_name in config.AVAILABLE_CLASSIFIERS:
            model = cls.from_sklearn(model_name)
            models.append(model)
        return models

    def fit(self, X, y, **params):
        self.model.fit(X, y, **params)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X, **params):
        return self.model.predict_proba(X, **params)

    def predict_proba_binary(self, X, **params):
        return self.model.predict_proba(X, **params)[:, 1]

    def predict_label_and_proba(self, X):
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba_binary(X)

        return y_pred, y_pred_proba

    def score(self, X, y):
        return self.model.score(X, y)

    def get_params(self, deep):
        return self.model.get_params(deep)

    def set_params(self, **params):
        if self.name == "Logistic Regression":
            if params["penalty"] == "l1":
                params["solver"] = "saga"
        self.model.set_params(**params)
        self.params = params
        return self

    @property
    def param_fn(self):
        if self._param_fn is None:
            self._param_fn = optuna_params.get_param_fn(self.name)
        return self._param_fn

    # @param_fn.setter
    # def param_fn(self, param_fn: Optional[Callable] = None):
    #     if param_fn is None:
    #     self._param_fn = param_fn

    def feature_importance(self):
        if self.name == "Logistic Regression":
            importance = self.model.coef_[0]
        elif self.name in ["AdaBoost", "Random Forest", "XGBoost"]:
            importance = self.model.feature_importances_
        else:
            raise ValueError(
                f"For model {self.name} feature \
                               importance could not be calculated."
            )
        return importance

    def save_to_mlflow(self):
        if self.model == "XGBoost":
            mlflow.xgboost.log_model(self.model, "model")
        else:
            try:
                mlflow.sklearn.log_model(self.model, "model")
            except Exception:
                print("Could not save model to mlflow.")

    @classmethod
    def from_mlflow(cls, model_uri):
        try:
            model = mlflow.sklearn.load_model(model_uri)
        except Exception:
            try:
                model = mlflow.xgboost.load_model(model_uri)
            except Exception:
                print("Could not load model from mlflow.")
                return None
        return cls(model, name=model_uri.split("/")[-1])


class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Ensemble model for MLClassifiers.

    Parameters:
    clf : `iterable`
      A list of model objects.
    weights : `list` (default: `None`)
      If `None`, the majority rule voting will be applied to the predicted
        class labels. If a list of weights (`float` or `int`) is provided,
        the averaged raw probabilities (via `predict_proba`) will be used
        to determine the most confident class label.
    """

    def __init__(self, model_list, weights=None):
        self.model_list = model_list
        self.weights = weights

    def fit(self, X, y, **fit_params):
        """
        Fit the scikit-learn estimators.

        Parameters:
        X : numpy array, shape = [n_samples, n_features]
            Training data
        y : list or numpy array, shape = [n_samples]
            Class labels
        """
        for clf in self.model_list:
            clf.fit(X, y, **fit_params)

    def predict(self, X):
        """
        Parameters:
        X : numpy array, shape = [n_samples, n_features]

        Returns:
        maj : list or numpy array, shape = [n_samples]
            Predicted class labels by majority rule
        """
        self.classes_ = np.asarray([clf.predict(X) for clf in self.model_list])
        # if self.weights:
        #    avg = self.predict_proba(X)
        #         maj = np.apply_along_axis(lambda x: max(enumerate(x),
        #               key=operator.itemgetter(1))[0], axis=1, arr=avg)
        # else:
        #     maj = np.asarray(
        #         [
        #             np.argmax(np.bincount(self.classes_[:, c]))
        #             for c in range(self.classes_.shape[1])
        #         ]
        #     )

        # return maj

    def predict_proba(self, X):
        """
        Parameters:
        X : numpy array, shape = [n_samples, n_features]

        Returns:
        avg : list or numpy array, shape = [n_samples, n_probabilities]
            Weighted average probability for each class per sample.
        """
        self.probas_ = [clf.predict_proba(X) for clf in self.model_list]
        avg = np.average(self.probas_, axis=0, weights=self.weights)

        return avg

    def get_model_list(self):
        return self.model_list
