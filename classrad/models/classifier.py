import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

from classrad.training.optimizer import OptunaOptimizer


class MLClassifier(ClassifierMixin):
    """
    Class to provide a homogenous interface for all classifiers.
    """

    def __init__(
        self,
        classifier,
        name: str = None,
        params: dict = {},
    ):
        self.classifier = classifier
        self.name = name
        self.params = params
        self.available_classifiers = [
            "Random Forest",
            "AdaBoost",
            "Logistic Regression",
            "SVM",
            "Gaussian Process Classifier",
            "XGBoost",
        ]
        self.optimizer = None

    @classmethod
    def from_sklearn(cls, name: str, params: dict = {}):
        if name == "Random Forest":
            classifier = RandomForestClassifier(**params)
        elif name == "AdaBoost":
            classifier = AdaBoostClassifier(**params)
        elif name == "Logistic Regression":
            classifier = LogisticRegression(
                max_iter=1000,
                **params,
            )
        elif name == "SVM":
            classifier = SVC(
                probability=True,
                max_iter=1000,
                **params,
            )
        elif name == "Gaussian Process Classifier":
            classifier = GaussianProcessClassifier(**params)
        elif name == "XGBoost":
            classifier = XGBClassifier(
                verbosity=0,
                silent=True,
                use_label_encoder=False,
                **params,
            )
        else:
            raise ValueError("Classifier name not recognized")

        return cls(classifier, name, params)

    @classmethod
    def from_keras(cls, classifier, name, **params):
        """
        Args:
            keras_classifier (scikeras.KerasClassifier):
                keras model, wrapped for sklearn
            name (str): name, used to reference the model
        """
        return cls(classifier, name, **params)

    def set_optimizer(self, optimizer: str, param_fn=None, n_trials=100):
        if optimizer == "optuna":
            self.optimizer = OptunaOptimizer(
                self, param_fn=param_fn, n_trials=n_trials
            )
        elif optimizer == "gridsearch":
            pass
            # self.optimizer = GridSearchOptimizer()
        else:
            raise ValueError("Optimizer not recognized")

    def fit(self, X, y, **params):
        if self.classifier is None:
            raise AssertionError("Run .select_classifier first!")
        else:
            self.classifier.fit(X, y, **params)

    def predict(self, X):
        return self.classifier.predict(X)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    def predict_proba_binary(self, X):
        return self.classifier.predict_proba(X)[:, 1]

    def predict_label_and_proba(self, X):
        y_pred = self.classifier.predict(X)
        y_pred_proba = self.classifier.predict_proba_binary(X)

        return y_pred, y_pred_proba

    def score(self, X, y):
        return self.classifier.score(X, y)

    def get_params(self, deep):
        return self.classifier.get_params(deep)

    def set_params(self, **params):
        self.classifier.set_params(**params)
        self.params = params
        return self

    def get_available_classifiers(self):
        return self.available_classifiers

    def feature_importance(self):
        if self.name == "Logistic Regression":
            importance = self.classifier.coef_[0]
        elif self.name in ["AdaBoost", "Random Forest", "XGBoost"]:
            importance = self.classifier.feature_importances_
        else:
            raise ValueError(
                f"For classifier {self.name} feature \
                               importance could not be calculated."
            )
        return importance

    def name(self):
        return self.name


class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Ensemble classifier for MLClassifiers.

    Parameters:
    clf : `iterable`
      A list of classifier objects.
    weights : `list` (default: `None`)
      If `None`, the majority rule voting will be applied to the predicted
        class labels. If a list of weights (`float` or `int`) is provided,
        the averaged raw probabilities (via `predict_proba`) will be used
        to determine the most confident class label.
    """

    def __init__(self, classifier_list, weights=None):
        self.classifier_list = classifier_list
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
        for clf in self.classifier_list:
            clf.fit(X, y, **fit_params)

    def predict(self, X):
        """
        Parameters:
        X : numpy array, shape = [n_samples, n_features]

        Returns:
        maj : list or numpy array, shape = [n_samples]
            Predicted class labels by majority rule
        """
        self.classes_ = np.asarray(
            [clf.predict(X) for clf in self.classifier_list]
        )
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
        self.probas_ = [clf.predict_proba(X) for clf in self.classifier_list]
        avg = np.average(self.probas_, axis=0, weights=self.weights)

        return avg

    def get_classifier_list(self):
        return self.classifier_list
