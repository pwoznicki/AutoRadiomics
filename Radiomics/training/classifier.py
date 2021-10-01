
"""
Create a class MLClassifier, allow to choose a classifier from sklearn: Random Forest, AdaBoost, Linear SVM,
Gaussian Process Classifier, Lasso. Allow also to choose an ensemble of all classifiers.
"""
import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

class MLClassifier(ClassifierMixin):
    def __init__(self, classifier_name=None, classifier_parameters={}):
        self.classifier_name = classifier_name
        self.classifier_parameters = classifier_parameters
        self.classifier = None
        self.available_classifiers = ['Random Forest', 'AdaBoost', 'Logistic Regression' \
                                     'Gaussian Process Classifier']

    def fit(self, X, y):
        if self.classifier_name == 'Random Forest':
            self.classifier = RandomForestClassifier(**self.classifier_parameters)
        elif self.classifier_name == 'AdaBoost':
            self.classifier = AdaBoostClassifier(**self.classifier_parameters)
        elif self.classifier_name == 'Logistic Regression':
            self.classifier = LogisticRegression(**self.classifier_parameters)
        elif self.classifier_name == 'Gaussian Process Classifier':
            self.classifier = GaussianProcessClassifier(**self.classifier_parameters)
        else:
            raise ValueError('Classifier name not recognized')
        self.classifier.fit(X, y)

    def predict(self, X):
        return self.classifier.predict(X)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    def get_classifier(self):
        return self.classifier

    def get_classifier_name(self):
        return self.classifier_name

    def get_classifier_parameters(self):
        return self.classifier_parameters
    
    def get_available_classifiers(self):
        return self.available_classifiers
    
    def update_classifier(self, new_classifier_name):
        self.classifier_name = new_classifier_name


"""
Create class EnsembleClassifier that takes a list of classifiers and creates an ensemble of the classifiers, has functionality like sklearn models.
"""

class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Ensemble classifier for MLClassifiers.

    Parameters:
    clf : `iterable`
      A list of classifier objects.
    weights : `list` (default: `None`)
      If `None`, the majority rule voting will be applied to the predicted class labels.
        If a list of weights (`float` or `int`) is provided, the averaged raw probabilities (via `predict_proba`)
        will be used to determine the most confident class label.
    """
    def __init__(self, classifier_list, weights=None):
        self.classifier_list = classifier_list
        self.weights = weights

    def fit(self, X, y):
        """
        Fit the scikit-learn estimators. 

        Parameters:
        X : numpy array, shape = [n_samples, n_features]
            Training data
        y : list or numpy array, shape = [n_samples]
            Class labels
        """
        for clf in self.classifier_list:
            clf.fit(X, y)

    def predict(self, X):
        """
        Parameters:
        X : numpy array, shape = [n_samples, n_features]

        Returns:
        maj : list or numpy array, shape = [n_samples]
            Predicted class labels by majority rule
        """
        self.classes_ = np.asarray([clf.predict(X) for clf in self.classifier_list])
        if self.weights:
            avg = self.predict_proba(X)
            maj = np.apply_along_axis(lambda x: max(enumerate(x), key=operator.itemgetter(1))[0], axis=1, arr=avg)
        else:
            maj = np.asarray([np.argmax(np.bincount(self.classes_[:,c])) for c in range(self.classes_.shape[1])])

        return maj

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

# # Load the training data

# # Load the training data
# train_data = np.loadtxt('training_data.txt', skiprows=1, delimiter=' ')
# train_labels = train_data[:, 0]
# train_features = train_data[:, 1:]

# # Load the test data

# # Load the test data
# test_data = np.loadtxt('test_data.txt', skiprows=1, delimiter=' ')
# test_features = test_data[:, :]

# # Train the classifier

# # Train the classifier
# clf = EnsembleClassifier(clfs=[RandomForestClassifier(n_estimators=100),
#                                ExtraTreesClassifier(n_estimators=100),
#                                GradientBoostingClassifier(n_estimators=100)],
#                          weights=[2, 2, 1])
# clf.fit(train_features, train_labels)

# # Predict the test data

# # Predict the test data
# predictions = clf.predict(test_features)

# # Save the predictions to a file

# # Save the predictions to a file
# np.savetxt('predictions.txt', predictions.astype(int), fmt='%i')
