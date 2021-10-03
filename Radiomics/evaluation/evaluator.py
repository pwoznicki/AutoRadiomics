"""
Create an Evaluator class to evaluate predictions for classification task in terms of ROC AUC and sensitivity/specificity.
The models to evaluate are created on top of sklearn classifiers.
"""
from matplotlib import axes
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, auc
from Radiomics.utils.visualization import get_subplots_dimensions, plot_for_all
from .metrics import roc_auc_score
from .utils import get_optimal_threshold, get_sensitivity_specificity


plt.set_cmap('Pastel1')


class Evaluator():
    def __init__(self, dataset, models, n_jobs=1, 
                random_state=None):
        self.dataset = dataset
        self.models = models
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.results = None
        self.predictions = None
        self.predictions_proba = None
        self.predictions_proba_test = None
        self.predictions_test = None
        self.best_model = None
        self.best_model_name = None
        self.best_model_idx = None
        self.best_model_params = None
        self.best_model_param_names = None
        self.best_model_score = None
        self.best_model_scores = None
        self.best_model_scores_std = None
        self.model_names = [model.classifier_name for model in models]

    def evaluate_cross_validation(self):
        """
        Evaluate the models.
        """
        self.results = {}
        self.predictions = {}
        self.predictions_proba = {}
        self.predictions_proba_test = {}
        self.predictions_test = {}
        for model in self.models:
            model_name = model.classifier_name
            print(f'Evaluating model: {model_name}')
            model_predictions = []
            model_predictions_proba= []
            model_scores = []
            for fold_idx, (train_idx, val_idx) in enumerate(self.dataset.cross_validation_split()):
                print(f'Evaluating fold: {fold_idx}')
                self.dataset.get_cross_validation_fold(train_idx, val_idx)
                #Standarization of features:
                self.dataset.standardize_features_cross_validation()
                #Feature selection
                self.dataset.select_features_cross_validation()
                #Fit and predict
                model.fit(self.dataset.X_train_fold, self.dataset.y_train_fold)
                y_pred_fold = model.predict(self.dataset.X_val_fold)
                y_pred_proba_fold = model.predict_proba(self.dataset.X_val_fold)
                #Save results
                model_scores.append(roc_auc_score(self.dataset.y_val_fold, y_pred_fold))
                model_predictions.append(y_pred_fold)
                model_predictions_proba.append(y_pred_proba_fold)
            
            self.predictions[model_name] = model_predictions
            self.predictions_proba[model_name] = model_predictions_proba
            model_mean_score = np.mean(model_scores)
            model_std_score = np.std(model_scores)
            self.results[model_name] = (model_mean_score, model_std_score)

            self.dataset.standardize_features()
            self.dataset.select_features()

            model.fit(self.dataset.X_train, self.dataset.y_train)
            y_pred_test = model.predict(self.dataset.X_test)
            y_pred_proba_test = model.predict_proba(self.dataset.X_test)
            
            self.predictions_test[model_name] = y_pred_test
            self.predictions_proba_test[model_name] = y_pred_proba_test

            print(f'For {model_name} AUC = {model_mean_score} +/- {model_std_score}')
        self.best_model_idx = np.argmax([t[0] for t in self.results])
        self.best_model = self.models[self.best_model_idx]
        best_model_preds = self.predictions_proba_test[self.best_model.classifier_name][:,1]
        self.best_model_score_test = roc_auc_score(self.dataset.y_test.values, best_model_preds)
        print(f'Best model: {self.best_model.classifier_name} - AUC on test set = {self.best_model_score_test}')

        return self

    def plot_results(self):
        """
        Plot the results.
        """
        fig, ax = plt.subplots(figsize=(10,6))
        ax.errorbar(np.arange(len(self.model_names)), self.results[0], yerr=self.results[2], fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)
        ax.set_xticks(np.arange(len(self.model_names)))
        ax.set_xticklabels(self.model_names, rotation=45, ha='right')
        ax.set_ylabel('ROC AUC')
        ax.set_xlabel('Model')
        ax.set_title('Model Evaluation')
        plt.tight_layout()
        plt.show()

        return self
    
    def plot_predictions(self):
        """
        Plot the predictions.
        """
        fig, ax = plt.subplots(figsize=(10,6))
        ax.scatter(np.arange(len(self.dataset.y_test)), self.dataset.y_test, color='black', s=10, alpha=0.5)
        ax.scatter(np.arange(len(self.dataset.y_test)), self.predictions_test[self.best_model_idx], color='red', s=10, alpha=0.5)
        ax.set_xticks(np.arange(len(self.dataset.y_test)))
        ax.set_xticklabels(self.dataset.y_test, rotation=45, ha='right')
        ax.set_ylabel('Prediction')
        ax.set_xlabel('Sample')
        ax.set_title('Predictions')
        plt.tight_layout()
        plt.show()
        return self
    
    def plot_predictions_proba(self):
        """
        Plot the predictions.
        """
        fig, ax = plt.subplots(figsize=(10,6))
        ax.scatter(np.arange(len(self.dataset.y_test)), self.dataset.y_test, color='black', s=10, alpha=0.5)
        ax.scatter(np.arange(len(self.dataset.y_test)), self.predictions_proba_test[self.best_model_idx][:,1], color='red', s=10, alpha=0.5)
        ax.set_xticks(np.arange(len(self.dataset.y_test)))
        ax.set_xticklabels(self.dataset.y_test, rotation=45, ha='right')
        ax.set_ylabel('Prediction')
        ax.set_xlabel('Sample') 
        ax.set_title('Predictions')
        plt.tight_layout()
        plt.show()

        return self
    
    def plot_roc_curve(self, model_name, ax=None):
        """
        Plot the ROC curve.
        """
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=.8)
        y_true = self.dataset.y_test
        y_pred = self.predictions_proba_test[model_name][:,1]
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = np.round(auc(fpr, tpr), 3)
        label = f"{model_name} - AUC = {roc_auc}"
        ax.plot(fpr, tpr, lw=3, alpha=.8, label=label)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(model_name)
        ax.legend(loc='lower right')

        return self   
 
    def plot_roc_curve_all(self, title=None):
        """
        Plot the ROC Curve for all models.
        """
        nrows, ncols, figsize = get_subplots_dimensions(len(self.models))
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for i, model_name in enumerate(self.model_names):
            ax = fig.axes[i]
            self.plot_roc_curve(model_name, ax=ax)
        if title:
            fig.suptitle(title)
        else:
            fig.suptitle(f'ROC Curve for {self.dataset.task_name} in test dataset')
        fig.tight_layout()
        plt.show()

        return self
    
    def plot_precision_recall_curve(self, model_name, ax=None):
        """
        Plot the precision recall curve.
        """
        y_true = self.dataset.y_test
        y_pred = self.predictions_proba_test[model_name][:, 1]
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        ax.plot(recall, precision, lw=2, alpha=.8, label=model_name)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(model_name)
        ax.legend(loc='lower left')

        return self

    def plot_precision_recall_curve_all(self, title=None):
        """
        Plot the precision recall curve for all models.
        """
        n_models = len(self.models)
        nrows, ncols, figsize = get_subplots_dimensions(n_models)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for i in range(n_models):
            model_name = self.model_names[i]
            ax = fig.axes[i]
            self.plot_precision_recall_curve(model_name, ax=ax)
        if title:
            fig.suptitle(title)
        else:
            fig.suptitle(f'Precision-Recall Curve for {self.dataset.task_name} in test dataset')
        fig.tight_layout()        
        plt.show()

        return self
    
    def plot_confusion_matrix(self, model_name, ax=None):
        """
        Plot the confusion matrix for a single model.
        """
        y_true = self.dataset.y_test
        y_pred = self.predictions_test[model_name]
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(model_name)

        return self

    def plot_confusion_matrix_all(self, title=None):
        """
        Plot the confusion matrix for all models.
        """
        nrows, ncols, figsize = get_subplots_dimensions(len(self.models))
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for i, model_name in enumerate(self.model_names):
            ax = fig.axes[i]
            self.plot_confusion_matrix(model_name, ax=ax)
        if title:
            fig.suptitle(title)
        else:
            fig.suptitle(f'Confusion Matrix for {self.dataset.task_name} in test dataset)')
        plt.tight_layout()    
        plt.show()

        return self

    def plot_feature_importance(self, model, ax=None):
        """
        Plot importance of features for a single model
        Args:
            model [MLClassifier] - classifier
            ax (optional) - pyplot axes object
        """
        try:
            model_name = model.classifier_name
            importances = model.feature_importance()
            importance_df = pd.DataFrame({'feature': self.dataset.X_train.columns, 'importance': importances})
            sns.barplot(x='feature', y='importance', data=importance_df, ax=ax)
            # ax.set_xlabel('Feature Importance')
            ax.set_ylabel('Feature importance')
            ax.set_title(model_name)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
        except:
            print(f'For {model_name} feature importance cannot be calculated.')

        return self

    def plot_feature_importance_all(self, title=None):
        """
        Plot the feature importance for all models.
        """
        nrows, ncols, figsize = get_subplots_dimensions(len(self.models))
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for i, model in enumerate(self.models):
            ax = fig.axes[i]
            self.plot_feature_importance(model, ax=ax)
        if title:
            fig.suptitle(title)
        else:
            fig.suptitle(f'Feature Importance for {self.dataset.task_name}')
        fig.tight_layout()
        plt.show()

        return self

    def plot_all(self):
        """
        Plot all the graphs.
        """
        self.plot_roc_curve_all()
        self.plot_precision_recall_curve_all()
        self.plot_confusion_matrix_all()
        self.plot_feature_importance_all()

        return self