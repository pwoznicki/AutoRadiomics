"""
Create an Evaluator class to evaluate predictions for classification task in terms of ROC AUC and sensitivity/specificity.
The models to evaluate are created on top of sklearn classifiers.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, auc
from .metrics import roc_auc_score
from .utils import get_optimal_threshold, get_sensitivity_specificity
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
        self.results = []
        self.predictions = []
        self.predictions_proba = []
        self.predictions_proba_test = []
        self.predictions_test = []
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
            self.predictions.append(model_predictions)
            self.predictions_proba.append(model_predictions_proba)
            model_mean_score = np.mean(model_scores)
            model_std_score = np.std(model_scores)
            self.results.append((model_mean_score, model_std_score))

            self.dataset.standardize_features()
            self.dataset.select_features()

            model.fit(self.dataset.X_train, self.dataset.y_train)
            y_pred_test = model.predict(self.dataset.X_test)
            y_pred_proba_test = model.predict_proba(self.dataset.X_test)
            
            self.predictions_test.append(y_pred_test)
            self.predictions_proba_test.append(y_pred_proba_test)

            print(f'For {model_name} AUC = {model_mean_score} +/- {model_std_score}')
        self.best_model_idx = np.argmax([t[0] for t in self.results])
        self.best_model = self.models[self.best_model_idx]
        best_model_preds = self.predictions_proba_test[self.best_model_idx][:,1]
        self.best_model_score_test = roc_auc_score(self.dataset.y_test.values, best_model_preds)
        print(f'Best model: {self.best_model.classifier_name} - AUC on test set = {self.best_model_score_test}')

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
    
    def plot_roc_curve(self, title=None):
        """
        Plot the ROC curve.
        """
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=.8)
        for model_predictions in self.predictions_proba:
            fpr, tpr, thresholds = roc_curve(self.dataset.y_test, model_predictions[:,1])
            ax.plot(fpr, tpr, lw=2, alpha=.8)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        if title:
            ax.set_title(title)
        else:
            ax.set_title('ROC Curve')
        plt.tight_layout()
        plt.show()
        return self
    
    def plot_roc_curve_test(self, title=None):
        """
        Plot the ROC curve.
        """
        plt.rcParams['font.size'] = 16
        plt.set_cmap('Pastel1')

        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=.8)
        for model_name, model_predictions in zip(self.model_names, self.predictions_proba_test):
            y_true = self.dataset.y_test
            y_pred = model_predictions[:,1]
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
            roc_auc = np.round(auc(fpr, tpr), 3)
            label = f"{model_name} AUC = {roc_auc}"
            ax.plot(fpr, tpr, lw=3, alpha=.8, label=label)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        if title:
            ax.set_title(title)
        else:
            ax.set_title('ROC Curve')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()
        return self
    
    def plot_precision_recall_curve(self):
        """
        Plot the precision recall curve.
        """
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot([1, 0], [1, 0], linestyle='--', lw=2, color='black', alpha=.8)
        for model_predictions in self.predictions_proba:
            precision, recall, thresholds = precision_recall_curve(self.dataset.y_test, model_predictions[:,1])
            ax.plot(recall, precision, lw=2, alpha=.8)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision Recall Curve')
        plt.tight_layout()
        plt.show()
        return self
    
    def plot_precision_recall_curve_test(self):
        """
        Plot the precision recall curve.
        """
        plt.rcParams['font.size'] = 16
        plt.set_cmap('Pastel1')

        fig, ax = plt.subplots(figsize=(10,6))
        for model_name, model_predictions in zip(self.model_names, self.predictions_proba_test):
            precision, recall, thresholds = precision_recall_curve(self.dataset.y_test, model_predictions[:,1])
            breakpoint()
            ax.plot(recall, precision, lw=2, alpha=.8, label=model_name)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision Recall Curve')
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.show()
        return self
    
    def plot_confusion_matrix(self):
        """
        Plot the confusion matrix.
        """
        fig, ax = plt.subplots(figsize=(10,6))
        for model_predictions in self.predictions_proba:
            cm = confusion_matrix(self.dataset.y_test, model_predictions[:,1])
            sns.heatmap(cm, annot=True, fmt='d', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            plt.tight_layout()
            plt.show()
        return self
    
    def plot_confusion_matrix_test(self):
        """
        Plot the confusion matrix.
        """
        fig, ax = plt.subplots(figsize=(10,6))
        for model_predictions in self.predictions_test:
            cm = confusion_matrix(self.dataset.y_test, model_predictions)
            sns.heatmap(cm, annot=True, fmt='d', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            plt.tight_layout()
            plt.show()
        return self
    
    def plot_feature_importance(self):
        """
        Plot the feature importance.
        """
        fig, ax = plt.subplots(figsize=(10,6))
        for model_predictions in self.predictions_proba:
            feature_importance = pd.DataFrame(model_predictions[:,1], columns=['importance'])
            feature_importance['feature'] = self.X_train.columns
            feature_importance.sort_values(by='importance', ascending=False, inplace=True)
            sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax)
            ax.set_xlabel('Feature Importance')
            ax.set_ylabel('Feature')
            ax.set_title('Feature Importance')
            plt.tight_layout()
            plt.show()
        return self
    
    def plot_feature_importance_test(self):
        """
        Plot the feature importance.
        """
        fig, ax = plt.subplots(figsize=(10,6))
        for model_predictions in self.predictions_proba_test:
            feature_importance = pd.DataFrame(model_predictions[:,1], columns=['importance'])
            feature_importance['feature'] = self.dataset.X_train.columns
            feature_importance.sort_values(by='importance', ascending=False, inplace=True)
            sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax)
            ax.set_xlabel('Feature Importance')
            ax.set_ylabel('Feature')
            ax.set_title('Feature Importance')
            plt.tight_layout()
            plt.show()
        return self
    
    def plot_roc_curve_all(self):
        """
        Plot the ROC curve for all models.
        """
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=.8)
        for model_predictions in self.predictions_proba:
            fpr, tpr, thresholds = roc_curve(self.dataset.y_test, model_predictions[:,1])
            ax.plot(fpr, tpr, lw=2, alpha=.8)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        plt.tight_layout()
        plt.show()
        return self
    
    def plot_precision_recall_curve_all(self):
        """
        Plot the precision recall curve for all models.
        """
        fig, ax = plt.subplots(figsize=(10,6))
        for model_predictions in self.predictions_proba:
            precision, recall, thresholds = precision_recall_curve(self.dataset.y_test, model_predictions[:,1])
            ax.plot(recall, precision, lw=2, alpha=.8)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision Recall Curve')
        plt.tight_layout()
        plt.show()
        return self
    
    def plot_confusion_matrix_all(self):
        """
        Plot the confusion matrix for all models.
        """
        fig, ax = plt.subplots(figsize=(10,6))
        for model_predictions in self.predictions_proba:
            cm = confusion_matrix(self.dataset.y_test, model_predictions[:,1])
            sns.heatmap(cm, annot=True, fmt='d', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            plt.tight_layout()
            plt.show()
        return self

    def plot_feature_importance_all(self):
        """
        Plot the feature importance for all models.
        """
        fig, ax = plt.subplots(figsize=(10,6))
        for model_predictions in self.predictions_proba:
            feature_importance = pd.DataFrame(model_predictions[:,1], columns=['importance'])
            feature_importance['feature'] = self.X_train.columns
            feature_importance.sort_values(by='importance', ascending=False, inplace=True)
            sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax)
            ax.set_xlabel('Feature Importance')
            ax.set_ylabel('Feature')
            ax.set_title('Feature Importance')
            plt.tight_layout()
            plt.show()
        return self
    
    def plot_all(self):
        """
        Plot all the graphs.
        """
        self.plot_roc_curve()
        self.plot_roc_curve_test()
        self.plot_precision_recall_curve()
        self.plot_precision_recall_curve_test()
        self.plot_confusion_matrix()
        self.plot_confusion_matrix_test()
        self.plot_feature_importance()
        self.plot_feature_importance_test()
        return self
    
    def get_best_model(self):
        """
        Get the best model.
        """
        return self.best_model
    
    def get_best_model_name(self):
        """
        Get the best model name.
        """
        return self.best_model_name
    
    def get_best_model_params(self):
        """
        Get the best model parameters.
        """
        return self.best_model_params
    
    def get_best_model_param_names(self):
        """
        Get the best model parameter names.
        """
        return self.best_model_param_names
    
    def get_best_model_score(self):
        """
        Get the best model score.
        """
        return self.best_model_score
    
    def get_best_model_scores(self):
        """
        Get the best model scores.
        """
        return self.best_model_scores
    
    def get_best_model_scores_std(self):
        """
        Get the best model scores std.
        """
        return self.best_model_scores_std
    
    def get_results(self):
        """
        Get the results.
        """
        return self.results
    
    def get_predictions(self):
        """
        Get the predictions.
        """
        return self.predictions
    
    def get_predictions_proba(self):
        """
        Get the predictions proba.
        """
        return self.predictions_proba
    
    def get_predictions_proba_test(self):
        """
        Get the predictions proba test.
        """
        return self.predictions_proba_test
    
    def get_predictions_test(self):
        """
        Get the predictions test.
        """
        return self.predictions_test