from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import lofo
import mlflow
from typing import List
from classrad.config.type_definitions import PathLike
from classrad.utils.visualization import get_subplots_dimensions
from classrad.data.dataset import Dataset
from classrad.models.classifier import MLClassifier
from classrad.feature_selection.feature_selector import FeatureSelector
from classrad.training.optimizer import OptunaOptimizer, HyperparamOptimizer
from classrad.config import config
from sklearn.model_selection import cross_val_score


class Trainer:
    def __init__(
        self,
        dataset: Dataset,
        models: List[MLClassifier],
        result_dir: PathLike,
        meta_colnames: List[str] = [],
        feature_selection: str = "lasso",
        num_features: int = 10,
        experiment_name: str = "baseline",
    ):
        self.dataset = dataset
        self.models = models
        self.result_dir = Path(result_dir)
        self.meta_colnames = meta_colnames
        self.feature_selection = feature_selection
        self.num_features = num_features
        self.experiment_name = experiment_name
        self.model_names = [model.classifier_name for model in models]
        self.test_indices = None

        self.result_dir.mkdir(parents=True, exist_ok=True)

    def _tune_sklearn_gridsearch(self, model):
        optimizer = HyperparamOptimizer(
            dataset=self.dataset,
            model=model,
            param_dir=self.result_dir / "optimal_params",
        )
        model = optimizer.load_or_tune_hyperparameters()

    def _train_cross_validation_single_model(self, model):
        print(f"Training and infering model: {model.name()}")
        optimizer = OptunaOptimizer(model=model)
        params = optimizer.params()
        optimizer.study.optimize(self._objective(model, params), n_trials=100)
        best_hyperparams = optimizer.trial.params
        print(f"Best hyperparameters: {best_hyperparams}")
        return best_hyperparams

    def _init_mlflow(self):
        model_registry = Path(config.MLFLOW_DIR)
        model_registry.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri("file://" + str(model_registry.absolute()))
        mlflow.set_experiment(experiment_name=self.experiment_name)

    def _objective(self, model, params):
        model.set_params(**params)
        return cross_val_score(
            model, self.dataset.X_train, self.dataset.y_train, cv=5
        ).mean()
        # model.fit(self.dataset.X_train, self.dataset.y_train)
        # y_pred = model.predict(self.dataset.X_test)
        # auc = roc_auc_score(self.dataset.y_test, y_pred)
        # return {"AUC ROC": -auc}

    def train_cross_validation(self):
        """
        Train all the models.
        """
        # self.init_result_df()
        self._init_mlflow()
        # client = mlflow.tracking.MlflowClient()
        with mlflow.start_run(run_name="radiomics") as run:  # NOQA: F841
            # Feature standardization and selection
            self.dataset.standardize_features()
            feature_selector = FeatureSelector()
            self.dataset = feature_selector.fit_transform_dataset(
                self.dataset,
                method=self.feature_selection,
                k=self.num_features,
            )
            for model in self.models:
                best_hyperparams = self._train_cross_validation_single_model(
                    model
                )
                mlflow.log_param("model", model)
                mlflow.log_param(
                    "feature selection method", self.feature_selection
                )
                mlflow.log_param("features", self.dataset.best_features)

                mlflow.set_tag("best hyperparams", best_hyperparams)

        self.save_results()

        return self

    def plot_feature_importance(self, model, ax=None):
        """
        Plot importance of features for a single model
        Args:
            model [MLClassifier] - classifier
            ax (optional) - pyplot axes object
        """
        model_name = model.classifier_name
        try:
            importances = model.feature_importance()
            importance_df = pd.DataFrame(
                {
                    "feature": self.dataset.X_train.columns,
                    "importance": importances,
                }
            )
            sns.barplot(x="feature", y="importance", data=importance_df, ax=ax)
            ax.tick_params(axis="both", labelsize="x-small")
            ax.set_ylabel("Feature importance")
            ax.set_title(model_name)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        except Exception:
            print(f"For {model_name} feature importance cannot be calculated.")

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
            fig.suptitle(f"Feature Importance for {self.dataset.task_name}")
        fig.tight_layout()
        fig.savefig(
            self.result_dir / "feature_importance.png",
            bbox_inches="tight",
            dpi=100,
        )
        plt.show()

        return self

    def plot_lofo_importance(self, model):
        dataset = lofo.Dataset(
            df=self.dataset.df,
            target=self.target,
            features=self.dataset.best_features,
        )
        lofo_imp = lofo.LOFOImportance(
            dataset, model=model.classifier, scoring="neg_mean_squared_error"
        )
        importance_df = lofo_imp.get_importance()
        lofo.plot_importance(importance_df, figsize=(12, 12))
        plt.tight_layout()
        plt.show()


class Inferer:
    def __init__(self, dataset, model, result_dir):
        self.dataset = dataset
        self.model = model
        self.result_df = None
        self.result_dir = result_dir

    def init_result_df(self):
        self.result_df = self.dataset.df[self.meta_colnames].copy()
        self.test_indices = self.dataset.X_test.index.values
        self.result_df = self.add_splits_to_result_df(
            self.result_df, self.test_indices
        )

    def add_splits_to_result_df(self, result_df, test_indices):
        result_df["test"] = 0
        result_df.loc[test_indices, "test"] = 1
        result_df["cv_split"] = -1
        for i in range(self.dataset.n_splits):
            result_df.loc[self.dataset.X_val_fold[i].index, "cv_split"] = i

        return result_df

    def fit_eval_split(
        self, X_train, y_train, X_val, pred_colname, pred_proba_colname
    ):
        # Fit and predict
        self.model.fit(X_train, y_train)
        y_pred_fold, y_pred_proba_fold = self.model.predict_label_and_proba(
            X_val
        )
        # Write results
        fold_indices = X_val.index
        self.result_df.loc[fold_indices, pred_colname] = y_pred_fold

        self.result_df.loc[
            fold_indices, pred_proba_colname
        ] = y_pred_proba_fold

    def fit_eval_all(self):
        model_name = self.model.classifier_name
        pred_colname = f"{model_name}_pred"
        pred_proba_colname = f"{model_name}_pred_proba"
        self.result_df[pred_colname] = -1
        self.result_df[pred_proba_colname] = -1
        for i in range(self.dataset.n_splits):
            print(f"Evaluating fold: {i}")
            X_train_fold = self.dataset.X_train_fold[i]
            y_train_fold = self.dataset.y_train_fold[i]
            X_val_fold = self.dataset.X_val_fold[i]
            self.fit_eval_split(
                X_train_fold,
                y_train_fold,
                X_val_fold,
                pred_colname,
                pred_proba_colname,
            )
        self.fit_eval_split(
            self.dataset.X_train,
            self.dataset.y_train,
            self.dataset.X_test,
            pred_colname,
            pred_proba_colname,
        )

    def save_results(self):
        df_name = f"predictions_{self.dataset.task_name}.csv"
        self.result_df.to_csv(self.result_dir / df_name, index=False)
        return self
