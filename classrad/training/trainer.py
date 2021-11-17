import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lofo import Dataset, LOFOImportance, plot_importance
from classrad.utils.io import load_json
from classrad.utils.statistics import wilcoxon_unpaired
from classrad.utils.visualization import get_subplots_dimensions
from sklearn.model_selection import GridSearchCV


class Trainer:
    def __init__(
        self,
        dataset,
        models,
        result_dir,
        meta_colnames,
        num_features=10,
        n_jobs=1,
        random_state=None,
    ):
        self.dataset = dataset
        self.models = models
        self.result_dir = result_dir
        self.meta_colnames = meta_colnames
        self.num_features = num_features
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.model_names = [model.classifier_name for model in models]
        self.result_df = None
        self.test_indices = None

        self.result_dir.mkdir(parents=True, exist_ok=True)

    def init_result_df(self):
        self.result_df = self.dataset.df[self.meta_colnames].copy()
        self.result_df["test"] = 0
        self.test_indices = self.dataset.X_test.index.values
        self.result_df.loc[self.test_indices, "test"] = 1
        self.result_df["cv_split"] = -1
        for i in range(self.dataset.n_splits):
            self.result_df.loc[self.dataset.X_val_fold[i].index, "cv_split"] = i

    def train_cross_validation(self):
        """
        Evaluate the models.
        """
        self.init_result_df()
        # Feature standardization and selection
        self.dataset.standardize_features()
        self.dataset.select_features(k=self.num_features)

        for model in self.models:
            model_name = model.classifier_name
            pred_colname = f"{model_name}_pred"
            pred_proba_colname = f"{model_name}_pred_proba"
            self.result_df[pred_colname] = -1
            self.result_df[pred_proba_colname] = -1

            print(f"Training and infering model: {model_name}")

            optimizer = HyperparamOptimizer(
                dataset=self.dataset,
                model=model,
                param_dir=self.result_dir / "optimal_params",
            )
            model = optimizer.load_or_tune_hyperparameters()

            for i in range(self.dataset.n_splits):
                print(f"Evaluating fold: {i}")

                X_train_fold = self.dataset.X_train_fold[i]
                y_train_fold = self.dataset.y_train_fold[i]
                X_val_fold = self.dataset.X_val_fold[i]
                # Fit and predict
                model.fit(X_train_fold, y_train_fold)
                y_pred_fold, y_pred_proba_fold = model.predict_label_and_proba(
                    X_val_fold
                )
                # Write results
                fold_indices = X_val_fold.index
                self.result_df.loc[fold_indices, pred_colname] = y_pred_fold

                self.result_df.loc[fold_indices, pred_proba_colname] = y_pred_proba_fold

            model.fit(self.dataset.X_train, self.dataset.y_train)
            y_pred_test, y_pred_proba_test = model.predict_label_and_proba(
                self.dataset.X_test
            )
            self.result_df.loc[self.test_indices, pred_colname] = y_pred_test
            self.result_df.loc[
                self.test_indices, pred_proba_colname
            ] = y_pred_proba_test
        self.save_results()

        return self

    def save_results(self):
        df_name = f"predictions_{self.dataset.task_name}.csv"
        self.result_df.to_csv(self.result_dir / df_name, index=False)
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
                {"feature": self.dataset.X_train.columns, "importance": importances}
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
            self.result_dir / "feature_importance.png", bbox_inches="tight", dpi=100
        )
        plt.show()

        return self

    def plot_lofo_importance(self, model):
        dataset = Dataset(
            df=self.dataset.df,
            target=self.target,
            features=self.dataset.best_features,
        )
        lofo_imp = LOFOImportance(
            dataset, model=model.classifier, scoring="neg_mean_squared_error"
        )
        importance_df = lofo_imp.get_importance()
        plot_importance(importance_df, figsize=(12, 12))
        plt.tight_layout()
        plt.show()


class HyperparamOptimizer:
    def __init__(self, dataset, model, param_dir):
        self.dataset = dataset
        self.model = model
        self.param_dir = Path(param_dir)
        self.param_grid = None

    def get_grid_RandomForest(self):
        n_estimators = [200, 600, 1000]
        max_features = ["auto", "sqrt"]
        max_depth = [10, 50, None]
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 4]
        bootstrap = [True, False]

        self.param_grid = {
            "n_estimators": n_estimators,
            "max_features": max_features,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "bootstrap": bootstrap,
        }
        return self

    def get_grid_XGBoost(self):
        self.param_grid = {
            "learning_rate": [0.05, 0.10, 0.20, 0.30],
            "max_depth": [2, 4, 8, 12, 15],
            "min_child_weight": [1, 3, 7],
            "gamma": [0.0, 0.1, 0.3],
            "colsample_bytree": [0.3, 0.5, 0.7],
        }
        return self

    def get_grid_LogReg(self):
        self.param_grid = {
            "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            "penalty": ["l1", "l2", "elasticnet", "none"],
        }
        return self

    def get_grid_SVM(self):
        cs = [0.001, 0.01, 0.1, 1, 10]
        gammas = [0.001, 0.01, 0.1, 1]
        kernels = ["rbf"]
        self.param_grid = {"kernel": kernels, "C": cs, "gamma": gammas}
        return self

    def save_params(self, params):
        self.param_dir.mkdir(exist_ok=True)
        save_path = Path(self.param_dir) / (self.model.classifier_name + ".json")
        io.save_json(save_path, params)

    def load_params(self):
        param_path = self.param_dir / (self.model.classifier_name + ".json")
        print(f"Loading parameters from: {param_path}")
        if param_path.exists():
            optimal_params = io.load_json(param_path)
            self.model.set_params(optimal_params)
        else:
            raise FileNotFoundError(
                "No param file found. Run \
                                    `tune_hyperparameters` first."
            )

        return self

    def update_model_params_grid_search(self):
        if self.param_grid is None:
            raise ValueError("First select param grid!")
        else:
            param_searcher = GridSearchCV(
                estimator=self.model.classifier,
                param_grid=self.param_grid,
                scoring="roc_auc",
                cv=self.dataset.cv_splits,
                verbose=0,
                n_jobs=-1,
            )
            rs = param_searcher.fit(X=self.dataset.X_train, y=self.dataset.y_train)
            optimal_params = rs.best_params_
            print(f"Best params for {self.model.classifier_name}: {optimal_params}")
            self.model.set_params(optimal_params)
            self.save_params(optimal_params)

            return self

    def get_param_grid(self):
        model_name = self.model.classifier_name
        if model_name == "Random Forest":
            self.get_grid_RandomForest()
        elif model_name == "XGBoost":
            self.get_grid_XGBoost()
        elif model_name == "Logistic Regression":
            self.get_grid_LogReg()
        elif model_name == "SVM":
            self.get_grid_SVM()
        else:
            return ValueError(
                f"Hyperparameter tuning for {model_name} not implemented!"
            )

        return self

    def tune_hyperparameters(self):
        try:
            self.get_param_grid()
            self.update_model_params_grid_search()
        except Exception:
            pass

        return self

    def load_or_tune_hyperparameters(self):
        try:
            self.load_params()
        except Exception:
            print("Params couldn't be loaded. Starting hyperparameter tuning...")
            self.tune_hyperparameters()

        return self.model
