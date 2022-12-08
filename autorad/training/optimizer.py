import logging
from pathlib import Path

import mlflow
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import GridSearchCV

from autorad.config import config
from autorad.utils import io

log = logging.getLogger(__name__)


class OptunaOptimizer:
    def __init__(
        self,
        n_trials: int = 30,
    ):
        self.n_trials = n_trials

    def create_study(self, study_name):
        return optuna.create_study(
            direction="maximize",
            study_name=study_name,
            sampler=TPESampler(seed=config.SEED),
        )


class GridSearchOptimizer:
    def __init__(self, dataset, model, param_dir):
        self.dataset = dataset
        self.model = model
        self.param_dir = Path(param_dir)
        self.hyperparam_grid = None

    def get_grid_RandomForest(self):
        n_estimators = [200, 600, 1000]
        max_features = ["auto", "sqrt"]
        max_depth = [10, 50, None]
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 4]
        bootstrap = [True, False]

        self.hyperparam_grid = {
            "n_estimators": n_estimators,
            "max_features": max_features,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "bootstrap": bootstrap,
        }
        return self

    def get_grid_XGBoost(self):
        self.hyperparam_grid = {
            "learning_rate": [0.05, 0.10, 0.20, 0.30],
            "max_depth": [2, 4, 8, 12, 15],
            "min_child_weight": [1, 3, 7],
            "gamma": [0.0, 0.1, 0.3],
            "colsample_bytree": [0.3, 0.5, 0.7],
        }
        return self

    def get_grid_LogReg(self):
        self.hyperparam_grid = {
            "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            "penalty": ["l2", "none"],
        }
        return self

    def get_grid_SVM(self):
        cs = [0.001, 0.01, 0.1, 1, 5, 10]
        gammas = [0.001, 0.01, 0.1, 1]
        kernels = ["rbf"]
        self.hyperparam_grid = {"kernel": kernels, "C": cs, "gamma": gammas}
        return self

    def save_params(self, params):
        self.param_dir.mkdir(exist_ok=True)
        save_path = Path(self.param_dir) / (self.model.name + ".json")
        log.info(f"Saving parameters to: {str(save_path)}")
        io.save_json(params, save_path)

    def load_params(self):
        param_path = self.param_dir / (self.model.name + ".json")
        log.info(f"Loading parameters from: {param_path}")
        if param_path.exists():
            optimal_params = io.load_json(param_path)
            self.model.set_params(optimal_params)
        else:
            raise FileNotFoundError(
                "No param file found. Run `tune_hyperparameters` first."
            )

        return self

    def update_model_params_grid_search(self):
        if self.hyperparam_grid is None:
            raise ValueError("First select param grid!")
        else:
            param_searcher = GridSearchCV(
                estimator=self.model.classifier,
                param_grid=self.hyperparam_grid,
                scoring="roc_auc",
                cv=self.dataset.cv_splits,
                verbose=0,
                n_jobs=-1,
            )
            rs = param_searcher.fit(
                X=self.dataset.X_train, y=self.dataset.y_train
            )
            optimal_params = rs.best_params_
            log.info(
                f"Best params for {self.model.name}: \
                {optimal_params}"
            )
            self.model.set_params(optimal_params)
            self.save_params(optimal_params)
            mlflow.log_param("optimal_params", optimal_params)

            return self

    def get_hyperparam_grid(self):
        model_name = self.model.name
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
            log.info(
                "Params couldn't be loaded. Starting hyperparameter tuning..."
            )
            self.tune_hyperparameters()

        return self.model
