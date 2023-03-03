from typing import Callable

from optuna.trial import Trial


def get_param_fn(model_name) -> Callable:
    if model_name == "Random Forest":
        param_fn = params_RandomForest
    elif model_name == "XGBoost":
        param_fn = params_XGBoost
    elif model_name == "Logistic Regression":
        param_fn = params_LogReg
    elif model_name == "SVM":
        param_fn = params_SVM
    else:
        raise ValueError(
            f"Optuna parameters for {model_name} not implemented!"
        )
    return param_fn


def params_RandomForest(trial: Trial) -> dict:
    params = {
        "n_estimators": trial.suggest_int("rf_n_estimators", 50, 1000),
        "max_depth": trial.suggest_int("rf_max_depth", 2, 50),
        "max_features": trial.suggest_categorical(
            "rf_max_features", ["sqrt", "log2"]
        ),
        "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 10),
        "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 10),
        "bootstrap": trial.suggest_categorical("rf_bootstrap", [True, False]),
    }
    return params


def params_XGBoost(trial: Trial) -> dict:
    params = {
        "lambda": trial.suggest_float("xgb_lambda", 1e-8, 10.0),
        "alpha": trial.suggest_float("xgb_alpha", 1e-8, 10.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "subsample": 1.0,  # trial.suggest_float("xgb_subsample", 0.2, 1.0),
        "booster": trial.suggest_categorical(
            "xgb_booster", ["gbtree", "gblinear", "dart"]
        ),
    }
    if params["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        params["max_depth"] = trial.suggest_int("xgb_max_depth", 3, 9, step=2)
        # minimum child weight, larger the term more conservative the tree.
        params["min_child_weight"] = trial.suggest_int(
            "xgb_min_child_weight", 2, 10
        )
        params["eta"] = trial.suggest_float("xgb_eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        params["gamma"] = trial.suggest_float("xgb_gamma", 1e-8, 1.0, log=True)
        params["grow_policy"] = trial.suggest_categorical(
            "xgb_grow_policy", ["depthwise", "lossguide"]
        )

    if params["booster"] == "dart":
        params["sample_type"] = trial.suggest_categorical(
            "xgb_sample_type", ["uniform", "weighted"]
        )
        params["normalize_type"] = trial.suggest_categorical(
            "xgb_normalize_type", ["tree", "forest"]
        )
        params["rate_drop"] = trial.suggest_float(
            "xgb_rate_drop", 1e-8, 1.0, log=True
        )
        params["skip_drop"] = trial.suggest_float(
            "xgb_skip_drop", 1e-8, 1.0, log=True
        )
    return params


def params_SVM(trial: Trial) -> dict:
    params = {
        "kernel": trial.suggest_categorical(
            "svm_kernel", ["linear", "poly", "rbf", "sigmoid"]
        ),
        "C": trial.suggest_float("svm_C", 1e-3, 10.0),
        "gamma": trial.suggest_float("svm_gamma", 1e-3, 10.0),
        "degree": trial.suggest_int("svm_degree", 1, 5, 1),
    }
    return params


def params_LogReg(trial: Trial) -> dict:
    penalty = trial.suggest_categorical("lr_penalty", ["l2", "l1"])
    if penalty == "l1":
        solver = "saga"
    else:
        solver = "lbfgs"
    params = {
        "penalty": penalty,
        "C": trial.suggest_float("lr_C", 1e-3, 10.0),
        "solver": solver,
    }
    return params


def params_preprocessing(trial):
    params = {
        "oversampling_method": trial.suggest_categorical(
            "oversampling_method",
            ["placeholder"],
        ),
    }
    return params
