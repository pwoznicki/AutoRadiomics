from pathlib import Path
from typing import List, Optional

import lofo
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from autorad.config.type_definitions import PathLike
from autorad.data.dataset import FeatureDataset
from autorad.models.classifier import MLClassifier


def plot_feature_importance(
    dataset: FeatureDataset, model: MLClassifier, ax: plt.Axes
):
    """
    Plot importance of features for a single model
    Args:
        model [MLClassifier] - classifier
        ax (optional) - pyplot axes object
    """
    model_name = model.name
    try:
        importances = model.feature_importance()
        importance_df = pd.DataFrame(
            {
                "feature": dataset.features,
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


def plot_feature_importance_all(
    dataset: FeatureDataset,
    models: List[MLClassifier],
    result_dir: PathLike,
    title: Optional[str] = None,
):
    """
    Plot the feature importance for all models.
    """
    nrows, ncols, figsize = get_subplots_dimensions(len(models))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for i, model in enumerate(models):
        ax = fig.axes[i]
        plot_feature_importance(dataset, model, ax=ax)
    if title:
        fig.suptitle(title)
    else:
        fig.suptitle(f"Feature Importance for {dataset.task_name}")
    fig.tight_layout()
    fig.savefig(
        Path(result_dir) / "feature_importance.png",
        bbox_inches="tight",
        dpi=100,
    )
    plt.show()


def plot_lofo_importance(dataset: FeatureDataset, model: MLClassifier):
    lofo_dataset = lofo.Dataset(
        df=dataset.df,
        target=dataset.target,
        features=dataset.selected_features,
    )
    lofo_imp = lofo.LOFOImportance(
        lofo_dataset, model=model.classifier, scoring="neg_mean_squared_error"
    )
    importance_df = lofo_imp.get_importance()
    lofo.plot_importance(importance_df, figsize=(12, 12))
    plt.tight_layout()
    plt.show()


def common_roc_settings(ax, fontsize=12):
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="black", alpha=0.8)
    ax.set_xlabel("False Positive Rate", fontsize=fontsize)
    ax.set_ylabel("True Positive Rate", fontsize=fontsize)
    ax.legend(loc="lower right", fontsize="large")


def get_subplots_dimensions(n_plots):
    """
    For given number of plots returns the 'optimal' rows x columns distribution
    of subplots and figure size.
    Args:
        n_plots [int] - number of subplots to be includeed in the plot
    Returns:
        nrows [int] - suggested number of rows ncols [int] - suggested number of
        columns figsize [tuple[int, int]] - suggested figsize
    """
    if n_plots == 1:
        nrows = 1
        ncols = 1
        figsize = (12, 7)
    elif n_plots == 2:
        nrows = 1
        ncols = 2
        figsize = (13, 6)
    elif n_plots == 3:
        nrows = 1
        ncols = 3
        figsize = (20, 5)
    elif n_plots == 4:
        nrows = 2
        ncols = 2
        figsize = (14, 8)
    elif n_plots in [5, 6]:
        nrows = 2
        ncols = 3
        figsize = (20, 9)
    elif n_plots == 9:
        nrows = 3
        ncols = 3
        figsize = (18, 12)
    elif n_plots == 10:
        nrows = 2
        ncols = 5
        figsize = (20, 7)
    elif n_plots > 4:
        nrows = n_plots // 4 + 1
        ncols = 4
        figsize = (20, 7 + 5 * nrows)
    else:
        raise ValueError("Invalid number of plots")

    return nrows, ncols, figsize
