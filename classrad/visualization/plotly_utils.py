from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from classrad.config.type_definitions import PathLike
from classrad.data.dataset import FeatureDataset
from classrad.utils.statistics import compare_groups_not_normally_distributed

from .matplotlib_utils import get_subplots_dimensions


def boxplot_by_class(
    feature_dataset: FeatureDataset,
    result_dir: PathLike,
    neg_label: str = "Negative",
    pos_label: str = "Positive",
):
    """
    Plot the distributions of the selected features by the label class.
    """
    features = feature_dataset.best_features
    if features is None:
        raise ValueError("No features selected")
    nrows, ncols, figsize = get_subplots_dimensions(len(features))
    fig = make_subplots(rows=nrows, cols=ncols)
    xlabels = [
        pos_label if label == 1 else neg_label
        for label in feature_dataset.data.y_test
    ]
    xlabels = np.array(xlabels)
    # X_test = self.inverse_standardize(self.X_test)
    for i, feature in enumerate(features):
        y = feature_dataset.data.X_test[feature]
        _, p_val = compare_groups_not_normally_distributed(
            y[xlabels == neg_label].tolist(), y[xlabels == pos_label].tolist()
        )
        fig.add_trace(
            go.Box(y=y, x=xlabels, name=f"{feature} p={p_val}"),
            row=i // ncols + 1,
            col=i % ncols + 1,
        )
    fig.update_layout(title_text="Selected features:")
    fig.show()
    fig.write_html(Path(result_dir) / "boxplot.html")
    return fig


def waterfall_binary_classification(
    df: pd.DataFrame,
    prediction_colname: str,
    label_colname: str,
    labels: Sequence[str] = ("positive", "negative"),
) -> None:
    df_to_plot = df.copy()
    df_to_plot.sort_values(
        by=[prediction_colname, label_colname], inplace=True
    )
    df_to_plot[label_colname] = df_to_plot[label_colname].apply(
        lambda x: labels[0] if x == 1 else labels[1]
    )
    df_to_plot.reset_index(inplace=True, drop=True)
    df_to_plot["plotly_index"] = df_to_plot.index
    fig = px.bar(
        df_to_plot,
        x="plotly_index",
        y=prediction_colname,
        color=label_colname,
        color_discrete_sequence=px.colors.qualitative.T10,
        labels={
            prediction_colname: "Predicted score relative to threshold",
            label_colname: "",
            "plotly_index": "",
        },
        category_orders={label_colname: labels},
        width=1000,
        height=600,
    )

    fig.update_layout(
        bargap=0.0,
        bargroupgap=0.0,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        font=dict(size=20),
    )
    fig.show()
