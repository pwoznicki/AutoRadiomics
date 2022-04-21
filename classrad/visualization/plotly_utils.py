from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from autorad.config.type_definitions import PathLike
from autorad.data.dataset import FeatureDataset
from autorad.utils.statistics import compare_groups_not_normally_distributed

from .matplotlib_utils import get_subplots_dimensions


def hide_labels(fig):
    fig.update_layout(
        coloraxis_showscale=False, margin=dict(l=0, r=0, b=0, t=0)
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)


def boxplot_by_class(
    feature_dataset: FeatureDataset,
    result_dir: PathLike,
    neg_label: str = "Negative",
    pos_label: str = "Positive",
):
    """
    Plot the distributions of the selected features by the label class.
    """
    features = feature_dataset.selected_features
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
    y_true: Sequence[int],
    y_pred_proba: Sequence[float],
    threshold: float,
    labels: Sequence[str] = ("positive", "negative"),
) -> None:
    y_proba_rel_to_thr = [(val - threshold) for val in y_pred_proba]
    prediction_colname = "predictions"
    label_colname = "label"
    df = pd.DataFrame(
        {label_colname: y_true, prediction_colname: y_proba_rel_to_thr}
    )
    df.sort_values(by=[prediction_colname, label_colname], inplace=True)
    df[label_colname] = df[label_colname].apply(
        lambda x: labels[0] if x == 1 else labels[1]
    )
    df.reset_index(inplace=True, drop=True)
    df["plotly_index"] = df.index
    fig = px.bar(
        df,
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
    return fig
