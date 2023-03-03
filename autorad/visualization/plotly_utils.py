from typing import Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from autorad.utils import roc_utils

from .matplotlib_utils import get_subplots_dimensions


def hide_labels(fig):
    fig.update_layout(
        coloraxis_showscale=False, margin=dict(l=0, r=0, b=0, t=0)
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)


def default_formatting(fig, width=800, height=800):
    fig.update_layout(width=width, height=height)
    fig.update_layout(font={"size": 40})


def hide_axis(fig):
    fig.update_xaxes(showgrid=False, showline=False, zeroline=False)
    fig.update_yaxes(showgrid=False, showline=False, zeroline=False)


def boxplot_by_class(
    X: pd.DataFrame,
    y: pd.Series,
):
    """
    Plot the distributions of the selected features by the label class.
    """
    features = X.columns
    nrows, ncols, _ = get_subplots_dimensions(len(features))
    fig = make_subplots(rows=nrows, cols=ncols)
    for i, feature in enumerate(features):
        name = feature.replace("_", " ")
        fig.add_trace(
            go.Box(x=y, y=X[feature], name=name),
            row=i // ncols + 1,
            col=i % ncols + 1,
        )
    fig.update_layout(width=1300, height=700)
    fig.update_layout(font={"size": 19})
    fig.update_layout(
        legend=dict(
            orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1
        )
    )
    return fig


def plot_roc_curve(
    y_true,
    y_pred_proba,
    figsize: Sequence[int] = (600, 600),
):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    fig = go.Figure(
        data=[
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                line=dict(color="royalblue", width=2),
                name=f"AUC={roc_auc:.2f}",
            )
        ],
        layout=go.Layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            font=dict(size=20),
            plot_bgcolor="rgba(0, 0, 0, 0)",
            width=figsize[0],
            height=figsize[1],
            legend=dict(yanchor="bottom", xanchor="right"),
            showlegend=True,
        ),
    )
    fig.add_shape(type="line", line=dict(dash="dash"), x0=0, y0=0, x1=1, y1=1)
    fig.update_xaxes(showline=True, linewidth=2, linecolor="black")
    fig.update_yaxes(showline=True, linewidth=2, linecolor="black")
    return fig


def plot_precision_recall_curve(
    y_true,
    y_pred_proba,
    figsize: Sequence[int] = (600, 600),
):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            line=dict(color="royalblue", width=2),
            name=f"AUC={auc:.2f}",
        )
    )

    fig.update_layout(
        xaxis=dict(title="Recall", showgrid=False),
        yaxis=dict(title="Precision", showgrid=False),
        font=dict(size=20),
        plot_bgcolor="rgba(0, 0, 0, 0)",
        width=figsize[0],
        height=figsize[1],
        legend=dict(yanchor="top", xanchor="right"),
        showlegend=True,
    )
    fig.add_shape(type="line", line=dict(dash="dash"), x0=0, y0=1, x1=1, y1=0)
    return fig


def plot_waterfall(
    y_true: Sequence[int],
    y_pred_proba: Sequence[float],
    threshold: float | None = None,
    labels: Sequence[str] = ("positive", "negative"),
):
    if threshold is None:
        threshold = roc_utils.get_youden_threshold(y_true, y_pred_proba)

    y_proba_rel_to_thr = [(val - threshold) for val in y_pred_proba]
    df = (
        pd.DataFrame({"y_true": y_true, "y_pred": y_proba_rel_to_thr})
        .sort_values(by=["y_pred", "y_true"])
        .replace({"y_true": {0: labels[0], 1: labels[1]}})
    )
    fig = px.bar(
        df,
        x=range(len(y_true)),  # "plotly_index",
        y="y_pred",
        color="y_true",
        color_discrete_sequence=("red", "green"),
        labels={
            "y_pred": "Predicted score relative to threshold",
            "y_true": "",
        },
        category_orders={"y_true": labels},
        width=1000,
        height=600,
    )

    fig.update_layout(
        bargap=0.0,
        bargroupgap=0.0,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        font=dict(size=16),
    )
    return fig
