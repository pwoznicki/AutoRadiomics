from typing import Sequence

import pandas as pd
import plotly.express as px


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
