import pandas as pd
import plotly.express as px


def waterfall_binary_classification(
    df: pd.DataFrame, prediction_colname, label_colname
):
    df_to_plot = df.copy()
    df_to_plot.sort_values(by=[prediction_colname, label_colname])
    df_to_plot.reset_index(inplace=True, drop=True)
    df_to_plot["plotly_index"] = df.index
    fig = px.bar(
        df_to_plot,
        x="plotly_index",
        y=prediction_colname,
        color=label_colname,
        color_discrete_sequence=px.colors.qualitative.Pastel1,
        labels={
            prediction_colname: "Predicted score relative to threshold",
            label_colname: "Ground-truth label",
            "plotly_index": "test",
        },
    )
    fig.show()
