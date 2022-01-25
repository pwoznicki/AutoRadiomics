from hypothesis import strategies as st
from hypothesis.extra.pandas import data_frames, column, range_indexes


def medium_df():
    feature_columns = [
        column(
            name=f"Feature_{i}",
            elements=st.floats(min_value=0.0, max_value=1.0),
        )
        for i in range(10)
    ]
    label_column = column(name="Label", elements=st.booleans())
    all_columns = feature_columns + [label_column]
    df = data_frames(
        columns=all_columns,
        index=range_indexes(min_size=5, max_size=10),
    )
    return df


def simple_df():
    simple_df = data_frames(
        columns=[
            column(name="ID", elements=st.integers()),
            column(name="Label", elements=st.booleans()),
            column(
                name="Feature1",
                elements=st.floats(min_value=0.0, max_value=1.0),
            ),
            column(
                name="Feature2",
                elements=st.floats(min_value=-1000.0, max_value=1000.0),
            ),
        ],
        index=range_indexes(min_size=100, max_size=100),
    )
    return simple_df
