import hypothesis.extra.pandas as hpd
from hypothesis import strategies as st


def medium_df():
    feature_columns = [
        hpd.column(
            name=f"Feature_{i}",
            elements=st.floats(min_value=-1.0, max_value=1.0),
        )
        for i in range(10)
    ]
    label_column = hpd.column(name="Label", elements=st.booleans())
    all_columns = feature_columns + [label_column]
    df = hpd.data_frames(
        columns=all_columns,
        index=hpd.range_indexes(min_size=5, max_size=10),
    )

    return df


def simple_df():
    simple_df = hpd.data_frames(
        columns=[
            hpd.column(name="ID", elements=st.integers()),
            hpd.column(name="Label", elements=st.booleans()),
            hpd.column(
                name="Feature1",
                elements=st.floats(min_value=0.0, max_value=1.0),
            ),
            hpd.column(
                name="Feature2",
                elements=st.floats(min_value=-1000.0, max_value=1000.0),
            ),
        ],
        index=hpd.range_indexes(min_size=100, max_size=100),
    )
    return simple_df
