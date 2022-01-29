import great_expectations as ge
import pandas as pd
import typer

app = typer.Typer()


def test_paths(source_df: pd.DataFrame):
    df = ge.from_pandas(source_df)
    expected_columns = ["id", "image_path", "mask_path"]

    df.expect_columns_to_match_ordered_list(column_list=expected_columns)

    df.expect_column_values_to_be_unique(column="id")
    df.expect_column_values_to_be_unique(column="image_path")
    df.expect_column_values_to_be_unique(column="mask_path")

    df.expect_column_values_to_not_be_null(column="id")
    df.expect_column_values_to_not_be_null(column="image_path")
    df.expect_column_values_to_not_be_null(column="mask_path")

    expectation_suite = df.get_expectation_suite()
    report = df.validate(
        expectation_suite=expectation_suite, only_return_failures=True
    )

    return report
