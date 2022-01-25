import pytest
import pandas as pd
from pathlib import Path


@pytest.fixture
def empty_df():
    df = pd.DataFrame()
    return df


@pytest.fixture
def test_df():
    df = {}
    df["binary"] = pd.DataFrame(
        {
            "id": [str(i) for i in range(100)],
            "Feature1": [0.0 for i in range(100)],
            "Label": [i % 2 for i in range(100)],
        }
    )
    df["multiclass"] = pd.DataFrame(
        {
            "id": [str(i) for i in range(100)],
            "Feature1": [100 * i for i in range(100)],
            "Label": [i % 4 for i in range(100)],
        }
    )
    return df


class Helpers:
    # for common utils, following advice from
    # https://stackoverflow.com/questions/33508060/create-and-import-helper-functions-in-tests-without-creating-packages-in-test-di
    def tmp_dir(self):
        d = Path.tmp_path
        d.mkdir()
        return d


@pytest.fixture
def helpers():
    return Helpers()
