import tempfile
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def empty_df():
    df = pd.DataFrame()
    return df


@pytest.fixture
def binary_df():
    return pd.DataFrame(
        {
            "id": [str(i) for i in range(100)],
            "Feature1": [0.0 for i in range(100)],
            "Label": [i % 2 for i in range(100)],
        }
    )


@pytest.fixture
def multiclass_df():
    return pd.DataFrame(
        {
            "id": [str(i) for i in range(100)],
            "Feature1": [100 * i for i in range(100)],
            "Label": [i % 4 for i in range(100)],
        }
    )


@pytest.fixture
def test_dfs():
    return [binary_df, multiclass_df]


class Helpers:
    # for common utils, following advice from
    # https://stackoverflow.com/questions/33508060/create-and-import-helper-functions-in-tests-without-creating-packages-in-test-di
    @staticmethod
    def tmp_dir():
        dirpath = tempfile.mkdtemp()
        return Path(dirpath)


@pytest.fixture
def helpers():
    return Helpers
