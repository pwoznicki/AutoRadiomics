import pytest

from autorad.utils.statistics import compare_age_between_groups


def test_compare_age_between_groups():
    assert compare_age_between_groups([1, 2], [1, 2]) == 1
    assert compare_age_between_groups([1, 2], [2, 1]) == 1
    with pytest.raises(ValueError):
        compare_age_between_groups([1, 2], [-1, 2])
    with pytest.raises(ValueError):
        compare_age_between_groups([1, 2], None)
    assert compare_age_between_groups([1, 1], [2, 2]) == 0
