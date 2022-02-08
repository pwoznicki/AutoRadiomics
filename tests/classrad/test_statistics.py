import pytest
from classrad.utils.statistics import (
    compare_age_between_groups,
    compare_gender_between_groups,
)


def test_compare_age_between_groups():
    assert compare_age_between_groups([1, 2], [1, 2]) == 1
    assert compare_age_between_groups([1, 2], [2, 1]) == 1
    with pytest.raises(ValueError):
        compare_age_between_groups([1, 2], [-1, 2])
    with pytest.raises(ValueError):
        compare_age_between_groups([1, 2], None)
    assert compare_age_between_groups([1, 1], [2, 2]) == 0


def test_compare_gender_between_groups():
    genders_bool = [1, 0, 0, 1]
    groups_bool = [1, 1, 0, 0]
    assert compare_gender_between_groups(genders_bool, groups_bool) == 1
    groups_by_gender = [1, 0, 0, 1]
    assert compare_gender_between_groups(genders_bool, groups_by_gender) < 1
    genders_str = ["M", "F", "F", "M"]
    groups_str = ["train", "train", "test", "test"]
    assert compare_gender_between_groups(genders_str, groups_str) == 1
