import pytest
import polars as pl
import polars.testing as pt

from pylib.munge._sample_n_groups import sample_n_groups


@pytest.fixture
def single_group_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "group": ["A", "A", "B", "B", "C", "C", "C"],
            "value": [1, 2, 3, 4, 5, 6, 7],
        }
    )


@pytest.fixture
def multi_group_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "group1": ["A", "A", "A", "B", "B", "C", "C", "D", "D"],
            "group2": [1, 1, 2, 1, 2, 2, 3, 3, 3],
            "value": [10, 20, 30, 40, 50, 60, 70, 80, 90],
        }
    )


def test_sample_n_groups_single_column(single_group_df: pl.DataFrame):
    df = single_group_df
    n = 2
    result = sample_n_groups(df, group_by="group", n=n, seed=42)

    unique_groups = result.select(pl.col("group")).unique()
    assert len(unique_groups) == n

    expected_groups = unique_groups.to_series().to_list()
    expected = df.filter(pl.col("group").is_in(expected_groups))

    pt.assert_frame_equal(
        result, expected, check_column_order=False, check_row_order=False
    )


def test_sample_n_groups_multiple_columns(multi_group_df: pl.DataFrame):
    df = multi_group_df
    n = 2
    result = sample_n_groups(df, group_by=["group1", "group2"], n=n, seed=123)

    unique_groups = result.select(["group1", "group2"]).unique()
    assert len(unique_groups) == n

    filtered = df.filter(
        pl.concat_str("group1", "group2").is_in(
            unique_groups.select(pl.concat_str("group1", "group2")),
        ),
    )

    pt.assert_frame_equal(
        result, filtered, check_column_order=False, check_row_order=False
    )


def test_sample_n_groups_reproducibility(single_group_df: pl.DataFrame):
    df = single_group_df
    n = 2
    seed = 42

    result1 = sample_n_groups(df, group_by="group", n=n, seed=seed)
    result2 = sample_n_groups(df, group_by="group", n=n, seed=seed)

    pt.assert_frame_equal(
        result1, result2, check_column_order=False, check_row_order=False
    )
