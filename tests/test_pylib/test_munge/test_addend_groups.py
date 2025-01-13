import polars as pl

from pylib.munge._addend_groups import addend_groups


def test_addend_groups_single_group_default_aggs():
    df = pl.DataFrame({"group": ["A", "A"], "value": [1, 2]})
    result = addend_groups(df, group_by="group")
    assert result.shape == (3, 2)
    assert result[-1, "group"] == "A"
    assert result[-1, "value"] == 2


def test_addend_groups_multiple_groups_default_aggs():
    df = pl.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "value": [10, 20, 30, 40],
        }
    )
    result = addend_groups(df, group_by="group")
    assert result.shape == (6, 2)
    last_a = result.filter(pl.col("group") == "A").tail(1)
    assert last_a[0, "value"] == 20
    last_b = result.filter(pl.col("group") == "B").tail(1)
    assert last_b[0, "value"] == 40


def test_addend_groups_custom_agg():
    df = pl.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "time": [1, 2, 1, 3],
            "value": [10, 20, 30, 40],
        }
    )
    custom_agg = (pl.col("time") + 1).max().alias("time")
    result = addend_groups(
        df, group_by="group", aggs=[custom_agg, pl.exclude("time").last()]
    )
    assert result.shape == (6, 3)
    last_a = result.filter(pl.col("group") == "A").tail(1)
    assert last_a[0, "time"] == 3
    assert last_a[0, "value"] == 20
    last_b = result.filter(pl.col("group") == "B").tail(1)
    assert last_b[0, "time"] == 4
    assert last_b[0, "value"] == 40


def test_addend_groups_multiple_group_columns():
    df = pl.DataFrame(
        {
            "group1": ["A", "A", "B", "B"],
            "group2": [1, 2, 1, 2],
            "value": [10, 20, 30, 40],
        }
    )
    result = addend_groups(df, group_by=["group1", "group2"])
    assert result.shape == (8, 3)
    a2_last = result.filter(
        (pl.col("group1") == "A") & (pl.col("group2") == 2)
    ).tail(1)
    assert a2_last[0, "value"] == 20


def test_addend_groups_inner_only_true():
    df = pl.DataFrame(
        {
            "group": ["A", "A", "B", "C"],
            "value": [10, 20, 30, 40],
        }
    )
    result = addend_groups(df, group_by="group", inner_only=True)
    assert result.shape == (7, 2)
    groups_count = result.group_by("group").count()
    assert groups_count.shape == (3, 2)
