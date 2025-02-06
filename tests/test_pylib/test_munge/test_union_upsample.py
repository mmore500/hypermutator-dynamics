import polars as pl
from polars.testing import assert_frame_equal

from pylib.munge._union_upsample import union_upsample


def test_upsample_union_col():
    df = pl.DataFrame(
        {
            "group": ["A", "A", "B"],
            "timestamp": [1, 2, 1],
            "value": [10, 1, 5],
            "value2": [10, 1, 5],
        }
    )

    result = union_upsample(
        df,
        upsample="timestamp",
        group_by="group",
        fill_null_ops=[(pl.col("value"), {"strategy": "forward"})],
    )

    expected = pl.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "timestamp": [1, 2, 1, 2],
            "value": [10, 1, 5, 5],
            "value2": [10, 1, 5, None],
        }
    )

    assert_frame_equal(
        result,
        expected,
        check_column_order=False,
        check_row_order=False,
    )


def test_upsample_union_all():
    df = pl.DataFrame(
        {
            "group": ["A", "A", "B"],
            "timestamp": [1, 2, 1],
            "value": [10, 1, 5],
        }
    )

    result = union_upsample(
        df,
        upsample="timestamp",
        group_by="group",
        fill_null_ops=[(pl.all(), {"strategy": "forward"})],
    )

    expected = pl.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "timestamp": [1, 2, 1, 2],
            "value": [10, 1, 5, 5],
        }
    )

    assert_frame_equal(
        result,
        expected,
        check_column_order=False,
        check_row_order=False,
    )
