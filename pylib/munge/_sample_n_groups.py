import typing

import polars as pl


def sample_n_groups(
    df: pl.DataFrame,
    group_by: typing.Union[typing.Sequence[str], str],
    n: int,
    seed: typing.Optional[int] = None,
) -> pl.DataFrame:
    """Randomly sample `n` distinct groups from the given Polars DataFrame
    and return *all* rows belonging to those sampled groups.

    Parameters
    ----------
    df : pl.DataFrame
        The Polars DataFrame to sample from.
    group_by : str | list[str]
        The column(s) defining how rows are grouped.
    n : int
        The number of groups to sample.

        Must be less than the total number of distinct groups.
    seed : int, optional
        Random seed for reproducibility, by default None.

    Returns
    -------
    pl.DataFrame
        All rows from the sampled groups.
    """
    distinct_groups = df.select(group_by).unique()
    if seed is not None:
        distinct_groups = distinct_groups.sort(group_by)
    sampled_groups = distinct_groups.sample(n=n, seed=seed)

    return df.join(sampled_groups, on=group_by, how="inner")
