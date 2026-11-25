import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")

with app.setup:
    from IPython.display import display
    import marimo as mo
    import numpy as np
    import pandas as pd
    import polars as pl
    from scipy import stats as scipy_stats
    import seaborn as sns
    from teeplot import teeplot as tp
    from tqdm import tqdm
    from watermark import watermark

    from pylib._osf_fetch import osf_fetch
    from pylib.theory._calc_mutator_fixprob_closed import (
        calc_mutator_fixprob_closed,
    )
    from pylib.viz._add_legend_entry import add_legend_entry
    from pylib.viz._size_fixation_areaplot import size_fixation_areaplot


@app.cell(hide_code=True)
def do_watermark():
    mo.md(
        f"""
    ```Text
    {watermark(
        current_date=True,
        iso8601=True,
        machine=True,
        updated=True,
        python=True,
        iversions=True,
        globals_=globals(),
    )}
    ```
    """
    )
    return


@app.cell
def _():
    teeplot_subdir = "cupy-5050-traits"
    teeplot_subdir
    return (teeplot_subdir,)


@app.cell
def _():
    data = [
        {
            "url": "https://osf.io/duam2/download",
            "population structure": "well-mixed",
        },
        {
            "url": "https://osf.io/8q5v6/download",
            "population structure": "2D demes",
        },
        # {
        #     "url": "https://osf.io/s67d2/download",
        #     "population structure": "1D demes",
        # },
    ]
    dfs = []
    for item in tqdm(data):
        df = pd.read_parquet(osf_fetch(item["url"]))
        df["population structure"] = item["population structure"]
        df["available beneficial mutations"] = df["nBen"]
        df = pl.from_pandas(df)
        df = (
            df.with_columns(
                pl.when(pl.col("nRowSubgrid") == 0)
                .then(pl.col("nRow"))
                .otherwise(pl.col("nRowSubgrid"))
                .alias("nRowSubgrid"),
                pl.when(pl.col("nColSubgrid") == 0)
                .then(pl.col("nCol"))
                .otherwise(pl.col("nColSubgrid"))
                .alias("nColSubgrid"),
            )
            .with_columns(
                (pl.col("row").cast(pl.UInt64) // pl.col("nRowSubgrid")).alias(
                    "rowGroup"
                ),
                (pl.col("col").cast(pl.UInt64) // pl.col("nColSubgrid")).alias(
                    "colGroup"
                ),
            )
            .with_columns(
                (
                    pl.col("colGroup")
                    + (pl.col("colGroup").max() + 1) * pl.col("rowGroup")
                ).alias("group"),
            )
            .with_columns(
                (
                    pl.col("trait value").replace_strict(
                        {0: "nonmutator", 1: "mutator"},
                    )
                ).alias("genotype"),
            )
            .with_columns(
                (
                    pl.col("nColSubgrid").cast(pl.UInt64)
                    * pl.col("nRowSubgrid").cast(pl.UInt64)
                    * pl.col("tilePopSize")
                ).alias("population size"),
            )
            .with_columns(
                (pl.col("trait count") / pl.col("tilePopSize")).alias(
                    "fixation probability"
                ),
            )
        )
        assert (df["nCol"] > 0).all()
        assert (df["nRow"] > 0).all()
        assert (df["nColSubgrid"] > 0).all()
        assert (df["nRowSubgrid"] > 0).all()
        assert (df["tilePopSize"] > 0).all()
        assert (df["population size"] > 0).all()
        dfs.append(
            df.group_by(
                [
                    "group",
                    "genotype",
                    "replicate",
                    "population size",
                    "population structure",
                    "available beneficial mutations",
                ],
            ).agg(
                pl.col("fixation probability").mean(),
            ),
        )
    return (dfs,)


@app.cell
def _(dfs):
    dfxs = pl.concat(dfs, how="vertical_relaxed").to_pandas()
    dfxs["fixation\nprobability"] = dfxs["fixation probability"]
    dfxs["population\nstructure"] = dfxs["population structure"]
    dfxs["available\nbeneficial\nmutations"] = dfxs[
        "available beneficial mutations"
    ]

    display(dfxs.describe())
    display(dfxs.head())
    display(dfxs.tail())
    return (dfxs,)


@app.cell
def _(dfxs):
    apn = int(dfxs["population size"].min())  # agents per node
    apn
    return


@app.cell
def calc_model(dfxs):
    # math model predictions (see pylib/theory), for well-mixed populations
    # and for a 2D lattice of 256-agent demes
    model_N = np.geomspace(
        dfxs["population size"].min(), dfxs["population size"].max(), 60
    )
    model_xy = {
        float(K): [
            (
                model_N.tolist(),
                [
                    calc_mutator_fixprob_closed(N, K, deme_size=deme_size)
                    for N in model_N
                ],
            )
            for deme_size in (None, 256)
        ]
        for K in sorted(dfxs["available beneficial mutations"].unique())
        if K <= 4
    }
    return (model_xy,)


@app.cell
def _(dfxs, teeplot_subdir):
    for _errorbar in "sd", "se", "ci", None:
        with tp.teed(
            size_fixation_areaplot,
            fixprobs_df=dfxs[dfxs["available beneficial mutations"] <= 4],
            x="population size",
            y="fixation probability",
            hue="genotype",
            hue_order=["nonmutator", "mutator"],
            col="available\nbeneficial\nmutations",
            row="population\nstructure",
            style="genotype",
            style_order=["nonmutator", "mutator"],
            errorbar=_errorbar,
            height=1.65,
            facet_kws=dict(legend_out=False, margin_titles=True),
            markers=True,
            teeplot_subdir=teeplot_subdir,
        ) as _teed:
            # teed.figure.suptitle(f"n={apn} agents per node", x=0.63, y=1.06)
            _teed.set_titles(col_template="{col_var} = {col_name:.0f}")
            _teed.set_titles(row_template="{row_name}")
            _teed.set_ylabels("fixation\nprobability")
            _teed.tight_layout()
            sns.move_legend(
                _teed,
                "lower center",
                bbox_to_anchor=(0.5, -0.07),
                frameon=False,
                ncol=2,
                title=None,
            )
            for _ax in _teed.axes.flat:
                _ax.set_xticks([1e3, 1e5, 1e7])
            _teed.tight_layout()
    return


@app.cell
def _(dfxs, model_xy, teeplot_subdir):
    for _errorbar in "sd", "se", "ci", None:
        with tp.teed(
            sns.relplot,
            data=dfxs[
                (dfxs["available beneficial mutations"] <= 4)
                & (dfxs["genotype"] == "mutator")
            ],
            x="population size",
            y="fixation probability",
            hue="population\nstructure",
            col="available\nbeneficial\nmutations",
            style="population\nstructure",
            aspect=0.7,
            errorbar=_errorbar,
            height=2,
            facet_kws=dict(legend_out=False, margin_titles=True),
            kind="line",
            markers=True,
            palette="Dark2",
            teeplot_subdir=teeplot_subdir,
        ) as _teed:
            _teed.set(xscale="log")
            _teed.set(ylim=(0, 1))
            for _K, _ax in _teed.axes_dict.items():
                for _curve in model_xy[_K]:
                    _ax.plot(*_curve, color="black", linestyle="--")
            _teed.set_titles(col_template="{col_var} = {col_name:.0f}")
            _teed.set_titles(row_template="{row_name}")
            _teed.set_ylabels("mutator\nfixation\nprobability")
            _teed.tight_layout()
            sns.move_legend(
                _teed,
                "lower center",
                bbox_to_anchor=(0.5, -0.09),
                frameon=False,
                ncol=2,
                title=None,
            )
            add_legend_entry(_teed, "analytic approximation")
            for _ax in _teed.axes.flat:
                _ax.set_xticks([1e3, 1e5, 1e7])
                _ax.axhline(0.5, color="black", linestyle="--")
            _teed.tight_layout()
    return


@app.cell
def _(dfxs):
    groupby = dfxs[
        (dfxs["available beneficial mutations"] <= 4)
        & (dfxs["genotype"] == "mutator")
    ].groupby(["available beneficial mutations", "population size"])

    results = []
    for (beneficial_mut, pop_size), group in groupby:
        (g1, grp1), (g2, grp2) = group.groupby("population\nstructure")
        group_2D = grp1
        group_well_mixed = grp2

        n_g1 = grp1.shape[0]
        n_g2 = grp2.shape[0]

        stat, p_val = scipy_stats.mannwhitneyu(
            grp1["fixation probability"],
            grp2["fixation probability"],
            alternative="two-sided",
        )

        results.append(
            {
                "available beneficial mutations": beneficial_mut,
                "population size": pop_size,
                f"n {g1}": n_g1,
                f"n {g2}": n_g2,
                "U statistic": stat,
                "p-value": p_val,
                "Bonferroni corrected p-value": min(p_val * len(groupby), 1.0),
            }
        )

    results_table = pd.DataFrame(results)
    results_table
    return


if __name__ == "__main__":
    app.run()
