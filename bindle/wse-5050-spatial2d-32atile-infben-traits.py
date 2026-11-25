import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    from watermark import watermark

    import itertools as it

    from IPython.display import display
    import numpy as np
    import pandas as pd
    import polars as pl
    import seaborn as sns
    from teeplot import teeplot as tp
    from tqdm import tqdm

    from pylib.viz._get_nben_cmap import get_nben_cmap
    from pylib.viz._get_nben_hue_norm import get_nben_hue_norm
    from pylib.viz._size_fixation_areaplot import size_fixation_areaplot
    from pylib.viz._size_fixation_heatmap import size_fixation_heatmap
    from pylib.theory._calc_mutator_fixprob_closed import (
        calc_mutator_fixprob_closed,
    )


@app.cell(hide_code=True)
def do_watermark():
    mo.md(f"""
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
    """)
    return


@app.cell
def _():
    teeplot_subdir = "wse-5050-spatial2d-32atile-infben-traits"
    teeplot_subdir
    return (teeplot_subdir,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Prep Data
    """)
    return


@app.cell
def _():
    data = [
        {"url": "https://osf.io/p4tyb/download", "max ben muts": np.inf},
        {"url": "https://osf.io/93rdk/download", "max ben muts": 1},
        # {"url": "https://osf.io/fahu9/download", "max ben muts": 2},
        # {"url": "https://osf.io/2wxmb/download", "max ben muts": 3},
        # {"url": "https://osf.io/ra2fw/download", "max ben muts": 4},
        # {"url": "https://osf.io/6ek82/download", "max ben muts": 5},
    ]
    dfs = []
    for item in tqdm(data):
        df = pd.read_parquet(item["url"])
        df["available beneficial mutations"] = item["max ben muts"]
        df = pl.from_pandas(df)
        df = (
            df.with_columns(
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
        dfs.append(
            df.group_by(
                [
                    "group",
                    "genotype",
                    "replicate",
                    "population size",
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
    dfxs["available\nbeneficial mutations"] = dfxs[
        "available beneficial mutations"
    ]
    dfxs
    return (dfxs,)


@app.cell
def _(dfxs):
    apn = int(dfxs["population size"].min())  # agents per node
    apn
    return (apn,)


@app.cell
def calc_model(dfxs):
    # math model predictions (see pylib/theory), for a 2D lattice of demes;
    # the smallest population surveyed is a single deme
    deme_size = int(dfxs["population size"].min())
    model_N = np.geomspace(
        dfxs["population size"].min(), dfxs["population size"].max(), 60
    )
    model_xy = {
        float(K): (
            model_N.tolist(),
            [
                calc_mutator_fixprob_closed(N, K, deme_size=deme_size)
                for N in model_N
            ],
        )
        for K in sorted(dfxs["available beneficial mutations"].unique())
    }
    return (model_xy,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Size Fixation Area Plots
    """)
    return


@app.cell
def _(apn, dfxs, teeplot_subdir):
    for _errorbar in "sd", "se", "ci", None:
        tp.tee(
            size_fixation_areaplot,
            fixprobs_df=dfxs,
            x="population size",
            y="fixation probability",
            hue="genotype",
            hue_order=["nonmutator", "mutator"],
            col="available\nbeneficial mutations",
            style="genotype",
            style_order=["nonmutator", "mutator"],
            errorbar=_errorbar,
            height=2,
            markers=True,
            teeplot_outexclude="post",
            teeplot_subdir=teeplot_subdir,
            teeplot_postprocess=f"""
    teed.figure.suptitle("n={apn} agents per node", x=0.63, y=1.06)
    teed.set_titles(col_template="{{col_var}} = {{col_name:.0f}}")
    sns.move_legend(
        teed,
        "lower center",
        bbox_to_anchor=(0.25, 0.9),
        frameon=False,
        ncol=2,
        title=None,
    )
    """,
        )
    return


@app.cell
def _(apn, dfxs, model_xy, teeplot_subdir):
    for (num_abm, fixprobs_df), _errorbar in it.product(
        dfxs.groupby(["available beneficial mutations"]),
        ("sd", "se", "ci", None),
    ):
        tp.tee(
            size_fixation_areaplot,
            fixprobs_df=fixprobs_df,
            x="population size",
            y="fixation probability",
            col=None,
            hue="genotype",
            hue_order=["nonmutator", "mutator"],
            style="genotype",
            style_order=["nonmutator", "mutator"],
            aspect=1.75,
            errorbar=_errorbar,
            height=2,
            markers=True,
            teeplot_outattrs={
                "num-abm": num_abm,
            },
            teeplot_outexclude="post",
            teeplot_subdir=teeplot_subdir,
            teeplot_postprocess=f"""
    teed.figure.suptitle("n={apn} agents per node", x=0.42, y=1.06)
    """
            + f"""teed.ax.plot(
        *{model_xy[num_abm[0]]!r}, color="black", linestyle="--"
    )
    """,
        )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Size Fixation Curves
    """)
    return


@app.cell
def _(apn, dfxs, teeplot_subdir):
    for _errorbar in "sd", "se", "ci", None:
        tp.tee(
            sns.relplot,
            data=dfxs,
            x="population size",
            y="fixation probability",
            hue="genotype",
            hue_order=["nonmutator", "mutator"],
            col="available beneficial mutations",
            style="genotype",
            style_order=["nonmutator", "mutator"],
            aspect=1.5,
            col_wrap=3,
            errorbar=_errorbar,
            height=2,
            kind="line",
            markers=True,
            seed=1,
            teeplot_outexclude="post",
            teeplot_subdir=teeplot_subdir,
            teeplot_postprocess=f"""
    teed.set(xscale="log", ylim=(-0.02, 1.02))
    teed.figure.suptitle("n={apn} agents per node", x=0.63, y=1.02)
    teed.set_titles(col_template="{{col_var}} = {{col_name:.0f}}")
    sns.move_legend(
        teed,
        "lower center",
        bbox_to_anchor=(0.25, 0.97),
        frameon=False,
        ncol=2,
        title=None,
    )
    """,
        )
    return


@app.cell
def _(apn, dfxs, teeplot_subdir):
    for _errorbar in "sd", "se", "ci", None:
        tp.tee(
            sns.relplot,
            data=dfxs[dfxs["genotype"] == "mutator"],
            x="population size",
            y="fixation probability",
            hue="available beneficial mutations",
            aspect=1.5,
            height=3,
            kind="line",
            errorbar=_errorbar,
            linewidth=2,
            markers=True,
            palette=get_nben_cmap(),
            hue_norm=get_nben_hue_norm(),
            seed=1,
            teeplot_outexclude="post",
            teeplot_postprocess=f"""
    teed.set(xscale="log", ylim=(-0.02, 1.02))
    teed.figure.suptitle("n={apn} agents per node", x=0.63, y=1.02)
    teed.set_titles(col_template='{{col_var}} = {{col_name:.0f}}')
    """,
            teeplot_subdir=teeplot_subdir,
        )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Size Fixation Heatmap
    """)
    return


@app.cell
def _(dfxs, teeplot_subdir):
    tp.tee(
        size_fixation_heatmap,
        fixprobs_df=dfxs,
        genotype="mutator",
        index="available beneficial mutations",
        on="population size",
        values="fixation probability",
        teeplot_subdir=teeplot_subdir,
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Quality Check: Number Replicates Available
    """)
    return


@app.cell
def _(dfxs):
    categories = (
        pl.DataFrame(dfxs)
        .group_by(
            "population size",
        )
        .first()["population size"]
        .sort()
        .cast(pl.String)
        .to_list()
    )
    categories
    return (categories,)


@app.cell
def _(categories, dfxs):
    dfy = pl.DataFrame(dfxs).with_columns(
        pl.col("population size")
        .cast(pl.String)
        .cast(pl.Enum(categories))
        .alias("population size cat"),
    )
    display(dfy.describe()), display(dfy.head()), display(dfy.tail());
    return (dfy,)


@app.cell
def _(apn, dfy, teeplot_subdir):
    tp.tee(
        sns.displot,
        data=dfy.to_pandas(),
        x="population size cat",
        hue="genotype",
        hue_order=["nonmutator", "mutator"],
        col="available beneficial mutations",
        aspect=1.5,
        col_wrap=3,
        height=2,
        kind="hist",
        multiple="dodge",
        teeplot_outexclude="post",
        teeplot_postprocess=f"""
    teed.set(yscale='log')
    teed.figure.suptitle("n={apn} agents per node", x=0.63, y=1.02)
    teed.set_titles(col_template='{{col_var}} = {{col_name:.0f}}')
    for ax in teed.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")
    sns.move_legend(
        teed, "lower center",
        bbox_to_anchor=(0.25, 0.97),
        frameon=False,
        ncol=2,
        title=None,
    )
    """,
        teeplot_subdir=teeplot_subdir,
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Quality Check: Has Reached Fixation?
    """)
    return


@app.cell
def _(dfxs):
    dfxs_ = dfxs[dfxs["genotype"] == "mutator"].copy().reset_index(drop=True)
    return (dfxs_,)


@app.cell
def _(dfxs_):
    dfxs_["neither fixed"] = (dfxs_["fixation probability"] != 1.0) & (
        dfxs_["fixation probability"] != 0.0
    )
    any_unfixed = dfxs_["neither fixed"].any()
    any_unfixed
    return (any_unfixed,)


@app.cell
def _(any_unfixed, apn, dfxs_, teeplot_subdir):
    tp.tee(
        sns.relplot,
        data=dfxs_[dfxs_["neither fixed"]],
        x="population size",
        y="fixation probability",
        hue="genotype",
        hue_order=["nonmutator", "mutator"],
        col=[None, "available beneficial mutations"][int(any_unfixed)],
        style="genotype",
        style_order=["nonmutator", "mutator"],
        aspect=1.5,
        col_wrap=[None, 3][int(any_unfixed)],
        height=2,
        kind="scatter",
        markers=bool(any_unfixed),
        teeplot_outexclude="post",
        teeplot_postprocess=f"""
    teed.set(xscale='log', ylim=(-0.02, 1.02))
    teed.figure.suptitle("n={apn} agents per node", x=0.63, y=1.02)
    teed.set_titles(col_template='{{col_var}} = {{col_name:.0f}}')
    try:
        sns.move_legend(
            teed,
            "lower center",
            bbox_to_anchor=(0.25, 0.97),
            frameon=False,
            ncol=2,
            title=None,
        )
    except ValueError:
        pass
    """,
        teeplot_subdir=teeplot_subdir,
    )
    return


@app.cell
def _(dfxs_, teeplot_subdir):
    saveit, teed = tp.tee(
        sns.displot,
        data=dfxs_.reset_index(drop=True),
        x="population size",
        hue="neither fixed",
        hue_order=[True, False],
        col="available\nbeneficial mutations",
        col_wrap=4,
        aspect=1.3,
        height=1.8,
        multiple="fill",
        bins=dfxs_["population size"].nunique(),
        log_scale=(True, False),
        shrink=0.8,
        palette="Pastel2",
        teeplot_callback=True,
        teeplot_subdir=teeplot_subdir,
    )

    teed.set_titles(col_template="{col_var} = {col_name:.0f}")
    teed.set_ylabels("")
    # adapted from https://stackoverflow.com/a/68850867
    for ax in teed.axes.ravel():
        # add annotations
        for c in ax.containers:
            # custom label calculates percent and add an empty string so 0 value bars don't have a number
            labels = [
                f"{w * 100:0.2f}%" if (w := v.get_height()) > 0.5 else ""
                for v in c
            ]
            ax.bar_label(
                c,
                labels=labels,
                label_type="center",
                fontsize=8,
                rotation=90,
                padding=2
            )

    saveit()
    return


if __name__ == "__main__":
    app.run()
