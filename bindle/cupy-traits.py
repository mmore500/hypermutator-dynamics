import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    from watermark import watermark

    import itertools as it

    from IPython.display import display
    from matplotlib import ticker as mpl_ticker
    import pandas as pd
    import polars as pl
    import seaborn as sns
    from teeplot import teeplot as tp

    from pylib.munge._calc_fixprobs_from_traits import calc_fixprobs_from_traits
    from pylib.theory._calc_denovo_fixprob import calc_denovo_fixprob
    from pylib.theory._calc_denovo_fixprob_demes2d import (
        calc_denovo_fixprob_demes2d,
    )
    from pylib.theory._calc_mutator_fixprob import calc_mutator_fixprob


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
    teeplot_subdir = "cupy-traits"
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
    dfxs = pl.concat(
        [
            pl.from_pandas(
                calc_fixprobs_from_traits(
                    pd.read_parquet("https://osf.io/s67d2/download"),
                ),
            ).with_columns(
                pl.lit("50/50").alias("initial conditions"),
                pl.lit("1D demes").alias("population structure"),
            ),
            pl.from_pandas(
                calc_fixprobs_from_traits(
                    pd.read_parquet("https://osf.io/8q5v6/download"),
                ),
            ).with_columns(
                pl.lit("50/50").alias("initial conditions"),
                pl.lit("2D demes").alias("population structure"),
            ),
            pl.from_pandas(
                calc_fixprobs_from_traits(
                    pd.read_parquet("https://osf.io/duam2/download"),
                ),
            ).with_columns(
                pl.lit("50/50").alias("initial conditions"),
                pl.lit("well-mixed").alias("population structure"),
            ),
            pl.from_pandas(
                calc_fixprobs_from_traits(
                    pd.read_parquet("https://osf.io/seuyf/download"),
                ),
            ).with_columns(
                pl.lit("de novo").alias("initial conditions"),
                pl.lit("1D demes").alias("population structure"),
            ),
            pl.from_pandas(
                calc_fixprobs_from_traits(
                    pd.read_parquet("https://osf.io/ag4ur/download"),
                ),
            ).with_columns(
                pl.lit("de novo").alias("initial conditions"),
                pl.lit("2D demes").alias("population structure"),
            ),
            pl.from_pandas(
                calc_fixprobs_from_traits(
                    pd.read_parquet("https://osf.io/edwbu/download"),
                ),
            ).with_columns(
                pl.lit("de novo").alias("initial conditions"),
                pl.lit("well-mixed").alias("population structure"),
            ),
        ],
    )

    display(dfxs.describe()), display(dfxs.head()), display(dfxs.tail());
    return (dfxs,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Size Fixation Cliffplot
    """)
    return


@app.cell
def calc_model(dfxs):
    # math model predictions (see pylib/theory): well-mixed models for both
    # initial conditions, plus 2D deme model for de novo
    model_grid = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 20, 24, 32, 40]
    data_K = (
        dfxs.filter(pl.col("population structure") != "1D demes")
        .to_pandas()
        .groupby(
            ["initial conditions", "population structure", "population size"]
        )["available beneficial mutations"]
        .agg(["min", "max"])
    )
    # predictions extend only as far along the x axis as the data do
    model_xy = {}
    for (_init, _structure, _N), _range in data_K.iterrows():
        _calc = {
            ("50/50", "well-mixed"): calc_mutator_fixprob,
            ("de novo", "well-mixed"): calc_denovo_fixprob,
            ("de novo", "2D demes"): calc_denovo_fixprob_demes2d,
        }.get((_init, _structure))
        if _calc is None:  # no model for 50/50 in 2D demes at large K
            continue
        _Ks = [K for K in model_grid if _range["min"] <= K <= _range["max"]]
        model_xy.setdefault(_N, []).append(
            (_Ks, [_calc(int(_N), K) for K in _Ks]),
        )
    return (model_xy,)


@app.cell
def _(dfxs, model_xy, teeplot_subdir):
    data = dfxs.filter(
        pl.col("genotype") == "hypermutator",
        ~(
            (pl.col("population size") > 1679616)
            & (pl.col("population structure") == "1D demes")
        ),
        ~(
            (pl.col("population size") <= 256)
            & (pl.col("available beneficial mutations") > 12)
            & (pl.col("population structure") == "1D demes")
        ),
    ).with_columns(
        pl.col("fixation probability").alias("fix\nprob"),
        pl.col("population structure").alias("population\nstructure"),
        pl.col("initial conditions").alias("initial\nconditions"),
    )

    row = "population size"
    for errorbar, exclude in it.product(
        ["sd", "se", "ci", None],
        ["1D demes", None],
    ):
        with tp.teed(
            sns.relplot,
            data=data.filter(
                pl.col("population structure") != pl.lit(exclude)
                if exclude is not None
                else True,
            ),
            x="available beneficial mutations",
            y="fix\nprob",
            row=row,
            row_order=sorted(
                data[row].to_pandas().unique(), reverse=True
            ),
            hue="population\nstructure",
            hue_order=[
                *filter(
                    lambda x: x != exclude,
                    ("well-mixed", "2D demes", "1D demes"),
                ),
            ],
            style="initial\nconditions",
            style_order=["de novo", "50/50"],
            aspect=10,
            errorbar=errorbar,
            # facet_kws=dict(margin_titles=True),
            height=0.8,
            kind="line",
            markers=True,
            palette="Dark2",
            seed=1,
            teeplot_subdir=teeplot_subdir,
            teeplot_outattrs=(
                {"exclude": exclude.replace(" ", "-")}
                if exclude is not None
                else {}
            ),
        ) as teed:
            for _model_N, _model_ax in teed.axes_dict.items():
                for _model_curve in model_xy.get(_model_N, []):
                    _model_ax.plot(*_model_curve, color="black", linestyle="--")
            teed.set_titles(
                col_template="",
                row_template="",
            )
            teed.set(ylim=(0, 1), xlim=(1, 40))
            sns.move_legend(
                teed,
                "lower center",
                bbox_to_anchor=(0.4, 0.97),
                frameon=False,
                ncol=7,
                title=None,
                columnspacing=0.7,
            )
            for ax in teed.axes.flat:
                ax.axhline(0.5, ls=":", c="gray", lw=1)
                ax.set_yticks([0, 0.5, 1])
                ax.set_yticklabels(["", "0.5", ""])
                ax.set_ylabel("")

            teed.tight_layout()

            teed.figure.subplots_adjust(top=0.85)
            delta = 0.001  # Small height for the dummy axis
            pos = teed.axes.flat[0].get_position()
            dummy_ax = ax.figure.add_axes(
                [
                    pos.x0 + pos.width * 1.05,
                    pos.y0 - pos.height * data[row].to_pandas().nunique() * 1.16,
                    delta,
                    pos.height * data[row].to_pandas().nunique() * 1.58,
                ],
            )
            yvals = data[row]
            dummy_ax.set_ylim(yvals.min(), yvals.max())
            dummy_ax.yaxis.set_label_position("right")
            dummy_ax.set_ylabel(row)

            dummy_ax.set_yscale("log")
            dummy_ax.yaxis.set_ticks_position("right")
            dummy_ax.yaxis.set_major_locator(
                mpl_ticker.LogLocator(base=10),
            )
            formatter = mpl_ticker.LogFormatterMathtext(base=10)
            dummy_ax.yaxis.set_major_formatter(formatter)

            # Hide unwanted spines
            dummy_ax.get_xaxis().set_visible(False)

            teed.tight_layout()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Log-scaled x axis
    """)
    return


@app.cell
def _(dfxs, model_xy, teeplot_subdir):
    _data = dfxs.filter(
        pl.col("genotype") == "hypermutator",
        ~(
            (pl.col("population size") > 1679616)
            & (pl.col("population structure") == "1D demes")
        ),
        ~(
            (pl.col("population size") <= 256)
            & (pl.col("available beneficial mutations") > 12)
            & (pl.col("population structure") == "1D demes")
        ),
    ).with_columns(
        pl.col("fixation probability").alias("fix\nprob"),
        pl.col("population structure").alias("population\nstructure"),
        pl.col("initial conditions").alias("initial\nconditions"),
    )

    _row = "population size"
    for _errorbar, _exclude in it.product(
        ["sd", "se", "ci", None],
        ["1D demes", None],
    ):
        with tp.teed(
            sns.relplot,
            data=_data.filter(
                pl.col("population structure") != pl.lit(_exclude)
                if _exclude is not None
                else True,
            ),
            x="available beneficial mutations",
            y="fix\nprob",
            row=_row,
            row_order=sorted(
                _data[_row].to_pandas().unique(), reverse=True
            ),
            hue="population\nstructure",
            hue_order=[
                *filter(
                    lambda x: x != _exclude,
                    ("well-mixed", "2D demes", "1D demes"),
                ),
            ],
            style="initial\nconditions",
            style_order=["de novo", "50/50"],
            aspect=10,
            errorbar=_errorbar,
            # facet_kws=dict(margin_titles=True),
            height=0.8,
            kind="line",
            markers=True,
            palette="Dark2",
            seed=1,
            teeplot_subdir=teeplot_subdir,
            teeplot_outattrs={
                "xscale": "log",
                **(
                    {"exclude": _exclude.replace(" ", "-")}
                    if _exclude is not None
                    else {}
                ),
            },
        ) as _teed:
            for _model_N, _model_ax in _teed.axes_dict.items():
                for _model_curve in model_xy.get(_model_N, []):
                    _model_ax.plot(*_model_curve, color="black", linestyle="--")
            _teed.set_titles(
                col_template="",
                row_template="",
            )
            _teed.set(ylim=(0, 1), xlim=(1, 40))
            sns.move_legend(
                _teed,
                "lower center",
                bbox_to_anchor=(0.4, 0.97),
                frameon=False,
                ncol=7,
                title=None,
                columnspacing=0.7,
            )
            for _ax in _teed.axes.flat:
                _ax.axhline(0.5, ls=":", c="gray", lw=1)
                _ax.set_yticks([0, 0.5, 1])
                _ax.set_yticklabels(["", "0.5", ""])
                _ax.set_ylabel("")

            _teed.tight_layout()

            _teed.figure.subplots_adjust(top=0.85)
            _teed.set(xscale="log")
            for _ax in _teed.axes.flat:
                _ax.set_xticks([1, 2, 4, 8, 16, 40])
                _ax.xaxis.set_major_formatter(mpl_ticker.ScalarFormatter())
                _ax.xaxis.set_minor_formatter(mpl_ticker.NullFormatter())
            _delta = 0.001  # Small height for the dummy axis
            _pos = _teed.axes.flat[0].get_position()
            _dummy_ax = _ax.figure.add_axes(
                [
                    _pos.x0 + _pos.width * 1.05,
                    _pos.y0 - _pos.height * _data[_row].to_pandas().nunique() * 1.16,
                    _delta,
                    _pos.height * _data[_row].to_pandas().nunique() * 1.58,
                ],
            )
            _yvals = _data[_row]
            _dummy_ax.set_ylim(_yvals.min(), _yvals.max())
            _dummy_ax.yaxis.set_label_position("right")
            _dummy_ax.set_ylabel(_row)

            _dummy_ax.set_yscale("log")
            _dummy_ax.yaxis.set_ticks_position("right")
            _dummy_ax.yaxis.set_major_locator(
                mpl_ticker.LogLocator(base=10),
            )
            _formatter = mpl_ticker.LogFormatterMathtext(base=10)
            _dummy_ax.yaxis.set_major_formatter(_formatter)

            # Hide unwanted spines
            _dummy_ax.get_xaxis().set_visible(False)

            _teed.tight_layout()
    return


if __name__ == "__main__":
    app.run()
