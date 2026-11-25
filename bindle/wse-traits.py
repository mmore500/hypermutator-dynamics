import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    from watermark import watermark

    from IPython.display import display
    from matplotlib import ticker as mpl_ticker
    import pandas as pd
    import seaborn as sns
    from teeplot import teeplot as tp
    from tqdm import tqdm
    from pylib.theory._calc_denovo_fixprob import calc_denovo_fixprob
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
    teeplot_subdir = "wse-traits"
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
    dfxs_list = []

    for _data in tqdm([
        {
            "url": "https://osf.io/x6uvs/download",
            "initial conditions": "50/50",
            "genome model": "site-explicit",
        },
        {
            "url": "https://osf.io/bhk27/download",
            "initial conditions": "50/50",
            "genome model": "counter-based",
        },
        {
            "url": "https://osf.io/4mdws/download",
            "initial conditions": "de novo",
            "genome model": "site-explicit",
        },
        {
            "url": "https://osf.io/eacvu/download",
            "initial conditions": "de novo",
            "genome model": "counter-based",
        },
    ]):
        dfx = pd.read_parquet(_data["url"])
        dfx["initial conditions"] = _data["initial conditions"]
        dfx["genome model"] = _data["genome model"]
        print("apn", dfx["population size"].min())
        print(dfx["population size"].unique())
        dfxs_list.append(dfx)
    return (dfxs_list,)


@app.cell
def _(dfxs_list):
    dfxs = pd.concat(
        dfxs_list,
        ignore_index=True,
        sort=False,
    )
    display(dfxs.describe()), display(dfxs.head()), display(dfxs.tail());
    return (dfxs,)


@app.cell
def _(dfxs):
    # filter to population sizes shared in all data
    dfxs_1 = dfxs[
        dfxs["population size"] != 190848000
    ].reset_index(drop=True)
    return (dfxs_1,)


@app.cell
def _(dfxs_1):
    apn = dfxs_1["population size"].min()  # agents per node
    apn
    return


@app.cell
def _(dfxs_1):
    dfxs_1["initial\nconditions"] = dfxs_1["initial conditions"]
    dfxs_1["genome\nmodel"] = dfxs_1["genome model"]
    dfxs_1["fix\nprob"] = dfxs_1["fixation probability"]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## initial conditions
    """)
    return


@app.cell
def calc_model(dfxs_1):
    # math model predictions for well-mixed populations (see pylib/theory)
    model_K = [*range(1, 17)]
    model_xy = {
        (init, N): (model_K, [calc(int(N), K) for K in model_K])
        for init, calc in (
            ("50/50", calc_mutator_fixprob),
            ("de novo", calc_denovo_fixprob),
        )
        for N in sorted(dfxs_1["population size"].unique())
    }
    return (model_xy,)


@app.cell
def _(dfxs_1, model_xy, teeplot_subdir):
    _row = "population size"
    for _errorbar in ["sd", "se", "ci", None]:
        for _what, _data in dfxs_1.groupby("genome model"):
            with tp.teed(
                sns.relplot,
                data=_data[
                    _data["genotype"] == "hypermutator"
                ],
                x="available beneficial mutations",
                y="fix\nprob",
                row=_row,
                row_order=sorted(
                    _data[_row].unique(), reverse=True
                ),
                hue="initial\nconditions",
                hue_order=["de novo", "50/50"],
                style="initial\nconditions",
                style_order=["de novo", "50/50"],
                aspect=8,
                errorbar=_errorbar,
                # facet_kws=dict(margin_titles=True),
                height=0.65,
                kind="line",
                markers=True,
                palette="Set2_r",
                seed=1,
                teeplot_subdir=teeplot_subdir,
                teeplot_outattrs={
                    "what": _what.replace(" ", "-").replace("/", "-"),
                },
            ) as _teed:
                if _what == "counter-based":
                    for _N, _ax in _teed.axes_dict.items():
                        for _init in "de novo", "50/50":
                            _ax.plot(*model_xy[_init, _N], color="black", linestyle="--")
                _teed.set_titles(
                    col_template="",
                    row_template="",
                )
                _teed.set(
                    ylim=(0, 1),
                    xlim=(1, 16),
                )
                sns.move_legend(
                    _teed,
                    "lower center",
                    bbox_to_anchor=(0.45, 0.97),
                    frameon=False,
                    ncol=2,
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
                _delta = 0.001  # Small height for the dummy axis
                _pos = _teed.axes.flat[0].get_position()
                _dummy_ax = _ax.figure.add_axes(
                    [
                        _pos.x0 + _pos.width * 1.1,
                        _pos.y0 - _pos.height * _data[_row].nunique() * 1.3,
                        _delta,
                        _pos.height * _data[_row].nunique() * 1.7,
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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## genome model
    """)
    return


@app.cell
def _(dfxs_1, teeplot_subdir):
    _row = "population size"
    for _errorbar in ["sd", "se", "ci", None]:
        for _what, _data in dfxs_1.groupby("initial conditions"):
            with tp.teed(
                sns.relplot,
                data=_data[
                    _data["genotype"] == "hypermutator"
                ],
                x="available beneficial mutations",
                y="fix\nprob",
                row=_row,
                row_order=sorted(
                    _data[_row].unique(), reverse=True
                ),
                hue="genome\nmodel",
                hue_order=["site-explicit", "counter-based"],
                style="genome\nmodel",
                style_order=["site-explicit", "counter-based"],
                aspect=8,
                errorbar=_errorbar,
                # facet_kws=dict(margin_titles=True),
                height=0.65,
                kind="line",
                markers=True,
                palette="Accent",
                seed=1,
                teeplot_subdir=teeplot_subdir,
                teeplot_outattrs={
                    "what": _what.replace(" ", "-").replace("/", "-"),
                },
            ) as _teed:
                _teed.set_titles(
                    col_template="",
                    row_template="",
                )
                _teed.set(
                    ylim=(0, 1),
                    xlim=(1, 16),
                )
                sns.move_legend(
                    _teed,
                    "lower center",
                    bbox_to_anchor=(0.45, 0.97),
                    frameon=False,
                    title=None,
                    ncol=2,
                    columnspacing=0.7,
                )
                for _ax in _teed.axes.flat:
                    _ax.axhline(0.5, ls=":", c="gray", lw=1)
                    _ax.set_yticks([0, 0.5, 1])
                    _ax.set_yticklabels(["", "0.5", ""])
                    _ax.set_ylabel("")

                _teed.tight_layout()

                _teed.figure.subplots_adjust(top=0.85)
                _delta = 0.001  # Small height for the dummy axis
                _pos = _teed.axes.flat[0].get_position()
                _dummy_ax = _ax.figure.add_axes(
                    [
                        _pos.x0 + _pos.width * 1.1,
                        _pos.y0 - _pos.height * _data[_row].nunique() * 1.3,
                        _delta,
                        _pos.height * _data[_row].nunique() * 1.7,
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
