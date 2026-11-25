import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    from watermark import watermark

    import os

    import downstream
    import imageio
    from IPython.display import display
    import numpy as np
    import pandas as pd
    import polars as pl
    import seaborn as sns
    from teeplot import teeplot as tp
    from tqdm import tqdm

    from pylib.munge._addend_groups import addend_groups
    from pylib.munge._union_upsample import union_upsample
    from pylib.munge._squeeze import squeeze_int


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
    teeplot_subdir = "wse-denovo-spatial2d-explicitsite-timeseries"
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
    dfxs = pl.from_pandas(
        pd.read_parquet("https://osf.io/fm8dt/download"),
    ).with_row_index(
        "dstream_data_id",
    )
    display(dfxs.describe()), display(dfxs.head()), display(dfxs.tail());
    return (dfxs,)


@app.cell
def _():
    def unpack_logger_timeseries_data(df: pl.DataFrame) -> pl.DataFrame:

        # sample every 64th tile
        df = df.filter((pl.col("col") % 8 == 0) & (pl.col("row") % 8 == 0))

        # unpack dstream data items from logger hex strings
        dfu = downstream.dataframe.unpack_data_packed(df)
        df = downstream.dataframe.explode_lookup_unpacked(
            dfu.with_columns(
                # fast forward loggers to at least dstream_S entries
                # (i.e., buffer filled)
                # these will be semantically-correct zero values
                dstream_T=pl.max_horizontal(
                    pl.col("dstream_T"), pl.col("dstream_S")
                ),
            ),
            value_type="uint8",
        )

        # join metadata onto unpacked data
        df = df.join(
            dfu.select(
                "dstream_data_id",
                "CEREBRASLIB_HYPERMUT_NUM_AVAIL_BEN_MUTS",
                "CEREBRASLIB_TRAITLOGGER_DILATION",
                "col",
                "replicate",
                "row",
                "tile",
            ),
            on="dstream_data_id",
        )

        # apply dilation to dstream_T and dstream_Tbar
        df = df.with_columns(
            dstream_T=pl.col("dstream_T")
            * pl.col("CEREBRASLIB_TRAITLOGGER_DILATION"),
            dstream_Tbar=pl.col("dstream_Tbar")
            * pl.col("CEREBRASLIB_TRAITLOGGER_DILATION"),
        )

        # addend time time series groups with final zero value entry
        df = addend_groups(
            df,
            group_by=["replicate", "tile"],
            aggs=(
                pl.col("CEREBRASLIB_HYPERMUT_NUM_AVAIL_BEN_MUTS")
                .unique()
                .pipe(squeeze_int),
                pl.col("col").unique().pipe(squeeze_int),
                (pl.col("dstream_Tbar") + 1).max().alias("dstream_Tbar"),
                pl.lit(0).alias("dstream_value"),
                pl.col("row").unique().pipe(squeeze_int),
            ),
            inner_only=True,
        )

        # upsample to fill in dstream_Tbar dropped by some loggers
        # prevents ampling biases from longer-running vs shorter-running loggers
        df = union_upsample(
            df,
            upsample="dstream_Tbar",
            group_by="tile",
            fill_null_ops=[
                (pl.all(), {"strategy": "forward"}),
                (pl.col("dstream_Tbar", "dstream_value"), {"strategy": "zero"}),
                (pl.all(), {"strategy": "backward"}),
            ],
        )
        assert len(df["dstream_Tbar"].value_counts()["count"].unique()) == 1

        return df
    return (unpack_logger_timeseries_data,)


@app.cell
def _(dfxs, unpack_logger_timeseries_data):
    res = []
    for _, df in tqdm(dfxs.group_by("replicate")):
        df = unpack_logger_timeseries_data(df)
        res.append(df)

    df = pl.concat(res)
    return (df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Plot Data
    """)
    return


@app.cell
def _(df):
    # take the mean of dstream_value at each dstream_Tbar across tiles
    df_mean = df.group_by("replicate", "dstream_Tbar").agg(
        pl.col("dstream_value").mean(),
        pl.col("CEREBRASLIB_HYPERMUT_NUM_AVAIL_BEN_MUTS")
        .unique()
        .pipe(squeeze_int),
    )
    return (df_mean,)


@app.cell
def _(df_mean, teeplot_subdir):
    with tp.teed(
        sns.relplot,
        data=df_mean.with_columns(
            pl.col("replicate")
            .rank(method="dense")
            .over("CEREBRASLIB_HYPERMUT_NUM_AVAIL_BEN_MUTS")
            .cast(pl.Int32)
            .alias("replicate number")
        ),
        x="dstream_Tbar",
        y="dstream_value",
        hue="replicate",
        col="replicate number",
        row="CEREBRASLIB_HYPERMUT_NUM_AVAIL_BEN_MUTS",
        alpha=0.5,
        aspect=1.5,
        errorbar=None,
        height=1.2,
        linewidth=1,
        legend=False,
        kind="line",
        marker="d",
        markersize=4,
        markers=True,
        facet_kws=dict(margin_titles=True),
        teeplot_subdir=teeplot_subdir,
        teeplot_outexclude=["marker"],
    ) as g:
        g.set(xlim=(10, 80000))
        g.set(ylim=(0.00001, 2))
        g.set_titles(
            col_template="",
            row_template="{row_name} Avail\nBen Muts",
        )
        g.set_axis_labels(x_var="Generations", y_var="Normomut\nPrevalence")
        for ax in g.axes.flat:
            ax.axhline(1, ls="--", color="black", linewidth=1, zorder=-10)
            ax.set_yscale("log")
            ax.set_xscale("log")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Animate Data
    """)
    return


@app.cell
def _(df, teeplot_subdir):
    os.makedirs(teeplot_subdir, exist_ok=True)

    for (replicate, nmut), dfr in tqdm(
        df.group_by("replicate", "CEREBRASLIB_HYPERMUT_NUM_AVAIL_BEN_MUTS"),
    ):
        frames = []
        for (dstream_Tbar,), dff in dfr.sort("dstream_Tbar").group_by(
            "dstream_Tbar", maintain_order=True
        ):

            dff = dff.with_columns(
                col_rank=pl.col("col").rank(method="dense").cast(pl.Int32) - 1,
                row_rank=pl.col("row").rank(method="dense").cast(pl.Int32) - 1,
            )

            num_rows = dff["row_rank"].max() + 1
            num_cols = dff["col_rank"].max() + 1

            frame = np.empty((num_rows, num_cols), dtype=np.uint8)
            frame[dff["row_rank"], dff["col_rank"]] = dff["dstream_value"] * 255
            frame = np.kron(frame, np.ones((4, 4), dtype=frame.dtype))
            frames.append(frame)

            outdir = f"{teeplot_subdir}/a=traitframes+nmut={nmut}+rep={replicate}"
            os.makedirs(outdir, exist_ok=True)
            outpath = f"{outdir}/dstream_Tbar={dstream_Tbar:06}+ext=.png"
            imageio.imwrite(outpath, frame.T)

        print(len(frames))
        outpath = f"{teeplot_subdir}/a=traits+nmut={nmut}+rep={replicate}+ext=.gif"
        imageio.mimsave(outpath, frames, duration=0.2)
    return



if __name__ == "__main__":
    app.run()
