import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")

with app.setup:
    from IPython.display import display
    from backstrip import backplot
    import marimo as mo
    import polars as pl
    import seaborn as sns
    from teeplot import teeplot as tp
    from watermark import watermark

    from pylib._osf_fetch import osf_fetch


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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Prep Data
    """)
    return


@app.cell
def _():
    cpu_perf = (
        pl.concat(
            [
                pl.read_parquet(
                    osf_fetch("https://osf.io/z65g3/download"),
                    use_pyarrow=True,
                ).with_columns(
                    pl.lit("CPU").alias("hardware"),
                    pl.lit("50/50 counter-based").alias("experiment design"),
                    pl.lit(256 * 81 * 81).alias("net population size"),
                ),
                pl.read_parquet(
                    osf_fetch("https://osf.io/tcv9q/download"),
                    use_pyarrow=True,
                ).with_columns(
                    pl.lit("CPU").alias("hardware"),
                    pl.lit("de novo counter-based").alias("experiment design"),
                    pl.lit(256 * 81 * 81).alias("net population size"),
                ),
            ],
        )
        .filter(pl.col("tile") == 0)
        .with_columns(
            (
                pl.col("net population size") / pl.col("tsc seconds per cycle")
            ).alias("throughput (agent-generations/sec)"),
        )
    )
    display(cpu_perf)
    display(cpu_perf.describe())
    return (cpu_perf,)


@app.cell
def _():
    cupy_perf = (
        pl.concat(
            [
                pl.read_parquet(
                    osf_fetch("https://osf.io/jz3eq/download"),
                    use_pyarrow=True,
                ).with_columns(
                    pl.lit("GPU").alias("hardware"),
                    pl.lit("50/50 counter-based").alias("experiment design"),
                    pl.lit(256 * 243 * 243).alias("net population size"),
                ),
                pl.read_parquet(
                    osf_fetch("https://osf.io/ya29p/download"),
                    use_pyarrow=True,
                ).with_columns(
                    pl.lit("GPU").alias("hardware"),
                    pl.lit("de novo counter-based").alias("experiment design"),
                    pl.lit(256 * 243 * 243).alias("net population size"),
                ),
            ],
        )
        .filter(pl.col("tile") == 0)
        .with_columns(
            (
                pl.col("net population size") / pl.col("tsc seconds per cycle")
            ).alias("throughput (agent-generations/sec)"),
        )
    )
    display(cupy_perf)
    display(cupy_perf.describe())
    return (cupy_perf,)


@app.cell
def _():
    wse_perf = pl.concat(
        [
            pl.read_parquet(
                osf_fetch("https://osf.io/h3tu4/download"),
                use_pyarrow=True,
            )
            .filter(pl.col("tile") == 749 * 993 - 1)
            .with_columns(
                pl.lit("WSE").alias("hardware"),
                pl.lit("50/50 site-explicit").alias("experiment design"),
                pl.lit(256 * 750 * 994).alias("net population size"),
            ),
            pl.read_parquet(
                osf_fetch("https://osf.io/w97nx/download"),
                use_pyarrow=True,
            )
            .filter(pl.col("tile") == 749 * 993 - 1)
            .with_columns(
                pl.lit("WSE").alias("hardware"),
                pl.lit("de novo site-explicit").alias("experiment design"),
                pl.lit(256 * 750 * 994).alias("net population size"),
            ),
            pl.read_parquet(
                osf_fetch("https://osf.io/vh9pb/download"),
                use_pyarrow=True,
            )
            .filter(pl.col("tile") == 749 * 993 - 1)
            .with_columns(
                pl.lit("WSE").alias("hardware"),
                pl.lit("50/50 counter-based").alias("experiment design"),
                pl.lit(256 * 750 * 994).alias("net population size"),
            ),
            pl.read_parquet(
                osf_fetch("https://osf.io/zp9gt/download"),
                use_pyarrow=True,
            )
            .filter(pl.col("tile") == 749 * 993 - 1)
            .with_columns(
                pl.lit("WSE").alias("hardware"),
                pl.lit("de novo counter-based").alias("experiment design"),
                pl.lit(256 * 750 * 994).alias("net population size"),
            ),
        ],
    ).with_columns(
        (
            pl.col("net population size") / pl.col("tsc seconds per cycle")
        ).alias("throughput (agent-generations/sec)"),
    )
    return (wse_perf,)


@app.cell
def _(cpu_perf, cupy_perf, wse_perf):
    df = pl.concat([cpu_perf, cupy_perf, wse_perf])
    display(df)
    display(df.describe())
    return (df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Absolute performance
    """)
    return


@app.cell
def _(df):
    with tp.teed(
        backplot,
        data=df.to_pandas(),
        col="experiment design",
        x="throughput (agent-generations/sec)",
        hue="hardware",
        aspect=2,
        col_wrap=2,
        height=1.5,
        orient="h",
        log_scale=(True, False),
        palette=sns.color_palette("Accent"),
        sharey=False,
    ) as _g:
        for ax in _g.axes.flat[2:]:
            ax.set_ylim(0.1, 0.5)
        _g.set_titles("{col_name}")
        _g.set(ylabel="")
    return


@app.cell
def _(df):
    df.group_by("hardware", "experiment design",).agg(
        pl.col("throughput (agent-generations/sec)")
        .mean()
        .alias("mean_throughput"),
        pl.col("throughput (agent-generations/sec)")
        .std()
        .alias("std_throughput"),
        pl.col("tsc cycle hertz").mean().alias("mean_speed"),
        pl.col("tsc cycle hertz").std().alias("std_speed"),
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Relative Performance
    """)
    return


@app.cell
def _(df):
    # Step 1: Filter for GPU and calculate the mean throughput per experiment design
    cpu_means = (
        df.filter(pl.col("hardware").str.contains("CPU"))
        .group_by("experiment design")
        .agg(pl.mean("throughput (agent-generations/sec)").alias("cpu_mean"))
    )
    print(cpu_means)

    # Step 2: Join the GPU mean back to the original DataFrame
    df_with_mean = df.join(cpu_means, on="experiment design", how="left")

    # Step 3: Calculate the speedup by dividing throughput by the GPU mean
    df_with_speedup = df_with_mean.with_columns(
        (
            pl.col("throughput (agent-generations/sec)") / pl.col("cpu_mean")
        ).alias("speedup")
    )

    # Show the resulting DataFrame with speedup
    df_with_speedup
    return (df_with_speedup,)


@app.cell
def _(df_with_speedup):
    with tp.teed(
        backplot,
        data=df_with_speedup.to_pandas(),
        y="experiment design",
        x="speedup",
        hue="hardware",
        aspect=2,
        height=2,
        order=["50/50 counter-based", "de novo counter-based"],
        orient="h",
        log_scale=(True, False),
        palette=sns.color_palette("Accent"),
    ) as _g:
        _g.set_titles("{col_name}")
        _g.set(ylabel="")
    return


@app.cell
def _(df_with_speedup):
    df_with_speedup.group_by("hardware", "experiment design",).agg(
        pl.col("speedup").mean().alias("mean_speedup"),
        pl.col("speedup").std().alias("std_speedup"),
    )
    return


if __name__ == "__main__":
    app.run()
