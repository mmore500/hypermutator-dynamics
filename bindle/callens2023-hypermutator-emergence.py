import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")

with app.setup:
    import pathlib

    import marimo as mo
    from matplotlib import pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy import stats as scipy_stats
    import seaborn as sns
    from teeplot import teeplot as tp
    from watermark import watermark

    from pylib.theory._calc_denovo_fixprob import calc_denovo_fixprob
    from pylib.theory._calc_selection_from_num_sweeps import (
        calc_selection_from_num_sweeps,
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
    teeplot_subdir = pathlib.Path(__file__).stem
    teeplot_subdir
    return (teeplot_subdir,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Applying the *de novo* mutator model to Callens et al. (2023)

    [Callens et al. (2023)](https://doi.org/10.1093/evlett/qrad019) evolved 95 *E. coli* populations for ~1,000 generations under antibiotic stress, osmotic stress, or no stress, and tracked emergence of hypermutators.
    Hypermutators emerged more often under osmotic stress, where more loci contribute to adaptation (43, versus 23 under antibiotic stress).

    Here we ask what the well-mixed *de novo* model (`calc_denovo_fixprob`) predicts for the probability that a hypermutator has fixed by the end of that experiment.

    ## Approach

    The model's parameters were chosen for the simulations in this repository, not for this experiment, so we set as many as possible from the experiment itself.

    - **Population size** $N$: bottleneck size from optical density, times generations per transfer, times $\ln 2$.
    - **Duration**: 148 transfers at $\log_2 100 \approx 6.64$ generations each.
    - **Adaptive potential** $K$: number of loci inferred to contribute to adaptation (Callens et al. Figure 2D). None is reported for the control, so it is treated as not limiting.
    - **Model clock** $s$: the model's selection strength is set so that nonmutators complete as many selective sweeps as the median number of mutations observed to fix in populations that stayed nonmutator (`calc_selection_from_num_sweeps`).

    That leaves the supply rate of hypermutators $\mu$, which the experiment does not measure.
    We show predictions across three orders of magnitude of $\mu$ rather than fit it.
    Remaining parameters keep their defaults: beneficial mutation rate $U_b = 10^{-6}$, deleterious mutation rate $U_d = 10^{-4}$, and 100-fold mutator strength.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Prep Data

    Tables are derived from the public-domain Dryad deposit, <https://doi.org/10.5061/dryad.hqbzkh1mr>; see `data/callens2023/README.md`.
    """)
    return


@app.cell
def _():
    data_dir = pathlib.Path(__file__).parent.parent / "data" / "callens2023"
    df_populations = pd.read_csv(data_dir / "populations.csv")
    df_is10 = pd.read_csv(data_dir / "is10_pcr.csv")
    df_od = pd.read_csv(data_dir / "optical_density.csv")
    return df_is10, df_od, df_populations


@app.cell
def _(df_populations):
    df_populations
    return


@app.cell
def _(df_is10):
    df_is10[df_is10["first detected transfer"].notna()]
    return


@app.cell
def _(df_od):
    df_od
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Observed Outcomes

    A population is counted as hypermutator-fixed if a non-synonymous, indel, or insertion sequence mutation in a known mutator gene reached frequency 0.9 by generation 1,000.
    """)
    return


@app.cell
def _(df_populations):
    stresses = ["control", "antibiotic", "osmotic"]
    df_populations["is hypermutator"] = (
        df_populations["mutator allele frequency"] > 0
    )
    df_populations["is hypermutator fixed"] = (
        df_populations["mutator allele frequency"] >= 0.9
    )
    df_observed = (
        df_populations.groupby("stress")
        .agg(
            **{
                "num populations": ("population", "size"),
                "num emerged": ("is hypermutator", "sum"),
                "num fixed": ("is hypermutator fixed", "sum"),
            },
        )
        .loc[stresses]
        .reset_index()
    )
    _cis = [
        scipy_stats.binomtest(int(k), int(n)).proportion_ci(method="wilson")
        for k, n in zip(df_observed["num fixed"], df_observed["num populations"])
    ]
    df_observed["fraction fixed"] = (
        df_observed["num fixed"] / df_observed["num populations"]
    )
    df_observed["ci lo"] = [ci.low for ci in _cis]
    df_observed["ci hi"] = [ci.high for ci in _cis]
    # sweeps completed, among populations that stayed nonmutator
    df_observed["num sweeps"] = (
        df_populations[~df_populations["is hypermutator"]]
        .groupby("stress")["num fixed mutations"]
        .median()
        .loc[stresses]
        .to_numpy()
    )
    df_observed
    return df_observed, stresses


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Model Inputs
    """)
    return


@app.cell
def _(df_observed, df_od):
    n_gen = int(round(148 * np.log2(100)))
    cells_per_ml_per_od = 8e8  # typical for E. coli in rich medium
    transfer_volume_ml = 0.01
    od = df_od.assign(
        weighted=df_od["OD600 mean"] * df_od["OD600 count"],
    ).groupby("stress")[["weighted", "OD600 count"]].sum()
    df_inputs = df_observed[["stress", "num sweeps"]].copy()
    df_inputs["OD600"] = (
        (od["weighted"] / od["OD600 count"]).loc[df_inputs["stress"]].to_numpy()
    )
    df_inputs["bottleneck size"] = (
        df_inputs["OD600"] * cells_per_ml_per_od * transfer_volume_ml
    )
    df_inputs["population size"] = (
        df_inputs["bottleneck size"] * np.log2(100) * np.log(2)
    )
    # loci contributing to adaptation, per Callens et al. Figure 2D;
    # not reported for control, so taken to be not limiting
    df_inputs["available beneficial mutations"] = (
        df_inputs["stress"].map({"antibiotic": 23, "osmotic": 43}).fillna(60)
    ).astype(int)
    df_inputs["selection strength"] = [
        calc_selection_from_num_sweeps(N, n_gen, num_sweeps)
        for N, num_sweeps in zip(
            df_inputs["population size"], df_inputs["num sweeps"]
        )
    ]
    df_inputs
    return df_inputs, n_gen


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Model Predictions
    """)
    return


@app.cell
def _(df_inputs, n_gen):
    mutator_supply_rates = [1e-8, 1e-7, 1e-6]
    df_model = pd.DataFrame(
        [
            {
                "stress": row["stress"],
                "mutator supply rate": mu,
                "fraction fixed": calc_denovo_fixprob(
                    row["population size"],
                    row["available beneficial mutations"],
                    s=row["selection strength"],
                    mu=mu,
                    n_gen=n_gen,
                ),
            }
            for __, row in df_inputs.iterrows()
            for mu in mutator_supply_rates
        ],
    )
    df_model.pivot(
        index="stress", columns="mutator supply rate", values="fraction fixed"
    )
    return df_model, mutator_supply_rates


@app.cell
def _(df_inputs, mutator_supply_rates, n_gen):
    # how prediction depends on number of sweeps completed, all else equal
    _N = df_inputs["population size"].mean()
    df_curve = pd.DataFrame(
        [
            {
                "num sweeps": num_sweeps,
                "mutator supply rate": mu,
                "fraction fixed": calc_denovo_fixprob(
                    _N,
                    60,
                    s=calc_selection_from_num_sweeps(_N, n_gen, num_sweeps),
                    mu=mu,
                    n_gen=n_gen,
                ),
            }
            for num_sweeps in np.arange(2.0, 12.1, 1.0)
            for mu in mutator_supply_rates
        ],
    )
    df_curve
    return (df_curve,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Observed vs. Predicted Hypermutator Fixation
    """)
    return


@app.cell
def _(df_model, df_observed, mutator_supply_rates, stresses, teeplot_subdir):
    def plot_fixation_by_stress(
        df_observed: pd.DataFrame, df_model: pd.DataFrame
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(4.5, 3))
        xs = np.arange(len(stresses))
        ax.errorbar(
            xs,
            df_observed["fraction fixed"],
            yerr=[
                df_observed["fraction fixed"] - df_observed["ci lo"],
                df_observed["ci hi"] - df_observed["fraction fixed"],
            ],
            color=sns.color_palette("colorblind")[0],
            marker="o",
            ls="none",
            label="observed (Callens et al. 2023)",
            zorder=3,
        )
        for mu in mutator_supply_rates:
            ys = (
                df_model[df_model["mutator supply rate"] == mu]
                .set_index("stress")
                .loc[stresses, "fraction fixed"]
            )
            ax.plot(
                xs,
                ys,
                color="black",
                linestyle="--",
                label="model" if mu == mutator_supply_rates[0] else None,
            )
            ax.annotate(
                rf"$\mu = 10^{{{np.log10(mu):.0f}}}$",
                xy=(xs[-1], ys.iloc[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(stresses)
        ax.set_xlim(-0.4, len(stresses) - 0.2)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("stress")
        ax.set_ylabel("fraction populations\nhypermutator fixed")
        ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(0.0, 0.72))
        sns.despine(fig)
        fig.tight_layout()
        return fig

    tp.tee(
        plot_fixation_by_stress,
        df_observed=df_observed,
        df_model=df_model,
        teeplot_subdir=teeplot_subdir,
    )
    return


@app.cell
def _(df_curve, df_observed, mutator_supply_rates, teeplot_subdir):
    def plot_fixation_by_sweeps(
        df_observed: pd.DataFrame, df_curve: pd.DataFrame
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(4.5, 3))
        for mu in mutator_supply_rates:
            curve = df_curve[df_curve["mutator supply rate"] == mu]
            ax.plot(
                curve["num sweeps"],
                curve["fraction fixed"],
                color="black",
                linestyle="--",
                label="model" if mu == mutator_supply_rates[0] else None,
            )
            ax.annotate(
                rf"$\mu = 10^{{{np.log10(mu):.0f}}}$",
                xy=(curve["num sweeps"].iloc[-1], curve["fraction fixed"].iloc[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
            )
        palette = sns.color_palette("colorblind")
        for (__, row), color, marker in zip(
            df_observed.iterrows(), palette, "osD"
        ):
            ax.errorbar(
                [row["num sweeps"]],
                [row["fraction fixed"]],
                yerr=[
                    [row["fraction fixed"] - row["ci lo"]],
                    [row["ci hi"] - row["fraction fixed"]],
                ],
                color=color,
                marker=marker,
                ls="none",
                label=f"observed, {row['stress']}",
                zorder=3,
            )
        ax.set_xlim(1.5, 13.5)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("selective sweeps completed by nonmutators")
        ax.set_ylabel("fraction populations\nhypermutator fixed")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        sns.despine(fig)
        fig.tight_layout()
        return fig

    tp.tee(
        plot_fixation_by_sweeps,
        df_observed=df_observed,
        df_curve=df_curve,
        teeplot_subdir=teeplot_subdir,
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Timing of Emergence

    Observed: transfer at which an IS10 insertion in *mutS* or *mutL* was first detected by PCR, among insertions still present at the end of the experiment.
    Model: probability that a hypermutator lineage bound for fixation has taken the lead by a given transfer.
    Detection by PCR lags the lead, and IS10 insertion is only one route to hypermutation, so only the shapes are comparable.
    """)
    return


@app.cell
def _(df_inputs, df_is10, mutator_supply_rates, n_gen):
    _persisting = df_is10[
        df_is10["first detected transfer"].notna()
        & df_is10["final status"].str.contains("IS")
    ]
    _num_assayed = df_is10.groupby("stress")["population"].nunique()
    df_timing_observed = pd.concat(
        [
            pd.DataFrame(
                {
                    "stress": stress,
                    "transfer": np.sort(group["first detected transfer"]),
                    "fraction populations": np.arange(1, len(group) + 1)
                    / _num_assayed[stress],
                },
            )
            for stress, group in _persisting.groupby("stress")
        ],
    )
    _inputs = df_inputs.set_index("stress")
    df_timing_model = pd.DataFrame(
        [
            {
                "stress": stress,
                "mutator supply rate": mu,
                "transfer": transfer,
                "fraction populations": calc_denovo_fixprob(
                    _inputs.loc[stress, "population size"],
                    _inputs.loc[stress, "available beneficial mutations"],
                    s=_inputs.loc[stress, "selection strength"],
                    mu=mu,
                    n_gen=int(transfer * n_gen / 148),
                ),
            }
            for stress in ("antibiotic", "osmotic")
            for mu in mutator_supply_rates
            for transfer in range(20, 149, 16)
        ],
    )
    df_timing_model
    return df_timing_model, df_timing_observed


@app.cell
def _(df_timing_model, df_timing_observed, mutator_supply_rates, teeplot_subdir):
    def plot_emergence_timing(
        df_timing_observed: pd.DataFrame, df_timing_model: pd.DataFrame
    ) -> plt.Figure:
        fig, axs = plt.subplots(1, 2, figsize=(7, 2.8), sharex=True, sharey=True)
        palette = sns.color_palette("colorblind")
        for ax, stress, color in zip(
            axs, ("antibiotic", "osmotic"), palette[1:]
        ):
            for mu in mutator_supply_rates:
                model = df_timing_model[
                    (df_timing_model["stress"] == stress)
                    & (df_timing_model["mutator supply rate"] == mu)
                ]
                ax.plot(
                    model["transfer"],
                    model["fraction populations"],
                    color="black",
                    linestyle="--",
                    label="model" if mu == mutator_supply_rates[0] else None,
                )
                ax.annotate(
                    rf"$\mu = 10^{{{np.log10(mu):.0f}}}$",
                    xy=(
                        model["transfer"].iloc[-1],
                        model["fraction populations"].iloc[-1],
                    ),
                    xytext=(4, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=7,
                )
            observed = df_timing_observed[
                df_timing_observed["stress"] == stress
            ]
            ax.step(
                [0, *observed["transfer"], 148],
                [
                    0,
                    *observed["fraction populations"],
                    observed["fraction populations"].max(),
                ],
                where="post",
                color=color,
                label="observed (IS10 PCR)",
            )
            ax.set_title(f"{stress} stress", fontsize=9)
            ax.set_xlabel("transfer")
            ax.set_xlim(0, 175)
            ax.set_ylim(-0.03, 1.03)
        axs[0].set_ylabel("fraction populations\nhypermutator emerged")
        axs[0].legend(frameon=False, fontsize=8, loc="upper left")
        sns.despine(fig)
        fig.tight_layout()
        return fig

    tp.tee(
        plot_emergence_timing,
        df_timing_observed=df_timing_observed,
        df_timing_model=df_timing_model,
        teeplot_subdir=teeplot_subdir,
    )
    return


if __name__ == "__main__":
    app.run()
