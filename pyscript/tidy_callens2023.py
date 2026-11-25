#!/usr/bin/env python
"""Tidy data from Callens et al. (2023), "Hypermutator emergence in
experimental Escherichia coli populations is stress-type dependent,"
Evolution Letters 7(4), for comparison with model predictions.

Source data are public domain (CC0) via Dryad,
https://doi.org/10.5061/dryad.hqbzkh1mr

Usage: tidy_callens2023.py <unzipped dryad dir> <output dir>
Requires openpyxl, in addition to project requirements.
"""

import datetime as dt
import re
import sys

import numpy as np
import pandas as pd

stress_names = {
    "A": "antibiotic",
    "S": "osmotic",
    "O": "osmotic",
    "L": "control",
}

# genes whose disruption is known to confer a hypermutator phenotype,
# per Callens et al. (2023)
mutator_genes = [
    "dnaQ",
    "mutS",
    "mutL",
    "mutH",
    "uvrD",
    "mutM",
    "mutY",
    "mutT",
]


def tidy_is10_pcr(path: str) -> pd.DataFrame:
    """One row per population per mismatch repair gene, with transfer
    (passage) at which IS10 insertion was first detected and first fixed."""
    records = []
    for gene, df in pd.read_excel(path, sheet_name=None, header=None).items():
        passage, date = [], []
        for label in df.iloc[1:, 0].astype(str):
            m1 = re.search(r"(?:Pl?1\s*=\s*P?|Plate 1 P|^P)(\d+)", label)
            m2 = re.search(r"(?:Pl?2\s*=\s*P?|Plate 2 P)(\d+)", label)
            d = re.search(r"(\d\d)/(\d\d)/(\d\d)", label)
            nums = [int(m.group(1)) for m in (m1, m2) if m]
            passage.append(np.mean(nums) if nums else np.nan)
            date.append(
                dt.date(
                    2000 + int(d.group(3)), int(d.group(2)), int(d.group(1))
                )
                if d
                else None
            )
        # "22/01/18" is a typo for 22/01/19
        date = [
            dt.date(2019, 1, 22) if d == dt.date(2018, 1, 22) else d
            for d in date
        ]
        # where label gives no passage number, interpolate by assay date
        known = sorted(
            (d.toordinal(), p)
            for d, p in zip(date, passage)
            if d and not np.isnan(p)
        )
        passage = [
            (
                p
                if not np.isnan(p)
                else float(np.interp(d.toordinal(), *zip(*known)))
            )
            for d, p in zip(date, passage)
        ]
        for col in range(1, df.shape[1]):
            assays = sorted(
                (
                    (p, status)
                    for p, status in zip(passage, df.iloc[1:, col])
                    if isinstance(status, str)
                ),
                key=lambda x: x[0],
            )
            if not assays:  # contaminated population, excluded from study
                continue
            detected = [p for p, status in assays if "IS" in status]
            fixed = [
                p for p, status in assays if status.strip() == f"{gene} IS"
            ]
            population = str(df.iloc[0, col])
            records.append(
                {
                    "population": population,
                    "stress": stress_names[population[0]],
                    "stress level": {"1": "20%", "2": "40%", "3": "60%"}.get(
                        population[1], ""
                    ),
                    "gene": gene[0].lower() + gene[1:],
                    "num assays": len(assays),
                    "first detected transfer": (
                        round(detected[0], 1) if detected else np.nan
                    ),
                    "first fixed transfer": (
                        round(fixed[0], 1) if fixed else np.nan
                    ),
                    "final transfer": round(assays[-1][0], 1),
                    "final status": assays[-1][1],
                },
            )
    return pd.DataFrame(records)


def tidy_populations(path: str) -> pd.DataFrame:
    """One row per sequenced population, with mutation counts and highest
    frequency reached by a candidate hypermutator allele at generation
    1,000."""
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df["freq"] = df["freq"].astype(str).str.replace(",", ".").astype(float)
    # synonymous substitutions are annotated like "A123A (GCG→GCA)"
    aa = df["annotation"].astype(str).str.extract(r"^([A-Z*])\d+([A-Z*])\s")
    df["is synonymous"] = aa[0].notna() & (aa[0] == aa[1])
    df["is mutator allele"] = (
        df["gene"].astype(str).str.strip().isin(mutator_genes)
        & ~df["is synonymous"]
    )
    records = []
    for population, group in df.groupby("population"):
        # antibiotic stress levels are labeled A400 (20%), A40 (40%), A60 (60%)
        prefix = re.match(r"^([A-Z]+)(\d*)\.", population)
        level = {"20": "20%", "400": "20%", "40": "40%", "60": "60%"}.get(
            prefix.group(2), ""
        )
        mutator = group[group["is mutator allele"]]
        records.append(
            {
                "population": population,
                "stress": stress_names[population[0]],
                "stress level": level,
                "num mutations": len(group),
                "num fixed mutations": int((group["freq"] >= 0.95).sum()),
                "mutator allele frequency": (
                    mutator["freq"].max() if len(mutator) else 0.0
                ),
                "mutator allele genes": " ".join(
                    sorted(mutator["gene"].str.strip().unique())
                ),
            },
        )
    return pd.DataFrame(records)


def tidy_optical_density(path: str) -> pd.DataFrame:
    """Mean blank-corrected OD600 after 24 h growth, by evolution medium."""
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df["stress"] = df["medium"].str[0].map(stress_names)
    df["stress level"] = (
        df["medium"]
        .str[1:]
        .map({"1": "20%", "2": "40%", "3": "60%"})
        .fillna("")
    )
    return (
        df.groupby(["stress", "stress level"])["raw - blank"]
        .agg(["mean", "std", "count"])
        .round(4)
        .rename(columns=lambda x: f"OD600 {x}")
        .reset_index()
    )


if __name__ == "__main__":
    __, src, dst = sys.argv
    tidy_is10_pcr(
        f"{src}/Timing_of_IS10_insertions_-_PCR_results.xlsx"
    ).to_csv(f"{dst}/is10_pcr.csv", index=False)
    tidy_populations(f"{src}/mutation_list_g1000.csv").to_csv(
        f"{dst}/populations.csv", index=False
    )
    tidy_optical_density(f"{src}/Compiled_OD_measures.csv").to_csv(
        f"{dst}/optical_density.csv", index=False
    )
