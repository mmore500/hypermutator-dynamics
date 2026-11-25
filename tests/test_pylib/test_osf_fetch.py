import pandas as pd

import pylib


def test_osf_fetch():
    cache_path = pylib.osf_fetch("vh9pb")
    assert cache_path.exists()
    assert cache_path.stat().st_size > 0

    df = pd.read_parquet(cache_path)
    assert len(df) > 0


def test_osf_fetch_url():
    cache_path = pylib.osf_fetch("https://osf.io/vh9pb/download")
    assert cache_path == pylib.osf_fetch("vh9pb")

    df = pd.read_parquet(cache_path)
    assert len(df) > 0
