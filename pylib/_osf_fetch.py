import pathlib
import re
import tempfile

import requests


def osf_fetch(slug: str) -> pathlib.Path:
    """Download a file from OSF by its slug, caching it under the system
    temp directory, and return the path to the cached file.

    A full OSF download URL (e.g., "https://osf.io/abc12/download") is also
    accepted in place of a bare slug.
    """
    match = re.fullmatch(r"https?://osf\.io/(\w+)(?:/download)?/?", slug)
    if match is not None:
        slug = match.group(1)
    cache_path = pathlib.Path(tempfile.gettempdir()) / slug
    url = f"https://osf.io/{slug}/download"
    if not cache_path.exists():
        print(f"downloading {url} -> {cache_path}")
        resp = requests.get(url, allow_redirects=True, timeout=180)
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)
    else:
        print(f"reusing cached {cache_path}")
    print(f"size: {cache_path.stat().st_size} bytes")
    return cache_path
