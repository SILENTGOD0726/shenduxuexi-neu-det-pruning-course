"""Install only pure Python wheels locally, verifying PyPI's published SHA256."""
import bootstrap
import hashlib
import json
import sys
import sysconfig
import requests
from download_resources import download, extract_checked


def main():
    site = bootstrap.ROOT / ".venv/Lib/site-packages"
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Run this script with the project's .venv interpreter")
    manifest = {}
    for name, version in [("torch-pruning", "1.6.1"), ("seaborn", "0.13.2"), ("sentry-sdk", "2.42.1")]:
        metadata = requests.get(f"https://pypi.org/pypi/{name}/{version}/json", timeout=30)
        metadata.raise_for_status()
        wheels = [x for x in metadata.json()["urls"] if x["filename"].endswith("py3-none-any.whl") or x["filename"].endswith("py2.py3-none-any.whl")]
        if len(wheels) != 1:
            raise RuntimeError(f"Expected one pure-Python wheel for {name}")
        record = wheels[0]
        path = bootstrap.ROOT / ".downloads" / record["filename"]
        info = download(record["url"], path)
        if info["sha256_recorded_locally"] != record["digests"]["sha256"]:
            raise ValueError(f"Published SHA256 mismatch: {name}")
        extract_checked(path, site)
        info["pypi_sha256_verified"] = True
        info["version"] = version
        manifest[name] = info
        print("LOCAL_WHEEL_READY", name, version, flush=True)
    (bootstrap.ROOT / "package_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
