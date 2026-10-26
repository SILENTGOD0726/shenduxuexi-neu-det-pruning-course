"""Download public resources into this project; preserve source URLs and hashes."""
import bootstrap
import hashlib
import json
import zipfile
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = bootstrap.ROOT
COMMIT = "44c7c3514d87a5e05cfb14dba5a3eeb6eb860e70"
OFFICIAL_PAGE = "https://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm"
DATA_URL = "https://drive.usercontent.google.com/download?id=1qrdZlaDi272eA79b0uCwwqPrm2Q_WI3k&export=download&confirm=t"
SOURCE_URL = f"https://codeload.github.com/ultralytics/ultralytics/zip/{COMMIT}"


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        print("REUSE", path.name, path.stat().st_size, flush=True)
    else:
        session = requests.Session()
        session.mount("https://", HTTPAdapter(max_retries=Retry(total=2, backoff_factor=1)))
        temp = path.with_suffix(path.suffix + ".part")
        with session.get(url, stream=True, timeout=(20, 120)) as response:
            response.raise_for_status()
            with temp.open("wb") as stream:
                for chunk in response.iter_content(1024 * 1024):
                    stream.write(chunk)
        temp.replace(path)
        print("DOWNLOADED", path.name, path.stat().st_size, flush=True)
    return {"url": url, "bytes": path.stat().st_size,
            "sha256_recorded_locally": hashlib.file_digest(path.open("rb"), "sha256").hexdigest()}


def extract_checked(archive, target):
    target.mkdir(parents=True, exist_ok=True)
    base = target.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            dest = (target / member.filename).resolve()
            if not dest.is_relative_to(base):
                raise ValueError("Archive path escapes the extraction directory")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Symlink in downloaded archive")
        bundle.extractall(target)


def main():
    downloads = ROOT / ".downloads"
    manifest = {"official_dataset_page": OFFICIAL_PAGE,
                "hash_note": "Locally recorded hashes; publisher SHA256 values were not supplied."}
    for name, url, target in [
        ("ultralytics_source.zip", SOURCE_URL, ROOT / ".vendor"),
        ("NEU-DET.zip", DATA_URL, ROOT / "data" / "raw"),
    ]:
        archive = downloads / name
        try:
            manifest[name] = download(url, archive)
            if not zipfile.is_zipfile(archive):
                raise ValueError(f"Response for {name} is not a ZIP archive")
            extract_checked(archive, target)
            print("EXTRACTED", name, flush=True)
        except (requests.RequestException, ValueError, zipfile.BadZipFile) as error:
            manifest[name] = {"url": url, "error": str(error)}
            print("FAILED", name, str(error), flush=True)
        (ROOT / "download_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["pretrained_weight"] = download(
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        ROOT / "weights/yolov8n.pt")
    (ROOT / "download_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
