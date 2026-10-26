"""Apply one documented environment-only fix to the pinned upstream source."""
import bootstrap
import hashlib
import json


def main():
    source = bootstrap.ROOT / ".vendor/ultralytics-44c7c3514d87a5e05cfb14dba5a3eeb6eb860e70/ultralytics/yolo/utils/__init__.py"
    old = "USER_CONFIG_DIR = Path(os.getenv('YOLO_CONFIG_DIR', get_user_config_dir()))"
    new = "USER_CONFIG_DIR = Path(os.environ['YOLO_CONFIG_DIR']) if os.getenv('YOLO_CONFIG_DIR') else get_user_config_dir()"
    raw = source.read_text(encoding="utf-8")
    before = hashlib.sha256(raw.encode()).hexdigest()
    if old in raw:
        source.write_text(raw.replace(old, new, 1), encoding="utf-8")
    elif new not in raw:
        raise RuntimeError("Pinned source did not match the expected one-line patch")
    info = {"file": "ultralytics/yolo/utils/__init__.py", "patch": "lazy YOLO_CONFIG_DIR fallback",
            "reason": "Avoid eager global-directory write probing even when project-local YOLO_CONFIG_DIR is provided",
            "upstream_commit": "44c7c3514d87a5e05cfb14dba5a3eeb6eb860e70",
            "hash_before_this_invocation": before,
            "hash_after": hashlib.sha256(source.read_text(encoding='utf-8').encode()).hexdigest()}
    target = bootstrap.ROOT / "results/compatibility_patch.json"
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        target.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("COMPATIBILITY_PATCH_READY", flush=True)


if __name__ == "__main__":
    main()
