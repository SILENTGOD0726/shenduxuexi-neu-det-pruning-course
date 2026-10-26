"""Small, process-local setup shared by the experiment entry points."""
import os
import sys
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_DLL_HANDLES = []
if os.name == "nt":
    dlls = Path(sys.base_prefix) / "DLLs"
    if dlls.is_dir():
        _DLL_HANDLES.append(os.add_dll_directory(str(dlls)))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("WANDB_DISABLED", "true")
_TEMP = ROOT / ".cache/tmp"
_TEMP.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(_TEMP)
os.environ["TMP"] = str(_TEMP)
import tempfile
tempfile.tempdir = str(_TEMP)

_SOURCE = ROOT / ".vendor/ultralytics-44c7c3514d87a5e05cfb14dba5a3eeb6eb860e70"
if _SOURCE.exists():
    sys.path.insert(0, str(_SOURCE))

_CONFIG = ROOT / ".cache/ultralytics/settings.yaml"
_FONT = Path("C:/Windows/Fonts/arial.ttf")
if _FONT.exists():
    _CONFIG.parent.mkdir(parents=True, exist_ok=True)
    if not (_CONFIG.parent / "Arial.ttf").exists():
        shutil.copy2(_FONT, _CONFIG.parent / "Arial.ttf")
if not _CONFIG.exists():
    _CONFIG.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG.write_text(json.dumps({
        "datasets_dir": str(ROOT / "data"), "weights_dir": str(ROOT / "weights"),
        "runs_dir": str(ROOT / "runs"), "uuid": "course-design-local",
        "sync": False, "api_key": "", "settings_version": "0.0.3",
    }), encoding="utf-8")


def configure_torch(seed=42):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch
