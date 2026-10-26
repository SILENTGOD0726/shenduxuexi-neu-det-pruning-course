"""Read-only framework/GPU check, saved as a reproducibility record."""
import bootstrap
import importlib.metadata as metadata
import json
import platform
import shutil
import torch
from experiment import environment, write_json


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("The tested experiment requires a working NVIDIA CUDA environment")
    bootstrap.configure_torch(42)
    value = torch.ones((8, 8), device="cuda")
    if not torch.isfinite(value @ value).all():
        raise RuntimeError("CUDA computation check failed")
    info = environment()
    info["cuda_small_computation_passed"] = True
    info["disk_free_GiB_at_check"] = shutil.disk_usage(bootstrap.ROOT).free / 2**30
    write_json(bootstrap.ROOT / "results/environment.json", info)
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
