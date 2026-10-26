"""Run real NEU-DET baseline, matched-budget controls, pruning and evaluation."""
import bootstrap
import argparse
from copy import deepcopy
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import time
import warnings
import csv

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.yolo.cfg import get_cfg
from pruning import complexity, convert_verified, prune_model
from training import CourseValidator, prepare_legacy_runtime, train

ROOT = bootstrap.ROOT


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def load_model(path):
    # Only official downloaded weights or checkpoints created by this project.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = checkpoint.get("ema") or checkpoint["model"]
    return model.float().cuda().eval()


def completed_checkpoint(path, expected_epochs):
    path = Path(path)
    if not path.exists():
        return False
    log = path.parent.parent / "results.csv"
    if not log.exists():
        raise RuntimeError(f"Checkpoint has no epoch log: {path}")
    with log.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != expected_epochs:
        raise RuntimeError(f"Incomplete training directory ({len(rows)}/{expected_epochs} epochs): {path.parent.parent}. Keep it as evidence and use a new run directory.")
    return True


def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": deepcopy(model).cpu().float(), "ema": None,
                "train_args": {"task": "detect", "imgsz": 320},
                "version": "8.0.90", "course_note": "Trusted project-generated FP32 weights"}, path)


def evaluate(model, data, split, out_dir, size=320, batch=32):
    prepare_legacy_runtime()
    validator = CourseValidator(args=get_cfg(overrides=dict(data=str(data), imgsz=size, batch=batch,
                                          device="0", half=False, workers=0, rect=False,
                                          plots=False, save_json=False, split=split,
                                          verbose=False, conf=0.001, iou=0.7, max_det=100, task="detect")),
                                save_dir=Path(out_dir))
    result = validator(model=deepcopy(model).float())
    boxes = validator.metrics.box
    observed = sorted(list(shape) for shape in validator.observed_input_shapes)
    if any(shape[-3:] != [3, size, size] for shape in observed):
        raise RuntimeError(f"Canonical evaluation did not use the declared input size: {observed}")
    metrics = {"mAP50": float(boxes.map50), "mAP50_95": float(boxes.map),
               "precision": float(boxes.mp), "recall": float(boxes.mr),
               "per_class_mAP50_95": {str(model.names[i]): float(value) for i, value in enumerate(boxes.maps)},
               "split": split, "precision_mode": "FP32", "confidence_for_AP": 0.001,
               "nms_iou": 0.7, "max_detections": 100,
               "precision_recall_operating_point": "Upstream maximum mean-F1 operating point; case images use a separate fixed confidence of 0.25"}
    metrics["observed_input_shapes"] = observed
    metrics.update(complexity(model, size))
    write_json(Path(out_dir) / "metrics.json", metrics)
    return metrics


def fixed_output(model, size=320):
    generator = torch.Generator().manual_seed(7321)
    sample = torch.rand((1, 3, size, size), generator=generator).cuda()
    model.eval().float()
    with torch.inference_mode():
        return model(sample)[0].detach().cpu().numpy()


def prepare_reload_reference(model, checkpoint):
    reference = Path(checkpoint).with_suffix(".reference.npz")
    np.savez_compressed(reference, output=fixed_output(model))
    return reference


def check_reload(checkpoint):
    model = load_model(checkpoint)
    reference = np.load(Path(checkpoint).with_suffix(".reference.npz"))["output"]
    actual = fixed_output(model)
    error = float(np.max(np.abs(actual - reference)))
    passed = bool(np.allclose(actual, reference, rtol=1e-5, atol=2e-4))
    info = {"independent_process_check": True, "maximum_absolute_output_error": error,
            "allclose": passed, "rtol": 1e-5, "atol": 2e-4,
            "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()}
    write_json(Path(checkpoint).with_suffix(".reload.json"), info)
    if not passed:
        raise RuntimeError(f"Checkpoint reload failed: {info}")
    print("RELOAD_OK " + json.dumps(info), flush=True)
    return info


def environment():
    versions = {}
    for name in ["torch", "torchvision", "numpy", "torch-pruning", "opencv-python", "Pillow", "matplotlib", "pandas", "seaborn"]:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    versions["ultralytics_source_version"] = "8.0.90"
    versions["ultralytics_source_commit"] = "44c7c3514d87a5e05cfb14dba5a3eeb6eb860e70"
    return {"packages": versions, "gpu": torch.cuda.get_device_name(0),
            "gpu_memory_GiB": torch.cuda.get_device_properties(0).total_memory / 2**30,
            "cuda_runtime": torch.version.cuda, "training_precision": "FP32",
            "upstream_in_training_validation": "FP16 EMA evaluation; canonical reported evaluation is explicit FP32",
            "python": __import__("sys").version.split()[0],
            "environment": "Project-local additional dependencies, system-site-packages PyTorch reuse"}


def smoke(args):
    out = ROOT / "run_smoke"
    data = ROOT / "data/smoke.yaml"
    start = time.perf_counter()
    trained = out / "baseline/weights/best.pt"
    if not trained.exists():
        trained = train(ROOT / "weights/yolov8n.pt", out / "baseline", data, 1, batch=8)
    model = load_model(trained)
    conversion = convert_verified(model)
    pruning = prune_model(model, .1)
    pre = evaluate(model, data, "val", out / "pruned_val", batch=8)
    tuned = out / "finetune/weights/best.pt"
    if not tuned.exists():
        tuned = train(model, out / "finetune", data, 1, batch=8, lr=.001)
    model = load_model(tuned)
    post = evaluate(model, data, "val", out / "tuned_val", batch=8)
    prepare_reload_reference(model, tuned)
    result = {"scope": "24 real training images / 24 real validation images; compatibility only",
              "conversion": conversion, "pruning": pruning, "before_finetune": pre,
              "after_finetune": post, "checkpoint": str(Path(tuned).relative_to(ROOT)),
              "environment": environment(), "wall_seconds": time.perf_counter() - start}
    write_json(out / "final_info.json", result)
    print("SMOKE_COMPLETE", flush=True)


def matrix(args):
    data = ROOT / "data/neu.yaml"
    for seed in args.seeds:
        out = ROOT / f"run_seed{seed}"
        info_path = out / "final_info.json"
        if info_path.exists():
            info = json.loads(info_path.read_text(encoding="utf-8"))
        else:
            info = {"seed": seed, "baseline_epochs": args.epochs, "extra_epochs": args.finetune_epochs,
                    "imgsz": 320, "batch": args.batch, "conditions": {}, "environment": environment(),
                    "selection": "Best checkpoint by validation fitness; test untouched until evaluate mode",
                    "training_setup": {"optimizer": "SGD", "baseline_lr": .01, "extra_lr": .001,
                                       "mosaic": 0, "flip_vertical": .5, "flip_horizontal": .5}}
        if info["baseline_epochs"] != args.epochs or info["extra_epochs"] != args.finetune_epochs:
            raise ValueError("Cannot change a saved run's training budget")
        start = time.perf_counter()
        baseline_path = out / "baseline/weights/best.pt"
        if "baseline" not in info["conditions"]:
            if not completed_checkpoint(baseline_path, args.epochs):
                baseline_path = train(ROOT / "weights/yolov8n.pt", out / "baseline", data,
                                      args.epochs, seed, batch=args.batch)
            baseline = load_model(baseline_path)
            metrics = evaluate(baseline, data, "val", out / "baseline_val", batch=args.batch)
            prepare_reload_reference(baseline, baseline_path)
            info["conditions"]["baseline"] = {"checkpoint": str(baseline_path.relative_to(ROOT)), "validation": metrics}
            write_json(info_path, info)
            del baseline
        conditions = [("control", 0.0), ("prune20", .2), ("prune40", .4)]
        for name, ratio in conditions:
            if name in info["conditions"]:
                continue
            model = load_model(baseline_path)
            conversion = convert_verified(model)
            record = {"conversion": conversion, "requested_channel_ratio": ratio}
            if ratio:
                record["pruning"] = prune_model(model, ratio)
                raw_path = out / f"{name}/pruned_before.pt"
                save_model(model, raw_path)
                record["before_checkpoint"] = str(raw_path.relative_to(ROOT))
                record["before_validation"] = evaluate(model, data, "val", out / f"{name}_before_val", batch=args.batch)
                prepare_reload_reference(model, raw_path)
            checkpoint = out / f"{name}/weights/best.pt"
            if not completed_checkpoint(checkpoint, args.finetune_epochs):
                checkpoint = train(model, out / name, data, args.finetune_epochs, seed,
                                   batch=args.batch, lr=.001)
            del model
            tuned = load_model(checkpoint)
            record["checkpoint"] = str(checkpoint.relative_to(ROOT))
            record["validation"] = evaluate(tuned, data, "val", out / f"{name}_val", batch=args.batch)
            prepare_reload_reference(tuned, checkpoint)
            info["conditions"][name] = record
            info["wall_seconds_this_invocation"] = time.perf_counter() - start
            write_json(info_path, info)
            del tuned
            torch.cuda.empty_cache()
        info["matrix_complete"] = True
        write_json(info_path, info)
        print("SEED_COMPLETE", seed, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "matrix", "reload"], default="matrix")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--finetune-epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    bootstrap.configure_torch(args.seeds[0])
    prepare_legacy_runtime()
    warnings.filterwarnings("ignore", category=FutureWarning)
    if args.mode == "smoke":
        smoke(args)
    elif args.mode == "matrix":
        matrix(args)
    else:
        if args.checkpoint is None:
            parser.error("--checkpoint is required for reload")
        check_reload(args.checkpoint)


if __name__ == "__main__":
    main()
