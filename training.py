"""Preserve pruned module structures and FP32 checkpoint weights during training."""
import bootstrap
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import torch
from tqdm import tqdm as original_tqdm
from ultralytics.yolo.v8.detect.train import DetectionTrainer
from ultralytics.yolo.v8.detect.val import DetectionValidator
from ultralytics.yolo.utils.torch_utils import de_parallel


def quiet_tqdm(*args, **kwargs):
    kwargs["disable"] = True
    return original_tqdm(*args, **kwargs)


class CourseValidator(DetectionValidator):
    # Old upstream image annotation relies on removed Pillow APIs. Our own
    # evidence renderer draws predictions after canonical evaluation instead.
    def preprocess(self, batch):
        prepared = super().preprocess(batch)
        if not hasattr(self, "observed_input_shapes"):
            self.observed_input_shapes = set()
        self.observed_input_shapes.add(tuple(prepared["img"].shape))
        return prepared

    def plot_val_samples(self, batch, ni):
        return None

    def plot_predictions(self, batch, preds, ni):
        return None


class CourseTrainer(DetectionTrainer):
    def save_model(self):
        model = self.ema.ema if self.ema else de_parallel(self.model)
        checkpoint = {
            "epoch": self.epoch, "best_fitness": self.best_fitness,
            "model": deepcopy(model).cpu().float(), "ema": None,
            "optimizer": None, "updates": self.ema.updates if self.ema else 0,
            "train_args": vars(self.args), "date": datetime.now(timezone.utc).isoformat(),
            "version": "8.0.90", "course_note": "FP32 model-only checkpoint, not optimizer-resumable",
        }
        for file in [self.last] + ([self.best] if self.best_fitness == self.fitness else []):
            temp = file.with_suffix(".tmp")
            torch.save(checkpoint, temp)
            temp.replace(file)

    def final_eval(self):
        # Canonical final evaluation is explicit, FP32, and saved to JSON.
        return None

    def get_validator(self):
        from copy import copy
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return CourseValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def plot_training_samples(self, batch, ni):
        return None

    def plot_training_labels(self):
        return None

    def plot_metrics(self):
        return None


def prepare_legacy_runtime():
    import numpy as np
    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid
    # Quiet the per-batch bars; per-epoch evidence is still printed and saved.
    import ultralytics.yolo.engine.trainer as train_module
    import ultralytics.yolo.engine.validator as val_module
    train_module.tqdm = quiet_tqdm
    val_module.tqdm = quiet_tqdm


def train(model_or_path, out_dir, data, epochs, seed=42, imgsz=320, batch=32, lr=0.01):
    prepare_legacy_runtime()
    bootstrap.configure_torch(seed)
    out_dir = Path(out_dir)
    if (out_dir / "weights/best.pt").exists():
        raise FileExistsError(f"Completed training directory already exists: {out_dir}")
    overrides = dict(model=str(bootstrap.ROOT / "weights/yolov8n.pt"), data=str(data),
                     epochs=epochs, imgsz=imgsz, batch=batch, workers=0, device=0,
                     optimizer="SGD", lr0=lr, lrf=0.01, momentum=0.937,
                     weight_decay=0.0005, warmup_epochs=min(3, epochs / 5),
                     seed=seed, deterministic=True, amp=False, cache=False,
                     project=str(out_dir.parent), name=out_dir.name, exist_ok=True,
                     plots=False, save=True, patience=epochs + 1,
                     mosaic=0.0, mixup=0.0, close_mosaic=0, hsv_h=0.0,
                     hsv_s=0.0, hsv_v=0.2, translate=0.05, scale=0.1,
                     fliplr=0.5, flipud=0.5, verbose=False)
    if isinstance(model_or_path, (str, Path)):
        overrides["model"] = str(model_or_path)
    trainer = CourseTrainer(overrides=overrides)
    if isinstance(model_or_path, torch.nn.Module):
        trainer.model = model_or_path.float()
        for parameter in trainer.model.parameters():
            parameter.requires_grad_(True)

    def report_epoch(current):
        stats = {"stage": out_dir.name, "epoch": current.epoch + 1,
                 "epochs": epochs, "metrics": {k: float(v) for k, v in current.metrics.items()}}
        print("COURSE_EPOCH " + json.dumps(stats), flush=True)

    trainer.add_callback("on_fit_epoch_end", report_epoch)
    trainer.train()
    return trainer.best
