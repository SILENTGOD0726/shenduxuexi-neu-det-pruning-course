"""Freeze the experiment matrix, evaluate held-out data, benchmark and export evidence."""
import bootstrap
import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import time
import warnings
import random

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics.yolo.utils.ops import non_max_suppression
from experiment import check_reload, evaluate, load_model, write_json
from training import prepare_legacy_runtime

ROOT = bootstrap.ROOT
LABELS = {"baseline": "Baseline", "control": "Extra training", "prune20_before": "20%: before FT",
          "prune20": "20%: after FT", "prune40_before": "40%: before FT", "prune40": "40%: after FT"}


def preprocess(image):
    resized = cv2.resize(image, (320, 320), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0).cuda().float() / 255


def predict(model, image, confidence=.25):
    with torch.inference_mode():
        output = model(preprocess(image))[0]
        boxes = non_max_suppression(output, conf_thres=confidence, iou_thres=.7, max_det=100)[0]
    return boxes.detach().cpu().numpy()


def benchmark(model, images, rounds=3, repetitions=60):
    model.eval().float()
    prepared = [preprocess(image) for image in images]
    timings = {"model_only_ms": [], "preprocess_forward_nms_ms": []}
    with torch.inference_mode():
        for i in range(20):
            model(prepared[i % len(prepared)])
        torch.cuda.synchronize()
        for _ in range(rounds):
            for i in range(repetitions):
                torch.cuda.synchronize()
                started = time.perf_counter()
                model(prepared[i % len(prepared)])
                torch.cuda.synchronize()
                timings["model_only_ms"].append((time.perf_counter() - started) * 1000)
            for i in range(repetitions):
                torch.cuda.synchronize()
                started = time.perf_counter()
                output = model(preprocess(images[i % len(images)]))[0]
                non_max_suppression(output, conf_thres=.25, iou_thres=.7, max_det=100)
                torch.cuda.synchronize()
                timings["preprocess_forward_nms_ms"].append((time.perf_counter() - started) * 1000)
    result = {key: {"median": float(np.median(values)), "mean": float(np.mean(values)),
                    "p05": float(np.percentile(values, 5)), "p95": float(np.percentile(values, 95)),
                    "sample_count": len(values)} for key, values in timings.items()}
    result["conditions"] = {"device": torch.cuda.get_device_name(0), "batch": 1, "precision": "FP32",
                            "input": [1, 3, 320, 320], "tf32": False, "fused": False,
                            "warmup": 20, "rounds": rounds, "repetitions_per_round": repetitions,
                            "synchronization": "torch.cuda.synchronize before and after each measured call",
                            "scope": "in-RAM images; includes resize/color/tensor transfer and NMS in second metric; excludes disk I/O and drawing",
                            "uncertainty": "P05/P95 reflect repeated timing variation, not independent training uncertainty"}
    return result


def cases(model, class_names, subset="test", limit=18):
    selected = []
    for image_path in sorted((ROOT / "data/yolo/images" / subset).glob("*.jpg")):
        image = cv2.imread(str(image_path))
        predicted = predict(model, image)
        labels = np.loadtxt(ROOT / "data/yolo/labels" / subset / (image_path.stem + ".txt"), ndmin=2)
        ground_truth = []
        for cls, xc, yc, w, h in labels:
            ground_truth.append([xc*320-w*160, yc*320-h*160, xc*320+w*160, yc*320+h*160, int(cls)])
        matched = set()
        true_positive = 0
        for pred in sorted(predicted, key=lambda row: -row[4]):
            best, best_iou = None, 0
            for j, target in enumerate(ground_truth):
                if j in matched or int(pred[5]) != target[4]:
                    continue
                x1, y1 = max(pred[0], target[0]), max(pred[1], target[1])
                x2, y2 = min(pred[2], target[2]), min(pred[3], target[3])
                inter = max(0, x2-x1)*max(0, y2-y1)
                area_pred = max(0, pred[2]-pred[0])*max(0, pred[3]-pred[1])
                area_gt = (target[2]-target[0])*(target[3]-target[1])
                iou = inter / max(area_pred+area_gt-inter, 1e-9)
                if iou > best_iou:
                    best, best_iou = j, iou
            if best is not None and best_iou >= .5:
                matched.add(best)
                true_positive += 1
        false_positive, false_negative = len(predicted)-true_positive, len(ground_truth)-true_positive
        selected.append({"id": image_path.stem, "class": int(labels[0, 0]),
                         "tp": true_positive, "fp": false_positive, "fn": false_negative,
                         "predictions_xyxy_conf_class_at320": predicted.tolist(),
                         "ground_truth_xyxy_class_at320": ground_truth})
    write_json(ROOT / "results/case_analysis.json", {"checkpoint_selection": "best validation mAP50-95 among finetuned configurations, seed42",
                                                    "confidence": .25, "iou_match": .5, "split": subset, "images": selected})
    # Pick successful and failed examples for each class using a stated rule.
    chosen = []
    for cls in range(6):
        same = [x for x in selected if x["class"] == cls]
        successful = sorted(same, key=lambda x: (x["fp"]+x["fn"], -x["tp"], x["id"]))
        failed = sorted(same, key=lambda x: (-(x["fp"]+x["fn"]), x["tp"], x["id"]))
        chosen.extend([successful[0], failed[0]])
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16) if Path("C:/Windows/Fonts/arial.ttf").exists() else ImageFont.load_default()
    canvas = Image.new("RGB", (1200, 6*225), "white")
    draw = ImageDraw.Draw(canvas)
    for i, item in enumerate(chosen):
        x0, y0 = (i % 4)*300, (i//4)*450
        image = Image.open(ROOT / "data/yolo/images" / subset / (item["id"]+".jpg")).convert("RGB").resize((280, 280))
        canvas.paste(image, (x0+10, y0+45))
        draw.text((x0+10, y0+5), item["id"], fill="black", font=font)
        draw.text((x0+10, y0+25), f"TP {item['tp']}  FP {item['fp']}  FN {item['fn']}", fill="black", font=font)
        for box in item["ground_truth_xyxy_class_at320"]:
            xy = [box[0]/320*280+x0+10, box[1]/320*280+y0+45, box[2]/320*280+x0+10, box[3]/320*280+y0+45]
            draw.rectangle(xy, outline="#00B050", width=2)
        for box in item["predictions_xyxy_conf_class_at320"]:
            xy = [box[0]/320*280+x0+10, box[1]/320*280+y0+45, box[2]/320*280+x0+10, box[3]/320*280+y0+45]
            draw.rectangle(xy, outline="#E24A33", width=2)
        draw.text((x0+10, y0+335), "Green: annotation", fill="#008030", font=font)
        draw.text((x0+10, y0+358), "Red: prediction", fill="#C02020", font=font)
    canvas.crop((0, 0, 1200, 3*450)).save(ROOT / "results/detection_cases.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    bootstrap.configure_torch(42)
    prepare_legacy_runtime()
    warnings.filterwarnings("ignore", category=FutureWarning)
    infos = [json.loads((ROOT/f"run_seed{seed}/final_info.json").read_text(encoding="utf-8")) for seed in args.seeds]
    if not all(x.get("matrix_complete") for x in infos):
        raise RuntimeError("All planned training conditions must finish before test evaluation")
    frozen = {"seeds": args.seeds, "conditions": list(LABELS), "baseline_epochs": infos[0]["baseline_epochs"],
              "extra_epochs": infos[0]["extra_epochs"], "selection": "All six predeclared conditions evaluated; no test-based tuning",
              "manifest_sha256": __import__('hashlib').sha256((ROOT/'results/split_manifest.csv').read_bytes()).hexdigest()}
    freeze_path = ROOT / "results/frozen_evaluation_plan.json"
    if freeze_path.exists() and json.loads(freeze_path.read_text(encoding='utf-8')) != frozen:
        raise RuntimeError("A different final test plan has already been frozen")
    write_json(freeze_path, frozen)
    all_rows = []
    latency_images = [cv2.imread(str(p)) for p in sorted((ROOT/'data/yolo/images/val').glob('*.jpg'))[::20]]
    for seed, info in zip(args.seeds, infos):
        compact = deepcopy(info)
        evaluation_order = random.Random(seed).sample(list(LABELS), len(LABELS))
        for key in evaluation_order:
            parent = key.removesuffix("_before")
            record = info["conditions"][parent]
            relative = record["before_checkpoint"] if key.endswith("_before") else record["checkpoint"]
            checkpoint = ROOT / relative
            reload_check = check_reload(checkpoint)
            model = load_model(checkpoint)
            test = evaluate(model, ROOT/'data/neu.yaml', 'test', ROOT/f'run_seed{seed}/test_{key}')
            latency = benchmark(model, latency_images)
            result = {"seed": seed, "condition": key, "checkpoint": relative,
                      "test": test, "latency": latency, "reload_check": reload_check,
                      "seeded_evaluation_order": evaluation_order}
            write_json(ROOT/f"results/seed{seed}_{key}.json", result)
            all_rows.append({"seed": seed, "condition": key, "mAP50": test['mAP50'],
                             "mAP50_95": test['mAP50_95'], "precision": test['precision'], "recall": test['recall'],
                             "parameters": test['parameters'], "macs_per_image": test['macs_per_image'],
                             "model_ms_median": latency['model_only_ms']['median'],
                             "pipeline_ms_median": latency['preprocess_forward_nms_ms']['median']})
            del model
            torch.cuda.empty_cache()
        write_json(ROOT/f"results/training_seed{seed}.json", compact)
        for stage in ['baseline','control','prune20','prune40']:
            __import__('shutil').copy2(ROOT/f"run_seed{seed}/{stage}/results.csv", ROOT/f"results/curve_seed{seed}_{stage}.csv")
    with (ROOT/'results/summary.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    summary = {}
    for key in LABELS:
        rows = [x for x in all_rows if x['condition']==key]
        summary[key] = {metric: {"mean": float(np.mean([x[metric] for x in rows])),
                                  "std": float(np.std([x[metric] for x in rows],ddof=1)) if len(rows)>1 else None}
                        for metric in ['mAP50','mAP50_95','precision','recall','parameters','macs_per_image','model_ms_median','pipeline_ms_median']}
    write_json(ROOT/'results/summary.json',{'seeds':args.seeds,'n':len(args.seeds),'conditions':summary,
                                         'std_definition':'sample standard deviation across independent training seeds; not a significance test'})
    # Selection uses validation only, and the image examples are descriptive.
    first = infos[0]
    best_key = max(['control','prune20','prune40'],key=lambda k:first['conditions'][k]['validation']['mAP50_95'])
    best = load_model(ROOT/first['conditions'][best_key]['checkpoint'])
    cases(best,best.names)
    write_json(ROOT/'results/selected_model.json',{'selection_split':'val','seed':args.seeds[0],
                                                  'condition':best_key,'checkpoint':first['conditions'][best_key]['checkpoint']})
    print('FINAL_EVALUATION_COMPLETE',flush=True)


if __name__ == '__main__':
    main()
