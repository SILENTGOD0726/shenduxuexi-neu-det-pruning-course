"""Audit the official NEU-DET archive and create a fixed, grouped YOLO split."""
import bootstrap
import argparse
import csv
import hashlib
import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = bootstrap.ROOT
NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def read_annotation(xml_path, image_path):
    tree = ET.parse(xml_path).getroot()
    with Image.open(image_path) as image:
        width, height = image.size
        gray = image.convert("L")
        pixel_hash = hashlib.sha256(gray.tobytes()).hexdigest()
        thumb = np.asarray(gray.resize((32, 32), Image.Resampling.BILINEAR), dtype=np.int16)
        small = np.asarray(gray.resize((9, 8), Image.Resampling.BILINEAR))
        dhash = int("".join("1" if v else "0" for v in (small[:, 1:] > small[:, :-1]).ravel()), 2)
    assert int(tree.findtext("size/width")) == width
    assert int(tree.findtext("size/height")) == height
    boxes = []
    for obj in tree.findall("object"):
        name = obj.findtext("name").strip()
        if name not in NAMES:
            raise ValueError(f"Unknown class {name}: {xml_path.name}")
        xyxy = [float(obj.findtext("bndbox/" + key)) for key in ("xmin", "ymin", "xmax", "ymax")]
        x1, y1, x2, y2 = xyxy
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"Invalid box {xyxy}: {xml_path.name}")
        boxes.append((NAMES.index(name), xyxy))
    if not boxes:
        raise ValueError(f"No boxes: {xml_path.name}")
    return {"id": image_path.stem, "image": image_path, "xml": xml_path,
            "width": width, "height": height, "boxes": boxes,
            "class": boxes[0][0], "pixel_sha256": pixel_hash,
            "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "dhash": dhash, "thumbnail": thumb}


def prepare(seed=2026):
    raw = ROOT / "data/raw/NEU-DET"
    images = {p.stem: p for p in (raw / "IMAGES").glob("*.jpg")}
    xmls = {p.stem: p for p in (raw / "ANNOTATIONS").glob("*.xml")}
    if images.keys() != xmls.keys():
        raise ValueError("Images and annotations do not have the same IDs")
    items = [read_annotation(xmls[key], images[key]) for key in sorted(images)]
    parent = list(range(len(items)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(b)] = find(a)

    exact, near = [], []
    for i, one in enumerate(items):
        for j in range(i):
            two = items[j]
            if one["pixel_sha256"] == two["pixel_sha256"]:
                union(i, j)
                exact.append([one["id"], two["id"]])
            elif (one["dhash"] ^ two["dhash"]).bit_count() <= 2:
                mae = float(np.abs(one["thumbnail"] - two["thumbnail"]).mean())
                if mae <= 3:
                    union(i, j)
                    near.append({"a": one["id"], "b": two["id"], "thumbnail_mae": mae})
    groups = defaultdict(list)
    for i in range(len(items)):
        groups[find(i)].append(i)
    by_class = defaultdict(list)
    for group in groups.values():
        # Multiclass/duplicate groups stay intact, using their modal class for stratification.
        cls = Counter(items[i]["class"] for i in group).most_common(1)[0][0]
        by_class[cls].append(group)
    rng = random.Random(seed)
    split = {}
    for cls, class_groups in sorted(by_class.items()):
        rng.shuffle(class_groups)
        counts = Counter()
        total = sum(map(len, class_groups))
        targets = {"train": total * .7, "val": total * .15, "test": total * .15}
        for group in class_groups:
            choice = max(targets, key=lambda s: (targets[s] - counts[s]) / max(targets[s], 1))
            for i in group:
                split[items[i]["id"]] = choice
            counts[choice] += len(group)

    data = ROOT / "data/yolo"
    exported = ROOT / "results"
    exported.mkdir(exist_ok=True)
    rows = []
    distribution = {s: Counter() for s in ("train", "val", "test")}
    box_counts = {s: Counter() for s in distribution}
    duplicate_boxes = []
    effective_counts = Counter()
    for item in items:
        subset = split[item["id"]]
        imdir, labdir = data / "images" / subset, data / "labels" / subset
        imdir.mkdir(parents=True, exist_ok=True)
        labdir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item["image"], imdir / item["image"].name)
        labels = []
        for cls, (x1, y1, x2, y2) in item["boxes"]:
            w, h = item["width"], item["height"]
            labels.append(f"{cls} {(x1+x2)/2/w:.8f} {(y1+y2)/2/h:.8f} {(x2-x1)/w:.8f} {(y2-y1)/h:.8f}")
            box_counts[subset][NAMES[cls]] += 1
        (labdir / (item["id"] + ".txt")).write_text("\n".join(labels) + "\n", encoding="utf-8")
        effective_counts[subset] += len(set(labels))
        if len(labels) != len(set(labels)):
            duplicate_boxes.append({"id": item["id"], "split": subset,
                                    "duplicate_boxes": len(labels) - len(set(labels))})
        distribution[subset][NAMES[item["class"]]] += 1
        rows.append({"id": item["id"], "split": subset, "image_class": NAMES[item["class"]],
                     "boxes": len(item["boxes"]), "width": item["width"], "height": item["height"],
                     "image_sha256": item["image_sha256"], "pixel_sha256": item["pixel_sha256"]})
    config = {"path": data.as_posix(), "train": "images/train", "val": "images/val",
              "test": "images/test", "names": dict(enumerate(NAMES))}
    (ROOT / "data/neu.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    # A small real-data split is used only for compatibility checks, never for final claims.
    for name in ("train", "val"):
        subset = [x for x in rows if x["split"] == name]
        chosen = []
        for cls in NAMES:
            chosen.extend([x for x in subset if x["image_class"] == cls][:4])
        (ROOT / f"data/smoke_{name}.txt").write_text("\n".join((data / "images" / name / (x["id"] + ".jpg")).as_posix() for x in chosen) + "\n", encoding="utf-8")
    smoke = dict(config, train=(ROOT / "data/smoke_train.txt").as_posix(), val=(ROOT / "data/smoke_val.txt").as_posix())
    (ROOT / "data/smoke.yaml").write_text(yaml.safe_dump(smoke, sort_keys=False), encoding="utf-8")
    with (exported / "split_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert all(split[a] == split[b] for a, b in exact)
    assert all(split[p["a"]] == split[p["b"]] for p in near)
    audit = {"dataset": "NEU-DET", "source": "official author Google Drive link", "image_count": len(items),
             "box_count": sum(len(x["boxes"]) for x in items), "seed": seed,
             "split_origin": "Our seeded grouped class-stratified split; not an official benchmark split",
             "image_counts": {s: dict(v) for s, v in distribution.items()},
             "box_counts": {s: dict(v) for s, v in box_counts.items()},
             "exact_duplicate_pairs": exact, "near_duplicate_pairs": near,
             "near_duplicate_rule": "dHash64 Hamming <=2 and 32x32 grayscale MAE <=3; grouped before split",
             "near_duplicate_limit": "Heuristic does not prove independence of unknown source workpieces or exclude all near duplicates",
             "cross_split_duplicate_pairs_by_defined_rule": 0,
             "coordinate_convention": "XML coordinates treated as continuous image-edge coordinates, normalized directly without +/-1 or clipping"}
    (exported / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    deduplication = {"raw_xml_boxes": audit["box_count"],
                     "effective_unique_label_boxes": sum(effective_counts.values()),
                     "raw_by_split": {s: sum(v.values()) for s, v in box_counts.items()},
                     "effective_by_split": dict(effective_counts), "files": duplicate_boxes,
                     "handling": "Original converted labels are retained; upstream dataset verifier removes exact duplicate label rows before loading"}
    (exported / "annotation_deduplication.json").write_text(json.dumps(deduplication, indent=2), encoding="utf-8")
    # Six training examples, with original annotation boxes for a visual audit.
    canvas = Image.new("RGB", (900, 380), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 17) if Path("C:/Windows/Fonts/arial.ttf").exists() else ImageFont.load_default()
    for k, cls in enumerate(NAMES):
        item = next(x for x in items if x["class"] == k and split[x["id"]] == "train")
        image = Image.open(item["image"]).convert("RGB").resize((260, 260))
        x0, y0 = 20 + (k % 3) * 300, 35 + (k // 3) * 190
        image = image.resize((155, 155))
        canvas.paste(image, (x0, y0))
        draw.text((x0, y0 - 25), cls, fill="black", font=font)
        for _, box in item["boxes"]:
            xy = [box[0] / item["width"] * 155 + x0, box[1] / item["height"] * 155 + y0,
                  box[2] / item["width"] * 155 + x0, box[3] / item["height"] * 155 + y0]
            draw.rectangle(xy, outline="red", width=2)
    canvas.save(exported / "annotation_examples.png")
    print(json.dumps({"images": len(items), "boxes": audit["box_count"],
                      "split_counts": Counter(split.values()), "exact_pairs": len(exact), "near_pairs": len(near)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    prepare(args.seed)
