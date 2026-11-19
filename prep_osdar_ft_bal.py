#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Class-balanced k-shot data preparation for OSDaR23.

This is a variant of ``prep_osdar_ft.py`` with a label-only balanced frame
selection.  It selects frames that cover the rare railway classes
(signal/switch/train/pole) and then fills the budget with diverse frames.

Usage:
  python prep_osdar_ft_bal.py --seq 1.2 --k 30 --out .../data/osdar_ft/k30bal
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw

OSDAR_TO_RS = {
    "track": ["rail-track"], "train": ["on-rails"], "road_vehicle": ["car", "truck"],
    "signal_pole": ["pole"], "catenary_pole": ["pole"], "signal": ["traffic-light"],
    "buffer_stop": ["construction"], "crowd": ["human"], "person": ["human"],
    "switch": ["rail-raised"],
}
NAMES19 = ["road", "sidewalk", "construction", "rail-track", "fence", "pole",
           "traffic-light", "traffic-sign", "vegetation", "terrain", "sky",
           "human", "rail-raised", "car", "truck", "rail-embedded", "on-rails",
           "tram-track", "trackbed"]
NAME2ID = {n: i for i, n in enumerate(NAMES19)}
CROP_W, CROP_H = 2056, 1252
CRITICAL = ["signal", "switch", "train", "track", "catenary_pole"]


def parse_poly2d(val):
    return [(float(val[i]), float(val[i + 1])) for i in range(0, len(val) - 1, 2)]


def frame_image_map(fr, img_dir):
    st = fr.get("frame_properties", {}).get("streams", {}).get("rgb_highres_center", {})
    uri = st.get("uri")
    if uri:
        p = os.path.join(img_dir, os.path.basename(uri))
        if os.path.isfile(p):
            return p
    return None


def frame_stats(fr, objs_meta, W, H):
    """Per-class polygon area proxy and centroid for one frame."""
    stats = defaultdict(lambda: {"area": 0.0, "cx": 0.0, "cy": 0.0, "n": 0.0})
    for uid, o in fr.get("objects", {}).items():
        otype = objs_meta.get(uid, {}).get("type")
        if otype not in OSDAR_TO_RS:
            continue
        for e in o.get("object_data", {}).get("poly2d", []):
            if not e.get("name", "").startswith("rgb_highres_center"):
                continue
            v = e.get("val") or []
            if len(v) < 6:
                continue
            pts = parse_poly2d(v)
            xs = [min(max(p[0], 0), W - 1) for p in pts]
            ys = [min(max(p[1], 0), H - 1) for p in pts]
            area = max(0.0, (max(xs) - min(xs)) * (max(ys) - min(ys)))
            s = stats[otype]
            s["area"] += area
            s["cx"] += (max(xs) + min(xs)) / 2 * max(area, 1.0)
            s["cy"] += (max(ys) + min(ys)) / 2 * max(area, 1.0)
            s["n"] += max(area, 1.0)
    for s in stats.values():
        if s["n"] > 0:
            s["cx"] /= s["n"]
            s["cy"] /= s["n"]
    return stats


def select_balanced(frames, objs_meta, k, start=0):
    """Label-only balanced selection.

    Greedy:
    1. pick the frame with the largest area for each rare class (signal,
       switch, train, catenary_pole);
    2. fill the rest by a score that rewards rare-class presence and penalizes
       near-duplicate indices (< 2 frames apart);
    3. if still short, fill by stride.
    """
    keys = sorted(frames.keys(), key=lambda x: int(x))
    n = len(keys)
    if k >= n:
        return keys[start:start + k]

    stats = {}
    for key in keys:
        fr = frames[key]
        # use a nominal size; only relative areas matter
        stats[key] = frame_stats(fr, objs_meta, 4112, 2504)

    selected = []
    used = set()

    def add(key):
        if key not in used and len(selected) < k:
            selected.append(key)
            used.add(key)

    # rare-class anchors
    for cls in ["signal", "switch", "train", "catenary_pole"]:
        best = max(keys, key=lambda x: stats[x].get(cls, {}).get("area", 0.0))
        if stats[best].get(cls, {}).get("area", 0.0) > 0:
            add(best)

    # score: rare presence + normalized area of critical classes
    max_area = {c: max(stats[x].get(c, {}).get("area", 0.0) for x in keys) or 1.0 for c in CRITICAL}
    scored = []
    for key in keys:
        s = stats[key]
        rare_presence = sum(1 for c in ["signal", "switch", "train"] if s.get(c, {}).get("area", 0.0) > 0)
        crit_area = sum(s.get(c, {}).get("area", 0.0) / max_area[c] for c in CRITICAL)
        scored.append((rare_presence * 2.0 + crit_area, key))
    scored.sort(key=lambda t: -t[0])

    for score, key in scored:
        if len(selected) >= k:
            break
        if key in used:
            continue
        # diversity: avoid adjacent frames
        if any(abs(int(key) - int(u)) < 2 for u in used):
            continue
        add(key)

    # fill by stride if still short
    stride = max(1, n // k)
    for i in range(start, n):
        if len(selected) >= k:
            break
        add(keys[i])
    for i in range(start, n, stride):
        if len(selected) >= k:
            break
        add(keys[i])
    return selected[:k]


def render(fr, objs_meta, W, H):
    gt = Image.new("L", (W, H), 255)
    gd = ImageDraw.Draw(gt)
    bbox = [W, H, 0, 0]
    for uid, o in fr["objects"].items():
        otype = objs_meta.get(uid, {}).get("type")
        if otype not in OSDAR_TO_RS:
            continue
        cid = NAME2ID[OSDAR_TO_RS[otype][0]]
        for e in o.get("object_data", {}).get("poly2d", []):
            if not e.get("name", "").startswith("rgb_highres_center"):
                continue
            v = e.get("val") or []
            if len(v) < 6:
                continue
            pts = [(min(max(x, 0), W - 1), min(max(y, 0), H - 1)) for x, y in parse_poly2d(v)]
            gd.polygon(pts, fill=cid)
            for x, y in pts:
                bbox[0], bbox[1] = min(bbox[0], x), min(bbox[1], y)
                bbox[2], bbox[3] = max(bbox[2], x), max(bbox[3], y)
    if bbox[0] > bbox[2] or bbox[1] > bbox[3]:
        return np.array(gt, dtype=np.uint8), None
    return np.array(gt, dtype=np.uint8), bbox


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="1.2")
    ap.add_argument("--data-root", default="./data/osdar")
    ap.add_argument("--img-sub", default="unpacked/rgb_highres_center")
    ap.add_argument("--labels", default="ex/1_calibration_1.2_labels.json")
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--prefix", default="b")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    img_dir = os.path.join(args.data_root, args.img_sub)
    lab = json.load(open(os.path.join(args.data_root, args.labels), encoding="utf-8"))["openlabel"]
    objs_meta, frames = lab["objects"], lab["frames"]
    keys = select_balanced(frames, objs_meta, args.k, args.start)
    print(f"seq={args.seq} total={len(frames)} balanced={len(keys)}")
    print("train idx:", [int(k) for k in keys])

    os.makedirs(os.path.join(args.out, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "masks", "train"), exist_ok=True)
    ok = 0
    selected = []
    for key in keys:
        fr = frames[key]
        imgp = frame_image_map(fr, img_dir)
        if imgp is None:
            selected.append({"frame": key, "status": "no_image"})
            continue
        img = Image.open(imgp).convert("RGB")
        W, H = img.size
        gt, bbox = render(fr, objs_meta, W, H)
        if bbox is None:
            cx, cy = W // 2, int(H * 0.75)
        else:
            cx, cy = int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)
        x0 = min(max(cx - CROP_W // 2, 0), max(0, W - CROP_W))
        y0 = min(max(cy - CROP_H // 2, 0), max(0, H - CROP_H))
        name = f"{args.prefix}{int(key):04d}"
        img.crop((x0, y0, x0 + CROP_W, y0 + CROP_H)).save(
            os.path.join(args.out, "images", "train", f"{name}.jpg"), quality=95)
        Image.fromarray(gt[y0:y0 + CROP_H, x0:x0 + CROP_W]).save(
            os.path.join(args.out, "masks", "train", f"{name}.png"))
        selected.append({"frame": key, "status": "ok", "crop": [x0, y0, CROP_W, CROP_H]})
        ok += 1

    yml = os.path.join(args.out, "osdar_ft.yaml")
    with open(yml, "w") as f:
        f.write(f"path: {args.out}\ntrain: images/train\nval: images/train\nmasks_dir: masks\nnames:\n")
        for i, nm in enumerate(NAMES19):
            f.write(f"  {i}: {nm}\n")
    with open(os.path.join(args.out, "selection.json"), "w", encoding="utf-8") as f:
        json.dump({"strategy": "balanced", "k": args.k, "selected": selected, "ok": ok}, f, ensure_ascii=False, indent=2)
    print(f"  train: {ok} pairs")
    print("yaml ->", yml)
    print("BAL_PREP_DONE")


if __name__ == "__main__":
    main()
