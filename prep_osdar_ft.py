#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OSDaR23 细调数据准备（k-shot 迁移实验用）。

对 1.2 序列选 k 帧（stride 采样）：
  - 裁剪：以"已映射对象多边形"的包围盒为中心取 2056x1252 窗口（clamp 到图像内），
    保证钢轨/杆件在视野内、且训练有效缩放 ≈0.5（与 RailSem19 训练尺度一致）
  - 掩码：poly2d 多边形按 OSDAR_TO_RS 映射光栅化（native 分辨率）后裁同一窗口；
    未标注/未映射区域 = 255（ignore）
  - 输出 YOLO-sem 结构：images/{train,val}/x.jpg + masks/{train,val}/x.png + yaml
    （nc=19，类名顺序与 RailSem19 训练一致——从 official_ids.json 的 19 类读）
用法:
  python prep_osdar_ft.py --seq 1.2 --k 30 --out /home/.../data/osdar_ft/k30
"""
import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw

# 与 eval_cross_osdar.py 同一映射（铁律：两处若改须同步）
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


def parse_poly2d(val):
    return [(float(val[i]), float(val[i + 1])) for i in range(0, len(val) - 1, 2)]


def frame_image_map(fr, img_dir):
    fp = fr.get("frame_properties", {})
    st = fp.get("streams", {}).get("rgb_highres_center", {})
    uri = st.get("uri")
    if uri:
        p = os.path.join(img_dir, os.path.basename(uri))
        if os.path.isfile(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="1.2")
    ap.add_argument("--data-root", default="./data/osdar")
    ap.add_argument("--img-sub", default="unpacked/rgb_highres_center")
    ap.add_argument("--labels", default="ex/1_calibration_1.2_labels.json")
    ap.add_argument("--k", type=int, default=30, help="训练帧数（stride 采样）")
    ap.add_argument("--val-k", type=int, default=0, help="预留 val 帧数（0=不做 val 划分，用 he_idx）")
    ap.add_argument("--prefix", default="f", help="文件名前缀（多序列混合时防冲突，如 a/b）")
    ap.add_argument("--start", type=int, default=0, help="采样起始帧号")
    ap.add_argument("--zoom-cls", default="", help="小类放大裁剪的目标类（OSDaR23 type，如 signal）")
    ap.add_argument("--zoom-k", type=int, default=0, help="放大裁剪帧数（按该类像素量 top-k）")
    ap.add_argument("--zoom-win", default="1372x834", help="放大裁剪窗口 WxH（保持与其他裁剪同纵横比）")
    ap.add_argument("--zoom-prefix", default="z", help="放大裁剪文件名前缀")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    img_dir = os.path.join(args.data_root, args.img_sub)
    lab = json.load(open(os.path.join(args.data_root, args.labels), encoding="utf-8"))["openlabel"]
    objs_meta, frames = lab["objects"], lab["frames"]
    fkeys = sorted(frames.keys(), key=lambda x: int(x))
    n = len(fkeys)
    # stride 采样 train；末尾 val（如果有）
    if args.val_k > 0:
        train_keys = [fkeys[i] for i in range(args.start, n - args.val_k, max(1, (n - args.val_k) // args.k))][:args.k]
        val_keys = fkeys[n - args.val_k:]
    else:
        idxs = list(range(args.start, n, max(1, n // args.k)))[:args.k]
        train_keys = [fkeys[i] for i in idxs]
        val_keys = []
    print(f"seq={args.seq} total={n} train={len(train_keys)} val={len(val_keys)}")
    print("train idx:", [int(k) for k in train_keys])

    os.makedirs(os.path.join(args.out, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "masks", "train"), exist_ok=True)
    if val_keys:
        os.makedirs(os.path.join(args.out, "images", "val"), exist_ok=True)
        os.makedirs(os.path.join(args.out, "masks", "val"), exist_ok=True)

    def render(fr, W, H):
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
            bbox = None
        return np.array(gt, dtype=np.uint8), bbox

    def one_split(keys, split):
        ok = 0
        for k in keys:
            fr = frames[k]
            imgp = frame_image_map(fr, img_dir)
            if imgp is None:
                print(f"  frame {k}: no image, skip")
                continue
            img = Image.open(imgp).convert("RGB")
            W, H = img.size
            gt, bbox = render(fr, W, H)
            if bbox is None:
                cx, cy = W // 2, int(H * 0.75)
            else:
                cx, cy = int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)
            x0 = min(max(cx - CROP_W // 2, 0), W - CROP_W)
            y0 = min(max(cy - CROP_H // 2, 0), H - CROP_H)
            img.crop((x0, y0, x0 + CROP_W, y0 + CROP_H)).save(
                os.path.join(args.out, "images", split, f"{args.prefix}{int(k):04d}.jpg"), quality=95)
            Image.fromarray(gt[y0:y0 + CROP_H, x0:x0 + CROP_W]).save(
                os.path.join(args.out, "masks", split, f"{args.prefix}{int(k):04d}.png"))
            ok += 1
        print(f"  {split}: {ok} pairs")
        return ok

    one_split(train_keys, "train")
    if val_keys:
        one_split(val_keys, "val")

    # ---- zoom pass: small-class oversampling (e.g., signal -> traffic-light) ----
    if args.zoom_cls and args.zoom_k > 0:
        zw, zh = [int(v) for v in args.zoom_win.lower().split("x")]
        cand = []
        for k in fkeys:
            fr = frames[k]
            sig_px, cx_sum, cy_sum, n = 0, 0.0, 0.0, 0
            for uid, o in fr["objects"].items():
                if objs_meta.get(uid, {}).get("type") != args.zoom_cls:
                    continue
                for e in o.get("object_data", {}).get("poly2d", []):
                    if not e.get("name", "").startswith("rgb_highres_center"):
                        continue
                    v = e.get("val") or []
                    if len(v) < 6:
                        continue
                    pts = parse_poly2d(v)
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    px = len(pts)  # 顶点数作为规模代理（小类多边形顶点少但计数不符时仍有排序意义）
                    # 用包围盒面积近似像素量
                    area = (max(xs) - min(xs)) * (max(ys) - min(ys))
                    sig_px += int(area)
                    cx_sum += (max(xs) + min(xs)) / 2 * max(area, 1)
                    cy_sum += (max(ys) + min(ys)) / 2 * max(area, 1)
                    n += max(area, 1)
            if n > 0:
                cand.append((k, sig_px, cx_sum / n, cy_sum / n))
        cand.sort(key=lambda t: -t[1])
        picked = cand[: args.zoom_k]
        print(f"zoom({args.zoom_cls}) top-{len(picked)} frames: {[int(c[0]) for c in picked]}")
        n_ok = 0
        for k, px, cx, cy in picked:
            fr = frames[k]
            imgp = frame_image_map(fr, img_dir)
            if imgp is None:
                continue
            img = Image.open(imgp).convert("RGB")
            W, H = img.size
            gt, _ = render(fr, W, H)
            x0 = min(max(int(cx) - zw // 2, 0), W - zw)
            y0 = min(max(int(cy) - zh // 2, 0), H - zh)
            img.crop((x0, y0, x0 + zw, y0 + zh)).save(
                os.path.join(args.out, "images", "train",
                             f"{args.zoom_prefix}{int(k):04d}.jpg"), quality=95)
            Image.fromarray(gt[y0:y0 + zh, x0:x0 + zw]).save(
                os.path.join(args.out, "masks", "train", f"{args.zoom_prefix}{int(k):04d}.png"))
            n_ok += 1
        print(f"  zoom crops written: {n_ok} (win {zw}x{zh})")

    yml = os.path.join(args.out, "osdar_ft.yaml")
    with open(yml, "w") as f:
        f.write(f"path: {args.out}\ntrain: images/train\n")
        if val_keys:
            f.write("val: images/val\n")
        else:
            f.write("val: images/train\n")
        f.write("masks_dir: masks\nnames:\n")
        for i, nm in enumerate(NAMES19):
            f.write(f"  {i}: {nm}\n")
    print("yaml ->", yml)
    print("FT_PREP_DONE")


if __name__ == "__main__":
    main()
