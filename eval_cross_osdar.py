#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OSDaR23 跨域零样本评估（M5，服务器运行）。

用法:
  python eval_cross_osdar.py --model /path/to/best.pt --seq 5.1 \
      --data-root ./data/osdar \
      --img-sub unpacked_51/rgb_highres_center \
      --labels unpacked_51/5_station_bergedorf_5.1_labels.json \
      --imgsz 1024 --device 0 --out eval_cross_51.md --montage 8

背景：RailSem19 训练的语义模型零样本迁移到 OSDaR23（项目真实数据）。
OSDaR23 官方标注（openlabel 格式）与 RailSem19 类表不同——22 个 things 类、
无 stuff 分割 GT，且部分类只有 bbox。因此本评估只报**可映射类**的逐类 IoU，
并分两级口径：
  L1 poly2d（多边形，主口径）：IoU 仅在「已标注区域」（所有可映射多边形并集）内计算，
     未标注区域不计 FP（避免把未标注误判为错）。
  L2 bbox（矩形近似，次口径）：pole/vehicle/buffer_stop 等只有 bbox 的类，
     按紧贴 bbox 当掩码近似——车辆类 bbox 含背景，IoU 是下界，仅供参考。

类映射（OSDaR23 type → RailSem19 类；按两官方类定义逐条对照，铁律）：
  track→rail-track · train→on-rails · switch→rail-raised · signal→traffic-light
  signal_pole/catenary_pole→pole · crowd/person→human · buffer_stop→construction
  road_vehicle→car|truck（合并计：预测为 car 或 truck 均算对）
  group_of_bicycles 等无对应类：跳过并记录。
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw

# OSDaR23 type -> (RailSem19 目标类名列表, 层级[thing/stuff])
OSDAR_TO_RS = {
    "track": (["rail-track"], "stuff"),
    "train": (["on-rails"], "thing"),
    "road_vehicle": (["car", "truck"], "thing"),
    "signal_pole": (["pole"], "thing"),
    "catenary_pole": (["pole"], "thing"),
    "signal": (["traffic-light"], "thing"),
    "buffer_stop": (["construction"], "thing"),
    "crowd": (["human"], "thing"),
    "person": (["human"], "thing"),
    "switch": (["rail-raised"], "thing"),
}
NO_RS_COUNTERPART = {
    "group_of_bicycles", "animal", "group_of_animals", "bicycle", "motorcycle",
    "wagons", "drag_shoe", "transition", "signal_bridge", "flame", "smoke",
    "wheelchair",
}
# 可评估 RS 类的可视化配色（BGR→RGB 任意固定色，仅目检用）
VIZ = {
    "rail-track": (200, 60, 60), "on-rails": (60, 180, 240), "rail-raised": (240, 120, 40),
    "traffic-light": (40, 220, 120), "pole": (250, 230, 60), "human": (230, 60, 200),
    "construction": (160, 160, 160), "car": (80, 120, 255), "truck": (140, 80, 255),
}


def parse_poly2d(val):
    """flat [x0,y0,x1,y1,...] -> [(x,y),...]"""
    pts = [(float(val[i]), float(val[i + 1])) for i in range(0, len(val) - 1, 2)]
    return pts


def render_frame_gt(fr, objs_meta, img_wh, name2id, use_bbox=False):
    """渲染单帧 GT。返回 (gt_mask[uint8, 255=未标注], region[bool], stats)。"""
    W, H = img_wh
    gt = Image.new("L", (W, H), 255)
    region = Image.new("L", (W, H), 0)
    gd, rd = ImageDraw.Draw(gt), ImageDraw.Draw(region)
    stats = defaultdict(int)
    things, stuffs = [], []
    for uid, o in fr["objects"].items():
        otype = objs_meta.get(uid, {}).get("type")
        if otype not in OSDAR_TO_RS:
            continue
        targets, layer = OSDAR_TO_RS[otype]
        ids = [name2id[t] for t in targets if t in name2id]
        if not ids:
            continue
        cid = ids[0]
        od = o.get("object_data", {})
        # poly2d（主口径）或 bbox（次口径，val=[x,y,w,h] → 矩形）
        entries = od.get("bbox", []) if use_bbox else od.get("poly2d", [])
        for e in entries:
            nm = e.get("name", "")
            if not nm.startswith("rgb_highres_center"):
                continue
            v = e.get("val") or []
            if use_bbox:
                if len(v) < 4:
                    continue
                x, y, bw, bh = [float(t) for t in v[:4]]
                pts = [(x, y), (x + bw, y), (x + bw, y + bh), (x, y + bh)]
            else:
                if len(v) < 6:
                    continue
                pts = parse_poly2d(v)
            pts = [(min(max(x, 0), W - 1), min(max(y, 0), H - 1)) for x, y in pts]
            (things if layer == "thing" else stuffs).append((pts, cid))
            rd.polygon(pts, fill=1)
            stats["_".join(targets)] += 1
    # things 覆盖 stuff（与 RailSem19 渲染优先级一致）
    for pts, cid in stuffs:
        gd.polygon(pts, fill=int(cid))
    for pts, cid in things:
        gd.polygon(pts, fill=int(cid))
    return np.array(gt, dtype=np.uint8), np.array(region, dtype=bool), dict(stats)


def frame_image_map(fr, img_dir):
    """用 labels 内 uri 权威匹配帧→图像；缺失时退化为数字前缀匹配。"""
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
    ap.add_argument("--model", required=True)
    ap.add_argument("--seq", required=True, help="序列名（如 5.1 / 1.2，仅用于报告）")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--img-sub", required=True, help="图像目录（相对 data-root）")
    ap.add_argument("--labels", required=True, help="labels json（相对 data-root）")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="eval_cross.md")
    ap.add_argument("--montage", type=int, default=8, help="目检拼图帧数（0=不生成）")
    ap.add_argument("--bbox-secondary", action="store_true",
                    help="同时计算 bbox 矩形近似口径（L2）")
    args = ap.parse_args()

    from ultralytics import YOLO

    img_dir = os.path.join(args.data_root, args.img_sub)
    lab_path = os.path.join(args.data_root, args.labels)
    with open(lab_path, encoding="utf-8") as f:
        ol = json.load(f)["openlabel"]
    objs_meta = ol["objects"]
    frames = ol["frames"]

    model = YOLO(args.model)
    names = [model.names[i] for i in range(len(model.names))]
    name2id = {n: i for i, n in enumerate(names)}
    print(f"[{args.seq}] 模型类数 {len(names)}；可映射 RS 类："
          f"{sorted({t for ts, _ in OSDAR_TO_RS.values() for t in ts if t in name2id})}")

    out_dir = os.path.splitext(args.out)[0] + "_masks"
    os.makedirs(out_dir, exist_ok=True)

    # 累计器：L1 poly2d / L2 bbox
    def new_acc():
        return {"inter": defaultdict(int), "union": defaultdict(int),
                "gtpix": defaultdict(int), "n": 0}
    acc = {"L1": new_acc(), "L2": new_acc()}
    skipped = defaultdict(int)
    montage_frames = []

    fkeys = sorted(frames.keys(), key=lambda k: int(k))
    for fk in fkeys:
        fr = frames[fk]
        imgp = frame_image_map(fr, img_dir)
        if imgp is None:
            skipped["no_image"] += 1
            continue
        img = Image.open(imgp)
        W, H = img.size
        res = model.predict(imgp, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        pred = res.semantic_mask.data.cpu().numpy().astype(np.uint8)
        if pred.shape != (H, W):
            skipped["shape_mismatch"] += 1
            continue

        for lvl, use_bbox in (("L1", False), ("L2", True)):
            if use_bbox and not args.bbox_secondary:
                continue
            gt, region, _ = render_frame_gt(fr, objs_meta, (W, H), name2id,
                                            use_bbox=use_bbox)
            a = acc[lvl]
            a["n"] += 1
            # 可映射类的并集判定：pred∈targets 即算命中该类
            for otype, (targets, _) in OSDAR_TO_RS.items():
                tids = [name2id[t] for t in targets if t in name2id]
                if not tids:
                    continue
                gtm = (gt == tids[0])  # GT 单类（渲染时取 targets[0]）
                # 注意：road_vehicle 的 GT 画在 targets[0]=car 上；pred 命中 car|truck 均算对
                predm = np.isin(pred, tids)
                inter = int(np.count_nonzero(gtm & predm & region))
                union = int(np.count_nonzero((gtm | predm) & region))
                key = "|".join(targets)
                a["inter"][key] += inter
                a["union"][key] += union
                a["gtpix"][key] += int(np.count_nonzero(gtm & region))
            if lvl == "L1" and len(montage_frames) < args.montage:
                montage_frames.append((fk, img.copy(), gt, pred))
        # 存 L1 GT 与 pred 供审计
        gt1, _, _ = render_frame_gt(fr, objs_meta, (W, H), name2id, use_bbox=False)
        Image.fromarray(gt1).save(os.path.join(out_dir, f"gt_{fk}.png"))
        Image.fromarray(pred).save(os.path.join(out_dir, f"pred_{fk}.png"))

    # ---- 汇总 ----
    def iou_of(a, key):
        u = a["union"][key]
        return (a["inter"][key] / u) if u > 0 else float("nan")

    lines = []
    lines.append(f"# OSDaR23 跨域零样本评估（{args.seq}）\n")
    lines.append(f"- 模型：`{args.model}`（RailSem19 训练，imgsz={args.imgsz}）")
    lines.append(f"- 数据：`{img_dir}` + `{lab_path}`")
    lines.append(f"- 帧数：L1={acc['L1']['n']} L2={acc['L2']['n']}；"
                 f"跳过：{dict(skipped) if skipped else '无'}")
    lines.append(f"- 口径 L1 = poly2d 多边形，IoU 仅在已标注区域（可映射多边形并集）内计算；"
                 f"L2 = bbox 矩形近似（次口径，车辆类含背景 IoU 为下界）")
    lines.append(f"- 类映射：track→rail-track · train→on-rails · switch→rail-raised · "
                 f"signal→traffic-light · signal_pole/catenary_pole→pole · "
                 f"crowd/person→human · buffer_stop→construction · road_vehicle→car|truck\n")

    for lvl in ("L1", "L2"):
        a = acc[lvl]
        if a["n"] == 0:
            continue
        lines.append(f"## {lvl} 口径逐类 IoU\n")
        lines.append("| GT 类(OSDaR23) | →RS 类 | GT 像素 | IoU |")
        lines.append("|---|---|---|---|")
        vals = []
        for key in sorted(a["union"]):
            v = iou_of(a, key)
            if not np.isnan(v):
                vals.append(v)
            vs = "NaN" if np.isnan(v) else f"{v*100:.2f}%"
            lines.append(f"| {key.split('|')[0]} | {key} | {a['gtpix'][key]:,} | {vs} |")
        if vals:
            lines.append(f"\n**{lvl} 可映射类平均 IoU = {np.mean(vals)*100:.2f}%**"
                         f"（{len(vals)} 类；非全 19 类，不可与 RailSem19 域内 mIoU 直接比）\n")

    # ---- 目检拼图 ----
    if args.montage and montage_frames:
        cols = min(4, len(montage_frames))
        rows = (len(montage_frames) + cols - 1) // cols
        tw = 480
        tiles = []
        for fk, img, gt, pred in montage_frames:
            w, h = img.size
            th = int(h * tw / w)
            t_rgb = img.resize((tw, th))
            t_gt = Image.fromarray(_colorize(gt, name2id)).resize((tw, th), Image.NEAREST)
            t_pr = Image.fromarray(_colorize(pred, name2id)).resize((tw, th), Image.NEAREST)
            tiles.append((fk, t_rgb, t_gt, t_pr))
        cell_h = tiles[0][1].size[1]
        canvas = Image.new("RGB", (cols * tw * 3 + (cols + 1) * 8,
                                   rows * (cell_h + 18) + 8), (20, 20, 20))
        for i, (fk, tr, tg, tp) in enumerate(tiles):
            r, c = divmod(i, cols)
            x = 8 + c * (tw * 3 + 8)
            y = 8 + r * (cell_h + 18)
            canvas.paste(tr, (x, y + 16))
            canvas.paste(tg, (x + tw + 4, y + 16))
            canvas.paste(tp, (x + 2 * tw + 8, y + 16))
            d = ImageDraw.Draw(canvas)
            d.text((x + 2, y + 2), f"f{fk}  RGB | GT(poly2d) | Pred", fill=(230, 230, 230))
        mpath = os.path.splitext(args.out)[0] + "_montage.png"
        canvas.save(mpath)
        lines.append(f"\n## 目检\n\n![]({os.path.basename(mpath)})"
                     f"（左：原图；中：官方 poly2d GT；右：模型预测；"
                     f"灰=未标注区域，仅已标注区参与 IoU）")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    csv_path = os.path.splitext(args.out)[0] + "_detail.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["level", "gt_class", "rs_class", "gt_pixels", "inter", "union", "iou"])
        for lvl in ("L1", "L2"):
            a = acc[lvl]
            for key in sorted(a["union"]):
                w.writerow([lvl, key.split("|")[0], key, a["gtpix"][key],
                            a["inter"][key], a["union"][key], f"{iou_of(a, key):.4f}"])
    print(f"[out] {args.out} + {csv_path} + masks/ ")


def _colorize(mask, name2id):
    """单通道类图 -> RGB 目检图。已映射类按 VIZ 配色；255/其余=深灰。"""
    h, w = mask.shape
    out = np.full((h, w, 3), 40, dtype=np.uint8)
    for name, (r, g, b) in VIZ.items():
        if name in name2id:
            out[mask == name2id[name]] = (r, g, b)
    return out


if __name__ == "__main__":
    main()
