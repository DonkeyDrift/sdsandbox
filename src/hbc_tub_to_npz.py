#!/usr/bin/env python3
"""HBC: mycar DonkeyCar 4.x catalog tub → bc_*.npz 转换器。

输入 root(data/ 顶层) + tub_2,跳过 tub_1(整 tub 被 manifest 删除)。
每行按 _index ∈ deleted_indexes 过滤;分片即局段——每 1000 保留帧一片,
删除洞 ≥10 帧强制断片(不跨洞拼接)。

输出键对齐 bc_train.load_dataset:image(N,H,W,3 uint8) steer throttle meta(N,4)。
throttle clip 到 [0,1](root 含 2.2% 负油门);steer 已在 [-1,1] 原样保留。

用法:python3 src/hbc_tub_to_npz.py --src /Users/hwp/Dev/DKC/mycar/data \
    --out outputs/bc_human_data --max-shard 1000 --gap-break 10
"""
import argparse
import json
import os

import numpy as np
from PIL import Image


def load_tub(tub_dir):
    """读 manifest + catalogs,返回保留帧 [(global_index, jpg_name, angle, throttle), ...]。"""
    with open(os.path.join(tub_dir, "manifest.json")) as f:
        lines = [json.loads(l) for l in f if l.strip()]
    man = lines[-1]
    deleted = set(man["deleted_indexes"])
    kept = []
    for p in man["paths"]:
        with open(os.path.join(tub_dir, p)) as f:
            for line in f:
                r = json.loads(line)
                if r["_index"] in deleted:
                    continue
                kept.append((r["_index"], r["cam/image_array"],
                             float(r["user/angle"]), float(r["user/throttle"])))
    kept.sort(key=lambda t: t[0])
    return kept


def shard(kept, max_shard, gap_break):
    """按保留帧切片;相邻保留帧原始索引差 ≥ gap_break 视为删除洞,强制断片。"""
    shards, cur = [], []
    for i, rec in enumerate(kept):
        if cur and (len(cur) >= max_shard or rec[0] - cur[-1][0] >= gap_break):
            shards.append(cur)
            cur = []
        cur.append(rec)
    if cur:
        shards.append(cur)
    return shards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/Users/hwp/Dev/DKC/mycar/data")
    ap.add_argument("--tubs", default=".,tub_2_26-10-23",
                    help="逗号分隔 tub 相对路径;跳过整删 tub")
    ap.add_argument("--out", default="outputs/bc_human_data")
    ap.add_argument("--max-shard", type=int, default=1000)
    ap.add_argument("--gap-break", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    all_stats, ep_id = [], 0
    neg_clip_total = 0
    montage_pool = []

    for tub_rel in args.tubs.split(","):
        tub_rel = tub_rel.strip()
        tub_dir = os.path.join(args.src, tub_rel) if tub_rel != "." else args.src
        kept = load_tub(tub_dir)
        if not kept:
            print(f"[skip] {tub_rel}: 保留 0 帧(整 tub 删除)")
            continue
        shards = shard(kept, args.max_shard, args.gap_break)
        print(f"[tub] {tub_rel}: 保留 {len(kept)} 帧 → {len(shards)} 片")

        angles, throttles, neg_clip = [], [], 0
        for sh in shards:
            n = len(sh)
            imgs = np.zeros((n, 240, 320, 3), dtype=np.uint8)
            steer = np.zeros(n, dtype=np.float32)
            thr = np.zeros(n, dtype=np.float32)
            meta = np.zeros((n, 4), dtype=np.float32)
            for j, (gi, jpg, ang, th) in enumerate(sh):
                imgs[j] = np.asarray(Image.open(
                    os.path.join(tub_dir, "images", jpg)).convert("RGB"))
                steer[j] = ang
                if th < 0:
                    neg_clip += 1
                thr[j] = min(max(th, 0.0), 1.0)
                meta[j] = (ep_id, 0, 0, gi)
            out = os.path.join(args.out, "bc_%05d.npz" % ep_id)
            np.savez_compressed(out, image=imgs, steer=steer, throttle=thr, meta=meta)
            angles.append(steer); throttles.append(thr)
            # 拼贴池:每片抽 2 帧
            pick = [0, n // 2] if n > 2 else [0]
            montage_pool.extend(imgs[i] for i in pick)
            print(f"  ep{ep_id}: {n} 帧 → {out}")
            ep_id += 1
        angles = np.concatenate(angles); throttles = np.concatenate(throttles)
        neg_clip_total += neg_clip
        all_stats.append({
            "tub": tub_rel, "frames": len(kept), "shards": len(shards),
            "steer_mean": float(angles.mean()), "steer_std": float(angles.std()),
            "steer_hard_frac": float((np.abs(angles) > 0.3).mean()),
            "thr_mean": float(throttles.mean()),
            "thr_gt02_frac": float((throttles > 0.2).mean()),
            "neg_clip": neg_clip,
            "d_thr_mean": float(np.abs(np.diff(throttles)).mean()),
            "d_steer_mean": float(np.abs(np.diff(angles)).mean()),
        })

    # ---- 数据卡 ----
    total = sum(s["frames"] for s in all_stats)
    print("\n==== HBC 数据卡 ====")
    for s in all_stats:
        print(json.dumps(s, ensure_ascii=False))
    print(f"合计有效帧={total} 局段={ep_id} 负油门clip={neg_clip_total}")
    all_thr = []
    for tub_rel in args.tubs.split(","):
        tub_rel = tub_rel.strip()
        tub_dir = os.path.join(args.src, tub_rel) if tub_rel != "." else args.src
        kept = load_tub(tub_dir)
        all_thr.extend(min(max(t, 0.0), 1.0) for _, _, _, t in kept)
    hist, edges = np.histogram(all_thr, bins=10, range=(0, 1))
    print("throttle hist(10 bins [0,1]):", hist.tolist())

    card = {"tubs": all_stats, "total_frames": total, "episodes": ep_id,
            "neg_clip_total": neg_clip_total, "throttle_hist_10": hist.tolist()}
    with open(os.path.join(args.out, "data_card.json"), "w") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)

    # 16 帧拼贴(4x4)
    pick = np.linspace(0, len(montage_pool) - 1, 16).astype(int)
    tiles = [montage_pool[i] for i in pick]
    grid = np.concatenate([np.concatenate(tiles[i * 4:(i + 1) * 4], axis=1)
                           for i in range(4)], axis=0)
    from PIL import Image as Im
    Im.fromarray(grid).save("outputs/hbc_mycar_montage.jpg", quality=88)
    print("拼贴 → outputs/hbc_mycar_montage.jpg")


if __name__ == "__main__":
    main()
