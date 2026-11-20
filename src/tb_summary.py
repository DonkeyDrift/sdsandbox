#!/usr/bin/env python3
"""读取 rl_tb 下指定 run 的 event 文件，打印 ep_len/ep_rew/bc_* 摘要与末段均值。"""
import argparse
import glob
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def summarize_run(path, label, tail=10):
    ea = EventAccumulator(path)
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    print("== %s (%s) ==" % (label, path))
    for tag in sorted(tags):
        if not any(k in tag for k in ("ep_len", "ep_rew", "bc_", "time/fps", "train/loss")):
            continue
        ev = ea.Scalars(tag)
        if not ev:
            continue
        vals = [e.value for e in ev]
        steps = [e.step for e in ev]
        tail_vals = vals[-tail:]
        print("  %-28s n=%d last=%.3f@%d  tail_mean=%.3f  min=%.3f max=%.3f"
              % (tag, len(vals), vals[-1], steps[-1],
                 sum(tail_vals) / len(tail_vals), min(vals), max(vals)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb", default="outputs/rl_tb")
    ap.add_argument("--run", action="append", required=True,
                    help="run-name 前缀(如 ppobc_canary)")
    ap.add_argument("--tail", type=int, default=10)
    args = ap.parse_args()
    for run in args.run:
        matches = sorted(glob.glob(os.path.join(args.tb, run + "_*")),
                         key=os.path.getmtime)
        if not matches:
            print("!! no tb dir for run %s under %s" % (run, args.tb))
            continue
        for path in matches:
            if os.path.isdir(path):
                summarize_run(path, run, tail=args.tail)


if __name__ == "__main__":
    main()
