#!/usr/bin/env python3
"""gate2 录制:用 fork gym 环境(量化异步 ctl_step_ms=20)驾驶一段会话,逐步落盘
原始遥测 JSON + python obs 向量 + python 动作 到 jsonl,供 C++ 离线回放对齐。

用法(需先有 sim 在跑):
  python3 src/cpp_client/gate2_record.py --port 9301 \
      --model outputs/rl_best_delay20j/best_model.zip \
      --out outputs/cpp_replay.jsonl --steps 600 --seed 8100

配对方法:包装 handler.fns["telemetry"],把原始 message deepcopy 进
stash[steps_since_reset](接收线程每帧恰好递增一次);主循环用 handler.observe()
阻塞到下一帧——observe 返回时 steps_since_reset 就是产生本次 obs 的帧号,
取 stash[该帧号] 即严格同帧(下一帧只会写别的 key,不会污染)。
"""

import argparse
import copy
import json
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/hwp/Dev/DKC/GymDonkeyCar")

from gym_donkeycar.envs.donkey_sim import DonkeyUnitySimContoller  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

SPAWN = {"node_min": 5, "node_max": 245, "lateral_offset": 1.0, "heading_noise": 8.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9301)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=8100)
    args = ap.parse_args()

    conf = {
        "exe_path": "remote",
        "host": "127.0.0.1",
        "port": args.port,
        "level": "mountain_track",
        "max_cte": 2.5,
        "sync_mode": False,
        "state_obs": True,
        "spawn_cfg": dict(SPAWN),
        "start_delay": 3.0,
        "sim_args": ["--rl"],
        "ctl_step_ms": 20,
        "cam_resolution": (120, 160, 3),
        "log_level": 20,
    }
    ctl = DonkeyUnitySimContoller(conf=conf)
    handler = ctl.handler
    ctl.wait_until_loaded()

    stash = {}
    orig = handler.fns["telemetry"]

    def wrapped(message):
        orig(message)
        stash[handler.steps_since_reset] = copy.deepcopy(message)

    handler.fns["telemetry"] = wrapped

    model = PPO.load(args.model, device="cpu")

    out = open(args.out, "w")
    recorded = 0

    def emit(obs, action, reward, done, step, ep):
        nonlocal recorded
        frame = handler.steps_since_reset
        rec = {
            "ep": int(ep),
            "telemetry": stash[frame],
            "obs": np.asarray(obs, dtype=np.float64).tolist(),
            "action": np.asarray(action, dtype=np.float64).tolist(),
            "reward": float(reward),
            "done": bool(done),
            "frame": int(frame),
            "step": int(step),
        }
        out.write(json.dumps(rec) + "\n")
        recorded += 1

    for ep in range(args.episodes):
        seed = args.seed + ep
        handler.reset(seed=seed)
        obs, reward, done, info = handler.observe()  # 阻塞到 reset 后首帧
        assert handler.steps_since_reset == 1
        action, _ = model.predict(obs, deterministic=True)
        emit(obs, action, reward, done, 0, ep)

        step = 0
        while step < args.steps and not done:
            action, _ = model.predict(obs, deterministic=True)
            ctl.take_action(action)
            obs, reward, done, info = handler.observe()  # 阻塞到下一帧
            step += 1
            assert handler.steps_since_reset == step + 1
            action_next, _ = model.predict(obs, deterministic=True)
            emit(obs, action_next, reward, done, step, ep)
        print("[record] ep%d done=%s steps=%d" % (ep, done, step))

    out.close()
    print("[record] frames=%d -> %s" % (recorded, args.out))
    ctl.quit()


if __name__ == "__main__":
    main()
