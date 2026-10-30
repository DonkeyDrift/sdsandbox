#!/usr/bin/env python3
"""INFLIGHT 一致性门录制:fork gym 环境(ctl_step_ms=20)驾驶,逐步落盘
原始遥测 JSON + 12 维 obs(8 状态 + 在飞 [a_{k-1}, a_{k-2}]) + 动作到 jsonl,
供 C++ dkcpp gate2 --obs-inflight 离线回放对齐。

在飞历史维护镜像训练侧:用 rl_train.inflight_pair(recent)——与
ActionDelayWrapper._augment 同一函数,顺序 [最新, 次新]。

--unit 模式(不起 sim):用 mock env 构造 ActionDelayWrapper(obs_inflight=True),
同一动作序列分别走 (a) wrapper._augment (b) 本脚本的 recent+inflight_pair 路径,
断言 12 维观测逐位一致;不过则退出码 1。先跑 --unit 再放量录制。

用法(需先有 sim 在跑):
  python3 src/cpp_client/inflight_record.py --port 9621 \
      --model outputs/rl_export/inflight_gate_model.zip \
      --out outputs/inflight_replay.jsonl --steps 600 --seed 8100 --episodes 2
"""

import argparse
import copy
import json
import sys

import numpy as np

sys.path.insert(0, "/Users/hwp/Dev/DKC/GymDonkeyCar")
sys.path.insert(0, "src")

from rl_train import inflight_pair  # noqa: E402

SPAWN = {"node_min": 5, "node_max": 245, "lateral_offset": 1.0, "heading_noise": 8.0}


def unit_check():
    """mock env 驱动 wrapper,对比 wrapper._augment 与录制路径的 recent 维护。"""
    import gymnasium as gym

    from rl_train import ActionDelayWrapper

    class MockEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (8,), dtype=np.float32)
            self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
            self.t = 0

        def reset(self, *, seed=None, options=None):
            self.t = 0
            return np.full(8, self.t, dtype=np.float32), {}

        def step(self, action):
            self.t += 1
            return np.full(8, self.t, dtype=np.float32), 0.0, False, False, {}

    rng = np.random.default_rng(0)

    def run_case(delay, jitter):
        wrapped = ActionDelayWrapper(MockEnv(), delay=delay, jitter=jitter, obs_inflight=True)

        # wrapper 路径
        buf = [None] * delay
        maxlen = max(delay + (jitter or 0), 2)
        obs_w, _ = wrapped.reset()  # reset 内 _fresh + augment(zeros)
        wrapper_obs = [np.asarray(obs_w, dtype=np.float32).copy()]
        # 录制路径:同 recent 语义(旧→新, maxlen 同 wrapper)
        rec_obs = [np.concatenate([np.full(8, 0, np.float32),
                                   *inflight_pair(buf)]).astype(np.float32)]
        for _ in range(12):
            action = rng.uniform(-1, 1, size=2).astype(np.float32)
            # 录制路径先算本步观测增强(决策视角 = wrapper 下一 obs 的语义对齐见下)
            buf.append(np.asarray(action, dtype=np.float32).copy())
            if len(buf) > maxlen:
                del buf[0]
            obs, *_ = wrapped.step(action)
            wrapper_obs.append(np.asarray(obs, dtype=np.float32).copy())
            rec_obs.append(np.concatenate([np.zeros(8, np.float32),
                                           *inflight_pair(buf)]).astype(np.float32))

        for i, (w, r) in enumerate(zip(wrapper_obs, rec_obs)):
            w = np.asarray(w).reshape(-1)
            r = np.asarray(r).reshape(-1)
            if not np.array_equal(w[8:], r[8:]):
                print("[unit] FAIL delay=%d jitter=%d frame %d wrapper=%s rec=%s"
                      % (delay, jitter, i, w[8:], r[8:]))
                return False
            # 前 8 维路径不同(mock 的 obs 计数不同)只比增广段
        print("[unit] PASS delay=%d jitter=%d frames=%d (augmented segments bit-equal)"
              % (delay, jitter, len(wrapper_obs)))
        return True

    ok = True
    for delay, jitter in [(1, 3), (1, 0), (2, 0), (1, 1)]:
        ok = run_case(delay, jitter) and ok

    # reset 增广形状:12 维
    wrapped = ActionDelayWrapper(MockEnv(), delay=1, jitter=3, obs_inflight=True)
    o, info = wrapped.reset()
    assert o.shape == (12,), o.shape
    assert np.all(o[8:] == 0), o[8:]
    print("[unit] reset obs shape=(12,) in-flight zeros: PASS")
    return 0 if ok else 1


def record(args):
    from gym_donkeycar.envs.donkey_sim import DonkeyUnitySimContoller
    from stable_baselines3 import PPO

    conf = {
        "exe_path": args.sim_path,
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
        "headless": False,  # 录制一致性门有窗即可(遮挡不影响遥测)
        # 注意:obs_inflight 不进 conf——增强在录制循环侧(策略视角),
        # 与训练侧 ActionDelayWrapper 同构;fork 层保持 8 维。
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

    def emit(obs12, action, frame, ep, step):
        nonlocal recorded
        rec = {
            "ep": int(ep),
            "telemetry": stash[frame],
            "obs": np.asarray(obs12, dtype=np.float64).tolist(),
            "action": np.asarray(action, dtype=np.float64).tolist(),
            "frame": int(frame),
            "step": int(step),
        }
        out.write(json.dumps(rec) + "\n")
        recorded += 1

    for ep in range(args.episodes):
        seed = args.seed + ep
        handler.reset(seed=seed)
        obs8, _, _, _ = handler.observe()
        assert handler.steps_since_reset == 1
        # 决策 0:历史空 → 在飞全零(镜像 wrapper reset + C++ clear_episode)
        recent = []  # 旧→新,只留最近 2 个
        obs12 = np.concatenate([np.asarray(obs8, dtype=np.float32),
                                *inflight_pair(recent)]).astype(np.float32)
        action, _ = model.predict(obs12, deterministic=True)
        emit(obs12, action, handler.steps_since_reset, ep, 0)

        step = 0
        done = False
        while step < args.steps and not done:
            recent.append(np.asarray(action, dtype=np.float32).copy())
            del recent[:-2]  # maxlen=2
            ctl.take_action(action)
            obs8, _, done, _ = handler.observe()
            step += 1
            assert handler.steps_since_reset == step + 1
            obs12 = np.concatenate([np.asarray(obs8, dtype=np.float32),
                                    *inflight_pair(recent)]).astype(np.float32)
            action, _ = model.predict(obs12, deterministic=True)
            emit(obs12, action, handler.steps_since_reset, ep, step)
        print("[record] ep%d done=%s steps=%d" % (ep, done, step))

    out.close()
    print("[record] frames=%d -> %s" % (recorded, args.out))
    ctl.quit()
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", action="store_true", help="离线 wrapper 等价单测(不起 sim)")
    ap.add_argument("--port", type=int, default=9621)
    ap.add_argument("--sim-path", default="build/donkey_sim.app/Contents/MacOS/donkey_sim")
    ap.add_argument("--model", required=False)
    ap.add_argument("--out", default="outputs/inflight_replay.jsonl")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--seed", type=int, default=8100)
    args = ap.parse_args()

    if args.unit:
        sys.exit(unit_check())
    if not args.model:
        ap.error("--model required (unless --unit)")
    sys.exit(record(args))


if __name__ == "__main__":
    main()
