#!/usr/bin/env python3
"""RL 策略评测:加载 PPO 模型,在模拟器里跑若干 episode 并统计回报/存活步数。

默认连接一个已在运行的模拟器(有窗口,可直观看到策略开车):
  1) 启动模拟器: build/donkey_sim.app/Contents/MacOS/donkey_sim --rl
  2) 评测:        python3 src/rl_eval.py --model outputs/rl_ppo.zip --episodes 3

也可以 --launch-sim 让脚本自起 headless 模拟器做隔离评测。
观测类型必须和训练一致(默认状态量,--vision 对应 CNN 模型)。
"""

import argparse

import gymnasium as gym
import gym_donkeycar  # noqa: F401 注册 donkey-* env
from stable_baselines3 import PPO

SPAWN_DEFAULT = {
    "node_min": 5,
    "node_max": 245,
    "lateral_offset": 1.0,
    "heading_noise": 8.0,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO policy against DonkeySim")
    parser.add_argument("--model", default="outputs/rl_ppo.zip", help="模型路径")
    parser.add_argument("--episodes", type=int, default=3, help="评测 episode 数")
    parser.add_argument("--max-steps", type=int, default=1500, help="单 episode 最大步数(截断)")
    parser.add_argument("--port", type=int, default=9091, help="模拟器端口(连接已有实例时)")
    parser.add_argument("--launch-sim", action="store_true", help="自起 headless 模拟器(端口同 --port)")
    parser.add_argument("--sim-path", default="build/donkey_sim.app/Contents/MacOS/donkey_sim")
    parser.add_argument("--sync", dest="sync_mode", action="store_true", default=False,
                        help="同步步进评测(默认异步实时,配合有窗模拟器观看)")
    parser.add_argument("--time-step", type=float, default=0.05)
    parser.add_argument("--vision", dest="state_obs", action="store_false", default=True,
                        help="图像观测(训练的 CNN 模型)")
    parser.add_argument("--spawn", dest="spawn", action="store_true", default=True,
                        help="随机出生(默认,与训练一致)")
    parser.add_argument("--no-spawn", dest="spawn", action="store_false", help="固定出生点")
    parser.add_argument("--max-cte", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=7, help="单 episode 种子(与 --seed-base 二选一)")
    parser.add_argument("--seed-base", type=int, default=None,
                        help="held-out 种子基数:第 i 局用 seed_base+i(训练未覆盖的种子域)")
    parser.add_argument("--json", action="store_true", help="末行输出机器可读 JSON 汇总")
    parser.add_argument("--cam-fps", type=float, default=0.0,
                        help="异步评测遥测/控制频率(Hz,0=默认20):60Hz 连续控制可消除冠军的高增益失稳")
    return parser.parse_args()


def main():
    args = parse_args()

    conf = {
        "exe_path": args.sim_path if args.launch_sim else "remote",
        "host": "127.0.0.1",
        "port": args.port,
        "headless": args.launch_sim,
        "level": "mountain_track",
        "max_cte": args.max_cte,
        "sync_mode": args.sync_mode,
        "sync_time_step": args.time_step,
        "state_obs": args.state_obs,
        "spawn_cfg": dict(SPAWN_DEFAULT) if args.spawn else None,
        "start_delay": 3.0,
        "sim_args": ["--rl"],
    }
    if not args.state_obs:
        conf["cam_resolution"] = (120, 160, 3)

    print("[rl_eval] 连接 127.0.0.1:%d sync=%s state_obs=%s" % (args.port, args.sync_mode, args.state_obs))
    env = gym.make("donkey-mountain-track-v0", conf=conf)
    model = PPO.load(args.model)

    results = []
    for ep in range(args.episodes):
        ep_seed = (args.seed_base + ep) if args.seed_base is not None else (args.seed + ep)
        obs, _ = env.reset(seed=ep_seed)
        total, steps, reason = 0.0, 0, "max_steps"
        while steps < args.max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            steps += 1
            if terminated:
                reason = "terminated(cte=%.2f hit=%s flipped=%s stuck=%.1f lap=%d pos=%s)" % (
                    info.get("cte", 0), info.get("hit"), info.get("flipped"),
                    info.get("stuck_time", 0), info.get("lap", 0), info.get("pos"))
                break
            if truncated:
                reason = "truncated"
                break
        results.append((total, steps, reason, info.get("lap", 0)))
        print("[rl_eval] ep%d (seed=%d): reward=%.1f steps=%d -> %s" % (ep, ep_seed, total, steps, reason))

    env.close()
    avg_r = sum(r[0] for r in results) / len(results)
    avg_l = sum(r[1] for r in results) / len(results)
    crashed = sum(1 for r in results if r[2].startswith("terminated"))
    laps = sum(r[3] for r in results)
    print("[rl_eval] 平均 reward=%.1f 平均步数=%.0f (1 步=%.2fs 仿真)" %
          (avg_r, avg_l, args.time_step if args.sync_mode else 0.05))
    if args.json:
        import json as _json
        print("RL_EVAL_JSON " + _json.dumps({
            "episodes": len(results), "avg_reward": round(avg_r, 1),
            "avg_steps": round(avg_l, 1), "crash_episodes": crashed,
            "crash_rate": round(crashed / len(results), 3), "laps": laps,
            "max_steps": args.max_steps,
            "seeds": (list(range(args.seed_base, args.seed_base + args.episodes))
                      if args.seed_base is not None else None),
        }))


if __name__ == "__main__":
    main()
