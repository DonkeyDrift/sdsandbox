#!/usr/bin/env python3
"""RL 策略评测:加载 PPO 模型,在模拟器里跑若干 episode 并统计回报/存活步数。

默认连接一个已在运行的模拟器(有窗口,可直观看到策略开车):
  1) 启动模拟器: build/donkey_sim.app/Contents/MacOS/donkey_sim --rl
  2) 评测:        python3 src/rl_eval.py --model outputs/rl_ppo.zip --episodes 3

也可以 --launch-sim 让脚本自起 headless 模拟器做隔离评测。
观测类型必须和训练一致(默认状态量,--vision 对应 CNN 模型)。
"""

import argparse

import numpy as np
import gymnasium as gym
import gym_donkeycar  # noqa: F401 注册 donkey-* env
from stable_baselines3 import PPO

SPAWN_DEFAULT = {
    "node_min": 5,
    "node_max": 245,
    "lateral_offset": 1.0,
    "heading_noise": 8.0,
}

# 场景名 -> gym env id(与 EditorBuildSettings 的 10 个可玩场景对应,menu 除外)
LEVEL_TO_ENV = {
    "mountain_track": "donkey-mountain-track-v0",
    "warehouse": "donkey-warehouse-v0",
    "sparkfun_avc": "donkey-avc-sparkfun-v0",
    "generated_track": "donkey-generated-track-v0",
    "generated_road": "donkey-generated-roads-v0",
    "roboracingleague_1": "donkey-roboracingleague-track-v0",
    "waveshare": "donkey-waveshare-v0",
    "mini_monaco": "donkey-minimonaco-track-v0",
    "warren": "donkey-warren-track-v0",
    "circuit_launch": "donkey-circuit-launch-track-v0",
}


class MergeStack(gym.Wrapper):
    """FrameStackObservation 对 channel-last 环境产出 (N,H,W,C) 引导维堆叠;
    每帧 HWC→CHW 转置后沿通道合并为 (N*C,H,W),与训练侧
    VecFrameStack(VecTransposeImage,channels_order='first') 同形。
    帧序:VecFrameStack 旧帧在前、新帧在最后(新帧=通道 9-11);
    FrameStackObservation 的 obs[0]=最旧,合并后同样新帧在 9-11,两
    侧天然对齐(探针显式断言)。"""

    def __init__(self, env):
        super().__init__(env)
        n, h, w, c = env.observation_space.shape  # channel-last!
        self._n = n
        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low.min(),
            high=env.observation_space.high.max(),
            shape=(n * c, h, w), dtype=env.observation_space.dtype)

    def _merge(self, obs):
        return np.concatenate([obs[i].transpose(2, 0, 1) for i in range(self._n)], axis=0)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._merge(obs), info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        return self._merge(obs), r, term, trunc, info


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
    parser.add_argument("--frame-stack", type=int, default=0,
                        help="视觉观测叠帧 N(需与训练一致,观测 (3N,120,160))")
    parser.add_argument("--spawn", dest="spawn", action="store_true", default=True,
                        help="随机出生(默认,与训练一致)")
    parser.add_argument("--no-spawn", dest="spawn", action="store_false", help="固定出生点")
    parser.add_argument("--max-cte", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=7, help="单 episode 种子(与 --seed-base 二选一)")
    parser.add_argument("--seed-base", type=int, default=None,
                        help="held-out 种子基数:第 i 局用 seed_base+i(训练未覆盖的种子域)")
    parser.add_argument("--json", action="store_true", help="末行输出机器可读 JSON 汇总")
    parser.add_argument("--action-repeat", type=int, default=1,
                        help="每个动作连续保持 N 个环境步:异步模式对齐训练的 0.05s 动作时长语义")
    parser.add_argument("--cam-fps", type=float, default=0.0,
                        help="异步评测遥测/控制频率(Hz,0=默认20):60Hz 连续控制可消除冠军的高增益失稳")
    parser.add_argument("--ctl-step", type=int, default=0,
                        help="量化异步控制步长(毫秒,相位协议):>0 时控制应用/遥测对齐物理网格边界,"
                             "动作-观测配对与同步 time_step 同构——同步冠军无需重训即可异步驾驶")
    parser.add_argument("--level", default="mountain_track",
                        help="场景名(Phase4 多场景零样本):%s" % ",".join(sorted(LEVEL_TO_ENV)))
    return parser.parse_args()


def main():
    args = parse_args()

    conf = {
        "exe_path": args.sim_path if args.launch_sim else "remote",
        "host": "127.0.0.1",
        "port": args.port,
        "headless": args.launch_sim,
        "level": args.level,
        "max_cte": args.max_cte,
        "sync_mode": args.sync_mode,
        "sync_time_step": args.time_step,
        "state_obs": args.state_obs,
        "spawn_cfg": dict(SPAWN_DEFAULT) if args.spawn else None,
        "start_delay": 3.0,
        "sim_args": ["--rl"],
    }
    if args.ctl_step > 0:
        conf["ctl_step_ms"] = args.ctl_step  # 量化异步控制(相位协议)
    if not args.state_obs:
        conf["cam_resolution"] = (120, 160, 3)

    env_id = LEVEL_TO_ENV.get(args.level)
    if env_id is None:
        raise SystemExit("未知场景 %r,可选: %s" % (args.level, sorted(LEVEL_TO_ENV)))
    print("[rl_eval] 连接 127.0.0.1:%d sync=%s state_obs=%s level=%s" %
          (args.port, args.sync_mode, args.state_obs, args.level))
    env = gym.make(env_id, conf=conf)
    if args.frame_stack > 0:
        from gymnasium.wrappers import FrameStackObservation
        env = MergeStack(FrameStackObservation(env, stack_size=args.frame_stack))
        print("[rl_eval] 视觉叠帧 x%d(合并通道)" % args.frame_stack)
    try:
        model = PPO.load(args.model)
    except Exception:
        from stable_baselines3 import SAC
        model = SAC.load(args.model)
        print("[rl_eval] SAC 模型已加载")

    results = []
    for ep in range(args.episodes):
        ep_seed = (args.seed_base + ep) if args.seed_base is not None else (args.seed + ep)
        obs, _ = env.reset(seed=ep_seed)
        total, steps, reason = 0.0, 0, "max_steps"
        while steps < args.max_steps:
            action, _ = model.predict(obs, deterministic=True)
            for _ in range(args.action_repeat):
                if steps >= args.max_steps:
                    break
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                steps += 1
                if terminated or truncated:
                    break
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
