#!/usr/bin/env python3
"""RL 策略评测:加载 PPO 模型,在模拟器里跑若干 episode 并统计回报/存活步数。

默认连接一个已在运行的模拟器(有窗口,可直观看到策略开车):
  1) 启动模拟器: build/donkey_sim.app/Contents/MacOS/donkey_sim --rl
  2) 评测:        python3 src/rl_eval.py --model outputs/rl_ppo.zip --episodes 3

也可以 --launch-sim 让脚本自起 headless 模拟器做隔离评测。
观测类型必须和训练一致(默认状态量,--vision 对应 CNN 模型)。
"""

import argparse
import os

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


class RecorderWrapper(gym.Wrapper):
    """现场录帧(crash forensics):每步截取 policy 视角画面入环形缓冲,每局结束
    落盘——末 N 帧序列 + 16 格蒙太奇 + meta.json,供赛后多模态逐帧复核死因
    (替代纯坐标聚类推断)。

    取帧优先级:① 叠帧观测 (3N,H,W) → 最新帧=末 3 通道(与 MergeStack/
    VecFrameStack 帧序一致);② 单帧图像观测 (H,W,C) 直接用;③ 状态量观测 →
    env.unwrapped.viewer.handler.image_array(fork 只要遥测带 JPG 就持续解码,
    与观测类型无关;要求 cam_config img_enc 非 none,--record 时会强制 JPG)。"""

    def __init__(self, env, out_dir, keep=24, scale=2):
        super().__init__(env)
        import collections
        self._buf = collections.deque(maxlen=keep)
        self._out_dir = out_dir
        self._scale = scale
        self._zero_frames = 0

    def _latest_frame(self, obs):
        arr = np.asarray(obs)
        if arr.ndim == 3 and arr.shape[0] > 3 and arr.shape[0] % 3 == 0 and arr.shape[-1] not in (1, 3):
            return arr[-3:].transpose(1, 2, 0)  # 叠帧:新帧在末 3 通道
        if arr.ndim == 3 and arr.shape[-1] in (1, 3):
            return arr
        return self.env.unwrapped.viewer.handler.image_array  # 状态量:遥测画面

    def reset(self, **kwargs):
        self._buf.clear()
        self._zero_frames = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        frame = np.asarray(self._latest_frame(obs))
        if frame.ndim == 3 and frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)
        if frame.size and not frame.any():
            self._zero_frames += 1
        self._buf.append(frame)
        return obs, r, term, trunc, info

    def save_episode(self, ep, seed, meta):
        import json
        import os
        from PIL import Image

        tag = "crash" if str(meta.get("reason", "")).startswith("terminated") else "survive"
        ep_dir = os.path.join(self._out_dir, "ep%02d_seed%d_%s" % (ep, seed, tag))
        frames_dir = os.path.join(ep_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        frames = list(self._buf)
        for i, fr in enumerate(frames):
            Image.fromarray(fr).save(os.path.join(frames_dir, "f_%04d.jpg" % i), quality=90)

        tail = frames[-16:]  # 蒙太奇:末 16 帧,4 列,行优先时间从早到晚
        cols, rows = 4, max(1, -(-len(tail) // 4))
        h, w = tail[0].shape[:2]
        sheet = Image.new("RGB", (cols * w * self._scale, rows * h * self._scale))
        for i, fr in enumerate(tail):
            img = Image.fromarray(fr).resize((w * self._scale, h * self._scale), Image.NEAREST)
            sheet.paste(img, ((i % cols) * w * self._scale, (i // cols) * h * self._scale))
        sheet.save(os.path.join(ep_dir, "montage.jpg"), quality=90)

        meta = dict(meta, episode=ep, seed=seed, frames_saved=len(frames),
                    zero_frames=self._zero_frames,
                    montage="末 %d 帧,4 列,行优先时间从早到晚,左上最旧右下最新" % len(tail))
        with open(os.path.join(ep_dir, "meta.json"), "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
        if self._zero_frames:
            print("[rl_eval] 警告: %d/%d 帧全黑——遥测无图像?(state 观测需 img_enc 非 none)"
                  % (self._zero_frames, len(frames)))
        return ep_dir


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
    parser.add_argument("--cam-res", type=int, nargs=2, default=(120, 160),
                        metavar=("H", "W"), help="相机分辨率(默认 120 160;须与模型训练一致)")
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
    parser.add_argument("--record", default=None,
                        help="现场录帧目录(如 outputs/rl_replay_8000):每局结束导出末 N 帧+蒙太奇+meta.json,"
                             "供赛后逐帧复核死因;状态量观测时强制 img_enc=JPG(渲染+编码,评测略慢);"
                             "同时启用自动防停车检测(末 16 帧相邻灰度 L1 帧差)")
    parser.add_argument("--progress-gate", type=float, default=1.0,
                        help="进展门(防停车伪存活):单局 mean_speed 低于此值判 parked/fail,不计入平均"
                             "(真司机标定的 ~50%,见 journal)")
    parser.add_argument("--park-epsilon", type=float, default=2.0,
                        help="自动防停车:--record 末 16 帧相邻灰度 L1 均值差 < 此值判 parked(标定值)")
    parser.add_argument("--record-frames", type=int, default=24, help="每局保留的末尾帧数(默认 24)")
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
        conf["cam_resolution"] = (args.cam_res[0], args.cam_res[1], 3)
        conf["cam_config"] = {"img_enc": "JPG"}  # 触发 fork 的 img_w/h 注入块(否则 sim 恒 120×160)
    if args.record and args.state_obs:
        conf["cam_config"] = {"img_enc": "JPG"}  # 录帧强制有图(state 观测默认 none 会全黑)
    if args.launch_sim and (not args.state_obs or args.record):
        # -nographics 下相机不渲染(遥测恒灰图,视觉评测会变"盲测");keep_gfx=离屏渲染
        conf["headless_keep_gfx"] = True

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
    if args.record:
        env = RecorderWrapper(env, args.record, keep=args.record_frames)
        print("[rl_eval] 现场录帧开启 -> %s (每局末 %d 帧)" % (args.record, args.record_frames))
    try:
        model = PPO.load(args.model)
    except Exception:
        from stable_baselines3 import SAC
        model = SAC.load(args.model)
        print("[rl_eval] SAC 模型已加载")

    def detect_park(rec_dir, eps):
        """末 16 帧相邻灰度 L1 均值差——停车 ≈ JPEG 噪声(≈0-2),真驾驶 ≫。"""
        import glob
        frames = sorted(glob.glob(os.path.join(rec_dir, "frames", "f_*.jpg")))
        if len(frames) < 4:
            return None
        from PIL import Image
        gs = [np.asarray(Image.open(f).convert("L"), dtype=np.int16) for f in frames[-16:]]
        d = [float(np.abs(gs[i + 1] - gs[i]).mean()) for i in range(len(gs) - 1)]
        return sum(d) / len(d)

    results = []
    parked_eps = 0
    for ep in range(args.episodes):
        ep_seed = (args.seed_base + ep) if args.seed_base is not None else (args.seed + ep)
        obs, _ = env.reset(seed=ep_seed)
        total, steps, reason, spd_sum = 0.0, 0, "max_steps", 0.0
        while steps < args.max_steps:
            action, _ = model.predict(obs, deterministic=True)
            for _ in range(args.action_repeat):
                if steps >= args.max_steps:
                    break
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                steps += 1
                spd_sum += float(info.get("speed", 0.0))
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
        mean_speed = spd_sum / max(steps, 1)
        rec_dir = None
        if args.record:
            rec_dir = env.save_episode(ep, ep_seed, {
                "model": args.model, "level": args.level, "sync_mode": args.sync_mode,
                "ctl_step": args.ctl_step, "frame_stack": args.frame_stack,
                "state_obs": args.state_obs, "reward": round(total, 1), "steps": steps,
                "reason": reason, "lap": info.get("lap", 0), "mean_speed": round(mean_speed, 3),
            })
        parked = mean_speed < args.progress_gate
        frame_diff = detect_park(rec_dir, args.park_epsilon) if rec_dir else None
        if frame_diff is not None and frame_diff < args.park_epsilon:
            parked = True
        results.append((total, steps, reason, info.get("lap", 0), mean_speed, parked))
        tag = " PARKED(fail)" if parked else ""
        print("[rl_eval] ep%d (seed=%d): reward=%.1f steps=%d mean_speed=%.2f%s -> %s"
              % (ep, ep_seed, total, steps, mean_speed, tag, reason))
        if rec_dir:
            print("[rl_eval] 现场 -> %s%s" % (rec_dir, " 帧差=%.2f" % frame_diff if frame_diff is not None else ""))
        if parked:
            parked_eps += 1

    env.close()
    valid = [r for r in results if not r[5]]
    avg_r = sum(r[0] for r in valid) / max(len(valid), 1)
    avg_l = sum(r[1] for r in valid) / max(len(valid), 1)
    avg_s = sum(r[4] for r in valid) / max(len(valid), 1)
    crashed = sum(1 for r in valid if r[2].startswith("terminated"))
    laps = sum(r[3] for r in valid)
    print("[rl_eval] 有效局 %d/%d(parked 剔除 %d) 平均 reward=%.1f 平均步数=%.0f 平均速度=%.2f"
          % (len(valid), len(results), parked_eps, avg_r, avg_l, avg_s))
    if args.json:
        import json as _json
        print("RL_EVAL_JSON " + _json.dumps({
            "episodes": len(results), "valid_episodes": len(valid),
            "parked_episodes": parked_eps,
            "avg_reward": round(avg_r, 1), "avg_steps": round(avg_l, 1),
            "mean_speed": round(avg_s, 3),
            "progress_gate": args.progress_gate, "park_epsilon": args.park_epsilon,
            "crash_episodes": crashed,
            "crash_rate": round(crashed / max(len(valid), 1), 3), "laps": laps,
            "max_steps": args.max_steps,
            "per_episode": [{"reward": round(r[0], 1), "steps": r[1], "mean_speed": round(r[4], 3),
                             "parked": r[5], "reason": r[2][:40]} for r in results],
            "seeds": (list(range(args.seed_base, args.seed_base + args.episodes))
                      if args.seed_base is not None else None),
        }))


if __name__ == "__main__":
    main()
