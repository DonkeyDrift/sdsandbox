#!/usr/bin/env python3
"""RL 训练入口:PPO + sdsandbox DonkeySim 新协议(同步步进/状态量观测/随机出生)。

每个 worker 起一个独立 sim 实例(--rl 模式),端口 port_base+rank:
  python3 src/rl_train.py                    # 默认:2 env、同步步进、状态量观测
  python3 src/rl_train.py --vision --envs 1  # 图像观测(CNN),sim 需出窗口渲染
"""

import argparse

import gymnasium as gym
import gym_donkeycar  # noqa: F401 注册 donkey-* env(子进程 import 本模块时也需要)
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

SPAWN_DEFAULT = {
    "node_min": 5,
    "node_max": 245,
    "lateral_offset": 1.0,
    "heading_noise": 8.0,
}


def parse_args():
    parser = argparse.ArgumentParser(description="PPO RL training against sdsandbox DonkeySim")
    parser.add_argument("--envs", type=int, default=2, help="并行 sim/worker 数")
    parser.add_argument("--steps", type=int, default=20000, help="训练总步数")
    parser.add_argument("--port-base", type=int, default=9101, help="首个 sim 端口,worker i 用 port_base+i")
    parser.add_argument("--sim-path", default="build/donkey_sim.app/Contents/MacOS/donkey_sim", help="sim 可执行文件路径")
    parser.add_argument("--sync", dest="sync_mode", action="store_true", default=True, help="同步步进(默认)")
    parser.add_argument("--async-mode", dest="sync_mode", action="store_false", help="异步遥测模式")
    parser.add_argument("--state-obs", dest="state_obs", action="store_true", default=True, help="状态量观测(默认)")
    parser.add_argument("--vision", dest="state_obs", action="store_false", help="图像观测(CNN 策略)")
    parser.add_argument("--img-enc", default="none", help="无图像观测时的 cam_config img_enc(默认 none)")
    parser.add_argument("--time-step", type=float, default=0.05, help="同步步进步长(秒)")
    parser.add_argument("--spawn", dest="spawn", action="store_true", default=True, help="随机出生到赛道节点(默认)")
    parser.add_argument("--no-spawn", dest="spawn", action="store_false", help="复位回出生点")
    parser.add_argument("--max-cte", type=float, default=2.5, help="出界 cross-track 误差阈值")
    parser.add_argument("--seed", type=int, default=1, help="训练/出生随机种子")
    parser.add_argument("--save", default="outputs/rl_ppo", help="最终模型保存路径")
    parser.add_argument("--tb", default="outputs/rl_tb", help="tensorboard 日志目录")
    return parser.parse_args()


def make_env(rank: int, args, spawn_cfg, headless, img_enc):
    port = args.port_base + rank

    def _init():
        conf = {
            "exe_path": args.sim_path,
            "host": "127.0.0.1",
            "port": port,
            "headless": headless,
            "level": "mountain_track",
            "max_cte": args.max_cte,
            "sync_mode": args.sync_mode,
            "sync_time_step": args.time_step,
            "state_obs": args.state_obs,
            "spawn_cfg": spawn_cfg,
            "start_delay": 6.0,
            "sim_args": ["--rl"],
            "cam_config": {"img_enc": img_enc},
        }
        if not args.state_obs:
            conf["cam_resolution"] = (120, 160, 3)
        return gym.make("donkey-mountain-track-v0", conf=conf)

    return _init


def main():
    args = parse_args()
    spawn_cfg = dict(SPAWN_DEFAULT) if args.spawn else None

    # vision 模式必须出窗口:sim 的 headless 是 -batchmode -nographics,无图形设备相机渲染不出图
    headless = args.state_obs
    img_enc = "JPG" if not args.state_obs else args.img_enc

    print("[rl_train] %d 个 worker,sim 端口分配:" % args.envs)
    for rank in range(args.envs):
        print("  env[%d] -> 127.0.0.1:%d" % (rank, args.port_base + rank))
    print("[rl_train] sync_mode=%s state_obs=%s img_enc=%s spawn=%s seed=%d"
          % (args.sync_mode, args.state_obs, img_enc, spawn_cfg, args.seed))

    envs = SubprocVecEnv([make_env(rank, args, spawn_cfg, headless, img_enc) for rank in range(args.envs)])
    envs = VecMonitor(envs)

    policy = "MlpPolicy" if args.state_obs else "CnnPolicy"
    model = PPO(
        policy,
        envs,
        n_steps=512,
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        seed=args.seed,
        tensorboard_log=args.tb,
        verbose=1,
    )

    ckpt = CheckpointCallback(save_freq=5000, save_path="outputs/rl_ppo_ckpt/", name_prefix="ppo")
    model.learn(args.steps, callback=ckpt)
    model.save(args.save)
    print("[rl_train] done, model saved to %s" % args.save)

    envs.close()


if __name__ == "__main__":
    main()
