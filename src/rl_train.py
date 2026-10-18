#!/usr/bin/env python3
"""RL 训练入口:PPO/SAC + sdsandbox DonkeySim 新协议(同步步进/状态量观测/随机出生)。

每个 worker 起一个独立 sim 实例(--rl 模式),端口 port_base+rank:
  python3 src/rl_train.py                       # 默认:PPO、2 env、同步步进、状态量观测
  python3 src/rl_train.py --envs 4 --steps 500000
  python3 src/rl_train.py --resume outputs/rl_ppo.zip --steps 500000   # 断点续训(曲线还在涨就接着训)
  python3 src/rl_train.py --algo sac --envs 1 --steps 100000           # 平台期后换样本效率更高的 SAC
  python3 src/rl_train.py --vision --envs 1     # 图像观测(CNN),sim 需出窗口渲染
"""

import argparse
import os

import gymnasium as gym
import gym_donkeycar  # noqa: F401 注册 donkey-* env
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

SPAWN_DEFAULT = {
    "node_min": 5,
    "node_max": 245,
    "lateral_offset": 1.0,
    "heading_noise": 8.0,
}


def parse_args():
    parser = argparse.ArgumentParser(description="RL training against sdsandbox DonkeySim")
    parser.add_argument("--algo", choices=["ppo", "sac"], default="ppo", help="PPO(默认,适合多env) / SAC(平台期精修,样本效率高)")
    parser.add_argument("--run-name", default="run", help="运行实例名:best/ckpt 目录按名隔离(并行多训练必须不同名)")
    parser.add_argument("--resume", default=None, help="从已有模型继续训练(如 outputs/rl_ppo.zip)")
    parser.add_argument("--lr", type=float, default=None,
                        help="覆盖学习率(续训已收敛模型建议 1e-4,默认按算法内置)")
    parser.add_argument("--ent-coef", type=float, default=None,
                        help="覆盖 PPO 熵系数(默认 0.01;策略退化时可降到 0.003)")
    parser.add_argument("--envs", type=int, default=2, help="并行训练 sim/worker 数")
    parser.add_argument("--eval-envs", type=int, default=1, help="评测专用 sim 数(0=关闭自动评测/最佳模型保存)")
    parser.add_argument("--eval-freq", type=int, default=2500, help="评测间隔(回调次数,4 env 时约每 1 万环境步)")
    parser.add_argument("--train-time-limit", type=int, default=2500,
                        help="训练 episode 步数上限(默认 2500):强策略永不撞墙时强制回合循环,刷新随机出生")
    parser.add_argument("--steps", type=int, default=20000, help="本次训练总步数(续训时累计)")
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
    parser.add_argument("--spawn-lateral", type=float, default=1.0, help="出生横向偏移幅度(米,救车课程用 1.8)")
    parser.add_argument("--spawn-heading", type=float, default=8.0, help="出生朝向噪声(度)")
    parser.add_argument("--spawn-node-min", type=int, default=5, help="出生节点范围下限(针对性课程用)")
    parser.add_argument("--spawn-node-max", type=int, default=245, help="出生节点范围上限")
    parser.add_argument("--cam-fps", type=float, default=0.0,
                        help="异步训练遥测/控制频率(Hz,0=默认20):异步鲁棒微调用 40")
    parser.add_argument("--max-cte", type=float, default=2.5, help="出界 cross-track 误差阈值")
    parser.add_argument("--seed", type=int, default=1, help="训练/出生随机种子")
    parser.add_argument("--save", default=None, help="最终模型保存路径(默认 outputs/rl_<algo>.zip)")
    parser.add_argument("--tb", default="outputs/rl_tb", help="tensorboard 日志目录")
    return parser.parse_args()


def make_env(rank: int, args, spawn_cfg, headless, img_enc, time_limit=None):
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
            "raise_on_stale": True,
            "start_delay": 6.0,
            "sim_args": ["--rl"],
            "cam_config": {"img_enc": img_enc},
        }
        if not args.state_obs:
            conf["cam_resolution"] = (120, 160, 3)
            conf["headless_keep_gfx"] = True  # 视觉用 -batchmode 离屏渲染:无窗口、无遮挡休眠
        if args.cam_fps > 0:
            conf["cam_config"]["fps"] = args.cam_fps  # 数值型:fork 有 fps>0 数值比较
        env = gym.make("donkey-mountain-track-v0", conf=conf)
        if time_limit:
            # 本环境不自带步数上限(只 terminated 不 truncated);EvalCallback 的
            # evaluate_policy 没有步数预算,好策略永不撞墙会把评测永远挂住。
            env = gym.wrappers.TimeLimit(env, max_episode_steps=time_limit)
        return env

    return _init


def main():
    args = parse_args()
    spawn_cfg = {"node_min": args.spawn_node_min, "node_max": args.spawn_node_max,
                 "lateral_offset": args.spawn_lateral, "heading_noise": args.spawn_heading} if args.spawn else None
    save_path = args.save or ("outputs/rl_%s.zip" % args.algo)
    if args.resume and os.path.abspath(save_path) == os.path.abspath(args.resume):
        # 续训时默认保存路径与来源相同会把结果写回源文件;一旦中途/末尾保存,
        # 源模型的步数计数与权重被污染,后续续训全部错位(2026-10-18 踩过)。
        save_path = "outputs/rl_%s_resumeout.zip" % args.algo
        print("[rl_train] 警告: --save 与 --resume 同源,自动改为 %s" % save_path)

    # vision 模式必须出窗口:sim 的 headless 是 -batchmode -nographics,无图形设备相机渲染不出图
    headless = args.state_obs
    img_enc = "JPG" if not args.state_obs else args.img_enc

    print("[rl_train] %d 个训练 worker + %d 个评测 worker,sim 端口分配:" % (args.envs, args.eval_envs))
    for rank in range(args.envs + args.eval_envs):
        print("  port %d -> %s" % (args.port_base + rank, "train[%d]" % rank if rank < args.envs else "eval"))
    print("[rl_train] algo=%s sync_mode=%s state_obs=%s img_enc=%s spawn=%s resume=%s seed=%d"
          % (args.algo, args.sync_mode, args.state_obs, img_enc, spawn_cfg, args.resume, args.seed))

    envs = SubprocVecEnv([make_env(rank, args, spawn_cfg, headless, img_enc,
                                   time_limit=args.train_time_limit) for rank in range(args.envs)])
    envs = VecMonitor(envs)

    policy = "MlpPolicy" if args.state_obs else "CnnPolicy"
    if args.resume:
        ctor = PPO if args.algo == "ppo" else SAC
        model = ctor.load(args.resume, env=envs, tensorboard_log=args.tb, verbose=1)
        if args.lr is not None:
            model.learning_rate = args.lr  # 覆盖已存学习率(续训已收敛模型必配)
        print("[rl_train] resumed from %s lr=%s" % (args.resume, args.lr))
    elif args.algo == "ppo":
        model = PPO(
            policy, envs,
            n_steps=512, batch_size=128, gamma=0.99, gae_lambda=0.95,
            ent_coef=args.ent_coef if args.ent_coef is not None else 0.01,
            learning_rate=args.lr if args.lr is not None else 3e-4,
            seed=args.seed, tensorboard_log=args.tb, verbose=1,
        )
    else:
        model = SAC(
            policy, envs,
            buffer_size=50000, learning_starts=1000, batch_size=256,
            gamma=0.99, train_freq=1, gradient_steps=1,
            learning_rate=args.lr if args.lr is not None else 3e-4,
            seed=args.seed, tensorboard_log=args.tb, verbose=1,
        )

    callbacks = [CheckpointCallback(save_freq=5000, save_path="outputs/rl_ckpt_%s/" % args.run_name,
                                    name_prefix=args.algo)]
    if args.eval_envs > 0:
        # 评测环境限时 2000 步(100s 仿真):600 步上限会被饱和型策略打满无法区分,
        # 且 evaluate_policy 无步数预算必须有上限兜底
        eval_env = SubprocVecEnv([make_env(args.envs, args, spawn_cfg, headless, img_enc, time_limit=2000)
                                  for _ in range(args.eval_envs)])
        eval_env = VecMonitor(eval_env)
        callbacks.append(EvalCallback(
            eval_env, best_model_save_path="outputs/rl_best_%s/" % args.run_name,
            log_path="outputs/rl_best_%s/" % args.run_name,
            eval_freq=args.eval_freq, n_eval_episodes=3, deterministic=True,
        ))
        print("[rl_train] 自动评测开启:每 %d 次回调评 3 局,最佳模型存 outputs/rl_best_%s/" % (args.eval_freq, args.run_name))

    model.learn(args.steps, callback=CallbackList(callbacks), reset_num_timesteps=not bool(args.resume))
    model.save(save_path)
    print("[rl_train] done, model saved to %s" % save_path)

    envs.close()


if __name__ == "__main__":
    main()
