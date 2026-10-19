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

import numpy as np
import gymnasium as gym
import gym_donkeycar  # noqa: F401 注册 donkey-* env
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
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
    parser.add_argument("--frame-stack", type=int, default=0,
                        help="视觉观测叠帧 N(仅 --vision):观测变 (3N,120,160),时序信息单因子;"
                             "注意与单帧模型不兼容(conv 输入通道数变),需从头训")
    parser.add_argument("--cam-dr", dest="cam_dr", action="store_true", default=False,
                        help="相机位姿域随机化(仅 --vision):每 reset 重发 cam_config," 
                             "offset ±2cm / rot ±1°(P3 因子 2,纯客户端)")
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
    parser.add_argument("--ctl-step", type=int, default=0,
                        help="量化异步控制步长(毫秒,相位协议):>0 时控制应用/遥测对齐物理网格边界,"
                             "动作-观测配对与同步 time_step 同构(异步训练配 --async-mode 用)")
    parser.add_argument("--keep-gfx", dest="keep_gfx", action="store_true", default=False,
                        help="异步状态量训练用 -batchmode 离屏渲染(不弹窗但主循环全速,量化遥测需要;headless 下异步流可能 ~1fps)")
    parser.add_argument("--freeze-actor-steps", type=int, default=0,
                        help="SAC 热启动专用:前 N 步冻结 actor 只训 critic(critic 冷启动保护),N 步后解冻")
    parser.add_argument("--max-cte", type=float, default=2.5, help="出界 cross-track 误差阈值")
    parser.add_argument("--action-hold-prob", type=float, default=0.0,
                        help="控制延迟课程:每步以该概率保持上一动作(异步相位差模拟,Phase2 用 0.3)")
    parser.add_argument("--action-hold-max", type=int, default=2,
                        help="控制延迟课程:单次最多连续保持 1..N 个物理步")
    parser.add_argument("--action-delay", type=int, default=0,
                        help="真·动作延迟(步):执行 N 步之前的动作,当前动作入队——精确模拟量化异步协议 "
                             "(ctl_step_ms)的固有死时间;1 = 延迟一步(与 --ctl-step 50 协议等价)")
    parser.add_argument("--action-jitter", type=int, default=0,
                        help="延迟抖动域随机(需配 --action-delay):每步从最近 K 个动作里均匀随机执行一个,"
                             "覆盖真实异步的时钟抖动/控制丢帧(策略动作偶发被跳过或重复)")
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
        elif args.keep_gfx:
            conf["headless_keep_gfx"] = True  # 量化异步训练:主循环全速,边界遥测不依附渲染
        if args.cam_fps > 0:
            conf["cam_config"]["fps"] = args.cam_fps  # 数值型:fork 有 fps>0 数值比较
        if args.ctl_step > 0:
            conf["ctl_step_ms"] = args.ctl_step  # 量化异步控制(相位协议)
        env = gym.make("donkey-mountain-track-v0", conf=conf)
        if args.cam_dr:
            if args.state_obs:
                raise SystemExit("--cam-dr 仅视觉观测适用")
            env = CamDRWrapper(env)  # 必须在帧stack/延迟包装之内层(直接包原生 env)
        if args.action_hold_prob > 0:
            env = StickyActionWrapper(env, hold_prob=args.action_hold_prob,
                                      hold_max=args.action_hold_max)
        if args.action_delay > 0:
            env = ActionDelayWrapper(env, delay=args.action_delay, jitter=args.action_jitter)
        if time_limit:
            # 本环境不自带步数上限(只 terminated 不 truncated);EvalCallback 的
            # evaluate_policy 没有步数预算,好策略永不撞墙会把评测永远挂住。
            env = gym.wrappers.TimeLimit(env, max_episode_steps=time_limit)
        return env

    return _init


class StickyActionWrapper(gym.Wrapper):
    """控制延迟课程(Phase 2 异步鲁棒):每步以 hold_prob 概率保持上一动作,
    单次连续保持 1..hold_max 个物理步。同步模式内模拟实时控制的相位抖动——
    真实异步 20Hz 下动作实际作用于 5 个物理步、且遥测往返有相位差,
    直接训出的策略对 delay 敏感(冠军异步必撞);先在同步里注入 delay 再续训。"""

    def __init__(self, env, hold_prob=0.3, hold_max=2):
        super().__init__(env)
        self.hold_prob = hold_prob
        self.hold_max = hold_max
        self._held_action = None
        self._hold_remaining = 0

    def reset(self, **kwargs):
        self._held_action = None
        self._hold_remaining = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        if self._hold_remaining > 0 and self._held_action is not None:
            action = self._held_action
            self._hold_remaining -= 1
        else:
            self._held_action = action
            if self.env.unwrapped.np_random.random() < self.hold_prob:
                self._hold_remaining = int(self.env.unwrapped.np_random.integers(1, self.hold_max + 1))
        return self.env.step(action)


class ActionDelayWrapper(gym.Wrapper):
    """真·动作延迟:执行 N 步之前收到的动作,当前动作入队(先进先出)。
    与量化异步协议(ctl_step_ms)的固有死时间逐步同构:策略对 o_k 出的动作
    在 o_{k+1} 时刻应用、o_{k+2} 才可见(s_k = post-a_{k-2})。
    队初的 None 执行零动作(sim 复位后控制即零)。注意与 StickyActionWrapper
    的区别:后者是"保持并丢弃"奇数步动作,不是延迟——本类才是延迟语义。
    jitter>0(域随机):每步从最近 jitter 个已入队动作里均匀随机执行一个,
    覆盖真实异步的时钟抖动与控制丢帧(动作偶发被重复执行或跳过)。"""

    def __init__(self, env, delay=1, jitter=0):
        super().__init__(env)
        self.delay = delay
        self.jitter = jitter
        self._buf = None

    def _fresh(self):
        import collections
        self._buf = collections.deque([None] * self.delay, maxlen=self.delay + (self.jitter or 0))

    def reset(self, **kwargs):
        self._fresh()
        return self.env.reset(**kwargs)

    def step(self, action):
        if self._buf is None:
            self._fresh()
        self._buf.append(action)
        if self.jitter and len(self._buf) > 1:
            k = min(self.jitter, len(self._buf) - 1)
            idx = int(self.env.unwrapped.np_random.integers(0, k))
            old = list(self._buf)[-2 - idx]  # 延迟 1..jitter(不含 0,贴近真实协议)
        else:
            old = self._buf[0]
        if old is None:
            old = np.zeros_like(action)
        return self.env.step(old)


class ActorUnfreezeCallback(BaseCallback):
    """SAC 热启动:N 步后解冻 actor(critic 冷启动保护期结束)。"""

    def __init__(self, unfreeze_at):
        super().__init__()
        self.unfreeze_at = unfreeze_at
        self._done = False

    def _on_step(self):
        if not self._done and self.model.num_timesteps >= self.unfreeze_at:
            for p in self.model.policy.actor.parameters():
                p.requires_grad_(True)
            self._done = True
            print("[rl_train] num_timesteps=%d:actor 已解冻" % self.model.num_timesteps)
        return True


class CamDRWrapper(gym.Wrapper):
    """相机位姿域随机化(P3 因子 2,纯客户端):每次 reset 把相机外参抖动
    offset x/y/z ±2cm、rot x/y/z ±1° 重发 cam_config。仿真侧 SetConfig 对非零
    值生效、零值保持默认——重采避开精确零。需在帧stack等包装之外(直接包原生 env)。"""

    def __init__(self, env, pos_jitter=0.02, rot_jitter=1.0):
        super().__init__(env)
        self.pos_jitter = pos_jitter
        self.rot_jitter = rot_jitter

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        u = self.env.unwrapped
        handler = getattr(getattr(u, "viewer", None), "handler", None)
        if handler is None:
            raise RuntimeError("CamDRWrapper 找不到 viewer.handler")
        rng = u.np_random

        def draw(scale):
            v = 0.0
            while v == 0.0:
                v = float(rng.uniform(-scale, scale))
            return v

        handler.send_cam_config(
            offset_x=draw(self.pos_jitter), offset_y=draw(self.pos_jitter),
            offset_z=draw(self.pos_jitter),
            rot_x=draw(self.rot_jitter), rot_y=draw(self.rot_jitter),
            rot_z=draw(self.rot_jitter))
        return obs, info


def _close_vec_env(env):
    """收尾关闭 VecEnv。中断时 worker 可能卡在同步步进的 sim 等待上,优雅 close 会
    在 pipe.recv() 死等(gymnasium 对 idle worker 的 close 无超时);先给 10s 优雅窗口,
    不行就强制 terminate。terminate 路径 worker 的 sim 子进程会成为孤儿,
    按端口清理(运维铁律)。"""
    if env is None:
        return
    for kwargs in ({"timeout": 10}, {"terminate": True}):
        try:
            env.close(**kwargs)
            return
        except Exception:
            continue


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
    if args.frame_stack > 0:
        if args.state_obs:
            raise SystemExit("--frame-stack 仅视觉观测适用")
        from stable_baselines3.common.vec_env import VecFrameStack
        envs = VecFrameStack(envs, n_stack=args.frame_stack, channels_order="first")
        print("[rl_train] 视觉叠帧 x%d,观测 (3x%d,120,160)——从头训(与单帧模型通道不兼容)" % (
            args.frame_stack, args.frame_stack))

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

    # save_freq 是 vec-step 粒度(=步数/envs):换算成每 ~25k 环境步一档,长跑墙钟可预期
    ckpt_freq = max(1, 25000 // args.envs)
    callbacks = [CheckpointCallback(save_freq=ckpt_freq, save_path="outputs/rl_ckpt_%s/" % args.run_name,
                                    name_prefix=args.algo)]
    if args.eval_envs > 0:
        # 评测环境限时 2000 步(100s 仿真):600 步上限会被饱和型策略打满无法区分,
        # 且 evaluate_policy 无步数预算必须有上限兜底
        eval_env = SubprocVecEnv([make_env(args.envs, args, spawn_cfg, headless, img_enc, time_limit=2000)
                                  for _ in range(args.eval_envs)])
        eval_env = VecMonitor(eval_env)
        if args.frame_stack > 0:
            from stable_baselines3.common.vec_env import VecFrameStack
            eval_env = VecFrameStack(eval_env, n_stack=args.frame_stack, channels_order="first")
        callbacks.append(EvalCallback(
            eval_env, best_model_save_path="outputs/rl_best_%s/" % args.run_name,
            log_path="outputs/rl_best_%s/" % args.run_name,
            eval_freq=args.eval_freq, n_eval_episodes=3, deterministic=True,
        ))
        print("[rl_train] 自动评测开启:每 %d 次回调评 3 局,最佳模型存 outputs/rl_best_%s/" % (args.eval_freq, args.run_name))

    if args.freeze_actor_steps > 0:
        if args.algo != "sac":
            raise SystemExit("--freeze-actor-steps 仅 SAC 热启动适用")
        for p in model.policy.actor.parameters():
            p.requires_grad_(False)
        print("[rl_train] actor 已冻结,前 %d 步单训 critic" % args.freeze_actor_steps)
        callbacks.append(ActorUnfreezeCallback(args.freeze_actor_steps))

    try:
        model.learn(args.steps, callback=CallbackList(callbacks), reset_num_timesteps=not bool(args.resume),
                    tb_log_name=args.run_name)
        model.save(save_path)
        print("[rl_train] done, model saved to %s" % save_path)
    except BaseException as exc:
        # 异常/中断(含 KeyboardInterrupt、EvalCallback EOFError)也落盘保命存档,
        # 否则崩溃发生在第一个存档点之前 = 全部白跑(2026-10-18 踩过)
        crash_path = save_path + ".crash"
        try:
            model.save(crash_path)
            print("[rl_train] 异常 %r:已保存保命存档 %s" % (exc, crash_path))
        except Exception:
            print("[rl_train] 异常 %r:保命存档也失败" % (exc,))
        raise
    finally:
        _close_vec_env(envs)
        if args.eval_envs > 0:
            _close_vec_env(eval_env)


if __name__ == "__main__":
    main()
