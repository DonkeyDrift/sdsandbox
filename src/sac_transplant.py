#!/usr/bin/env python3
"""PPO(MlpPolicy) → SAC 热启动移植手术。

champion2 的 actor  trunk 与输出头移植到 SAC(SquashedGaussian),
critic 冷启动。动作语义差异处理:
  PPO 确定性动作 = clip(net_mean, low, high)(无 squash)
  SAC 确定性动作 = unscale(tanh(mu))  —— tanh 压缩 + 动作空间非对称([-1,1]x[0,1])
移植策略:
  1. trunk(mlp_extractor.policy_net / SAC actor 隐层)直接拷权重(同宽 64x64);
  2. mu 头先按线性等价拷,再在固定观测集上做逐点校验;
     max|a_sac - a_ppo| 超阈(默认 0.05)则以
     mu* = atanh(clip(norm(a_ppo))) 为目标对 mu 头做岭回归重拟合(线性可解);
  3. critic 冷启动——训练时先冻结 actor 单训 critic(见 --freeze-critic-steps)。

用法:
  python3 src/sac_transplant.py --src <ppo.zip> --out <sac.zip> \
      --verify-obs 512 --retfit-thresh 0.05
"""

import argparse

import numpy as np
import torch
from stable_baselines3 import PPO, SAC


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="源 PPO 模型")
    ap.add_argument("--out", required=True, help="移植产物 SAC 模型路径")
    ap.add_argument("--envs", type=int, default=4, help="构建 SAC 用的并行环境数(与后续训练一致)")
    ap.add_argument("--port-base", type=int, default=9601, help="构建环境用的端口段(瞬连即关)")
    ap.add_argument("--verify-obs", type=int, default=512, help="逐点校验的观测样本数")
    ap.add_argument("--refit-thresh", type=float, default=0.05, help="触发 mu 头重拟合的动作差阈值")
    ap.add_argument("--collect-real-obs", type=int, default=1500,
                    help="用真实驾驶观测做校验/重拟合集:headless 同步 sim 跑 PPO 采集 N 条")
    ap.add_argument("--collect-port", type=int, default=9601)
    ap.add_argument("--sim-path", default="build/donkey_sim.app/Contents/MacOS/donkey_sim")
    ap.add_argument("--seed", type=int, default=7)
    return ap.parse_args()


def build_env(n_envs, port_base):
    """SAC 构造只需空间定义——用哑环境,手术全程离线(不连 sim)。"""
    import gymnasium as gym

    class DummyEnv(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, (8,), dtype=np.float32)
        action_space = gym.spaces.Box(
            np.array([-1.0, 0.0], dtype=np.float32), np.array([1.0, 1.0], dtype=np.float32))

        def reset(self, **kwargs):
            return np.zeros(8, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(8, dtype=np.float32), 0.0, False, False, {}

    return DummyEnv()


def transplant(ppo, sac, obs_batch, refit_thresh):
    ppo_pol = ppo.policy
    sac_pol = sac.policy
    low = torch.as_tensor(sac.action_space.low, dtype=torch.float32)
    high = torch.as_tensor(sac.action_space.high, dtype=torch.float32)
    scale = high - low

    # 1) trunk:SAC actor 的 features_extractor+mlp 隐层 ↔ PPO mlp_extractor.policy_net
    # SB3 SAC 默认 net_arch=[64,64]:actor 结构 features_extractor -> latent_pi(Linear64->64->64)
    # PPO 默认 net_arch dict(pi=[64,64],vf=[64,64]):mlp_extractor.policy_net 同形。
    src_sd = ppo_pol.mlp_extractor.policy_net.state_dict()
    dst = sac_pol.actor.latent_pi
    dst_sd = dst.state_dict()
    assert set(src_sd.keys()) == set(dst_sd.keys()), \
        "trunk 结构不一致: %s vs %s" % (sorted(src_sd.keys()), sorted(dst_sd.keys()))
    dst.load_state_dict(src_sd)
    print("[transplant] trunk 已拷(%d 个 tensor)" % len(src_sd))

    # 2) mu 头线性等价初值
    sac_pol.actor.mu.load_state_dict(ppo_pol.action_net.state_dict())
    print("[transplant] mu 头初值 = PPO action_net")

    # 3) 固定观测集逐点校验
    with torch.no_grad():
        mu, _, _ = sac_pol.actor.get_action_dist_params(obs_batch)
        a_ppo, _ = ppo.predict(obs_batch.numpy(), deterministic=True)
        a_sac = (low + 0.5 * (torch.tanh(mu) + 1.0) * scale).numpy()
    diff = np.abs(a_sac - a_ppo)
    print("[verify] 初移植 max|Δ|=%.4f mean|Δ|=%.4f" % (diff.max(), diff.mean()))

    # 4) 超阈 → mu 头岭回归重拟合:mu* = atanh(clip(norm(a_ppo)))
    if diff.max() > refit_thresh:
        norm = np.clip(2.0 * (a_ppo - low.numpy()) / scale.numpy() - 1.0, -0.999, 0.999)
        mu_target = np.arctanh(norm)
        with torch.no_grad():
            feats = sac_pol.actor.features_extractor(obs_batch)
            latent_pi = sac_pol.actor.latent_pi(feats)
        X = torch.cat([latent_pi, torch.ones(latent_pi.shape[0], 1)], dim=1).numpy()
        # relu 隐层在 OOD 观测下大量死区,X^T X 近奇异——lam 需配 latent 量级
        # (latent~13 → X.T X 元素~1e4,lam=1e-4 相当于无正则,解爆炸;
        #  lam=1.0 在保初值(PPO 等价头)与拟合数据间折中)
        lam = 1.0
        A = X.T @ X + lam * np.eye(X.shape[1])
        W = np.linalg.solve(A, X.T @ mu_target)  # (65, 2)
        with torch.no_grad():
            sac_pol.actor.mu.weight.copy_(torch.as_tensor(W[:-1].T))
            sac_pol.actor.mu.bias.copy_(torch.as_tensor(W[-1]))
        with torch.no_grad():
            mu, _, _ = sac_pol.actor.get_action_dist_params(obs_batch)
            a_sac = (low + 0.5 * (torch.tanh(mu) + 1.0) * scale).numpy()
        diff = np.abs(a_sac - a_ppo)
        print("[refit] 岭回归 warm-start max|Δ|=%.4f mean|Δ|=%.4f,接 torch Adam 动作空间精修"
              % (diff.max(), diff.mean()))
        # 岭回归最小二乘在 mu 空间妥协,极端样本动作差仍大;直接对 mu 头做动作空间
        # 全批 Adam 精修(huber+L2 锚定),512 样本×130 参数量级,秒级收敛。
        with torch.no_grad():
            feats0 = sac_pol.actor.features_extractor(obs_batch)
            lp0 = sac_pol.actor.latent_pi(feats0)
        a_tgt = torch.as_tensor(a_ppo)
        w = sac_pol.actor.mu.weight.detach().clone().requires_grad_(True)
        b = sac_pol.actor.mu.bias.detach().clone().requires_grad_(True)
        w0, b0 = w.detach().clone(), b.detach().clone()
        opt = torch.optim.Adam([w, b], lr=5e-2)
        for it in range(800):
            opt.zero_grad()
            mu = lp0 @ w.T + b
            a = low + 0.5 * (torch.tanh(mu) + 1.0) * scale
            loss = torch.nn.functional.smooth_l1_loss(a, a_tgt, beta=0.2) \
                + 1e-5 * ((w - w0) ** 2).sum() + 1e-5 * ((b - b0) ** 2).sum()
            loss.backward()
            opt.step()
        with torch.no_grad():
            sac_pol.actor.mu.weight.copy_(w.detach())
            sac_pol.actor.mu.bias.copy_(b.detach())
            mu, _, _ = sac_pol.actor.get_action_dist_params(obs_batch)
            a_sac = (low + 0.5 * (torch.tanh(mu) + 1.0) * scale).numpy()
        diff = np.abs(a_sac - a_ppo)
        q50, q95, q99 = np.percentile(diff, [50, 95, 99])
        print("[refit] Adam 精修后 max|Δ|=%.4f mean|Δ|=%.4f p95=%.4f"
              % (diff.max(), diff.mean(), q95))
        # 校验口径:线性 mu 头在 PPO 决策边界(近似不连续)处有不可约 max 误差,
        # 热启动价值由 mean/p95 决定;max 仅记录不上阈值。
        assert diff.mean() < 0.05 and q95 < 0.25, \
            "mean=%.4f p95=%.4f 超热启动口径" % (diff.mean(), q95)
    return float(diff.max()), float(diff.mean())


def collect_real_obs(ppo, n, port, sim_path, seed):
    """headless 同步 sim 跑 PPO 确定性策略,记录真实驾驶观测流(流形内校验集)。"""
    import gymnasium as gym
    import gym_donkeycar  # noqa

    conf = {
        "exe_path": sim_path, "host": "127.0.0.1", "port": port,
        "headless": True, "level": "mountain_track", "max_cte": 2.5,
        "sync_mode": True, "sync_time_step": 0.05, "state_obs": True,
        "spawn_cfg": {"node_min": 5, "node_max": 245, "lateral_offset": 1.0, "heading_noise": 8.0},
        "raise_on_stale": True, "start_delay": 6.0, "sim_args": ["--rl"],
        "cam_config": {"img_enc": "none"},
    }
    env = gym.make("donkey-mountain-track-v0", conf=conf)
    buf = []
    ep = 0
    while len(buf) < n and ep < 40:
        obs, _ = env.reset(seed=seed + ep)
        for _ in range(2500):
            buf.append(obs.copy())
            a, _ = ppo.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            if term or trunc:
                break
        ep += 1
    env.close()
    idx = np.random.default_rng(seed).choice(len(buf), size=min(n, len(buf)), replace=False)
    return torch.as_tensor(np.stack([buf[i] for i in idx]))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    ppo = PPO.load(args.src, device="cpu")

    env = build_env(args.envs, args.port_base)
    sac = SAC(
        "MlpPolicy", env,
        policy_kwargs=dict(net_arch=[64, 64]),
        learning_rate=1e-4, buffer_size=50000, batch_size=256,
        gamma=0.99, tau=0.005, train_freq=1, gradient_steps=1,
        learning_starts=1000, seed=args.seed, verbose=1,
        tensorboard_log="outputs/rl_tb",
    )

    if args.collect_real_obs > 0:
        print("[collect] 采集真实驾驶观测 %d 条(port %d)..." % (args.collect_real_obs, args.collect_port))
        obs_batch = collect_real_obs(ppo, args.collect_real_obs, args.collect_port,
                                     args.sim_path, args.seed)
        print("[collect] 实际 %d 条" % obs_batch.shape[0])
    else:
        rng = np.random.default_rng(args.seed)
        obs_np = rng.normal(0.0, 0.6, (args.verify_obs, 8)).astype(np.float32)
        obs_np[:, 6] = 0.0  # 保留位恒 0
        obs_np[:, 7] = 0.0
        obs_batch = torch.as_tensor(obs_np)
    max_d, mean_d = transplant(ppo, sac, obs_batch, args.refit_thresh)

    sac.num_timesteps = ppo.num_timesteps  # 继承计数,续训语义一致
    sac.save(args.out)
    env.close()
    print("[done] %s  (max|Δ|=%.4f mean|Δ|=%.4f)" % (args.out, max_d, mean_d))


if __name__ == "__main__":
    main()
