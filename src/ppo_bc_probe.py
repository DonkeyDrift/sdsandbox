#!/usr/bin/env python3
"""PPOBC 探针三件套（起训前硬门）：
  ① BC loss 梯度非零（action_net / features_extractor 均有非平凡 grad）
  ② λ=0 时同 seed 与原版 PPO learn 后参数逐位一致
  ③ 降采样对齐：真实人驾帧过 load_bc_dataset 后 shape/值域/空间对应正确

用法：python3 src/ppo_bc_probe.py
全绿 exit 0；任一红打印 FAIL 并 exit 1。
"""

import glob
import os
import sys

import numpy as np
import torch as th
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_train import PPOBC, load_bc_dataset  # noqa: E402

BC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "outputs", "bc_human_data")
OUT_H, OUT_W = 120, 160


class _DummyImageEnv(gym.Env):
    """随机图像 + 2 维连续动作的合成环境（不依赖 sim）。"""

    metadata = {"render_modes": []}

    def __init__(self):
        self.observation_space = spaces.Box(0, 255, (OUT_H, OUT_W, 3), np.uint8)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self._rng = np.random.default_rng(0)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        obs = self._rng.integers(0, 255, (OUT_H, OUT_W, 3), dtype=np.uint8)
        return obs, {}

    def step(self, action):
        obs = self._rng.integers(0, 255, (OUT_H, OUT_W, 3), dtype=np.uint8)
        return obs, 0.0, False, False, {}


def _make_vec(seed=0):
    env = DummyVecEnv([lambda: _DummyImageEnv()])
    env = VecMonitor(env)
    env.seed(seed)
    return env


def probe1_grad_nonzero():
    print("=== probe1: BC loss 梯度非零 ===")
    env = _make_vec(seed=1)
    model = PPOBC("CnnPolicy", env, n_steps=64, batch_size=32, n_epochs=1,
                  seed=1, bc_lambda=0.2, bc_data_dir=BC_DIR, bc_batch=64,
                  verbose=0)
    model._ensure_bc_data()
    model.policy.set_training_mode(True)
    model.policy.optimizer.zero_grad()
    obs, acts = model._sample_bc_batch()
    assert obs.shape == (64, 3, OUT_H, OUT_W), obs.shape
    assert obs.dtype == th.float32 and float(obs.min()) >= 0.0 and float(obs.max()) <= 1.0
    bc_loss = model._bc_mean_loss(obs, acts)
    assert th.isfinite(bc_loss), bc_loss
    bc_loss.backward()

    def _gn(module):
        return sum(float(p.grad.detach().abs().sum())
                   for p in module.parameters() if p.grad is not None)

    g_act = _gn(model.policy.action_net)
    g_ext = _gn(model.policy.features_extractor)
    g_all = sum(float(p.grad.detach().abs().sum())
                for p in model.policy.parameters() if p.grad is not None)
    print("  bc_loss=%.6f  action_net|g|=%.6e  features_extractor|g|=%.6e  all|g|=%.6e"
          % (bc_loss.item(), g_act, g_ext, g_all))
    env.close()
    assert g_act > 1e-6, "action_net 梯度为零/过小(假回传)"
    assert g_ext > 1e-6, "features_extractor 梯度为零/过小"
    assert g_all > 1e-6
    print("probe1 PASS")


def _mk_ppo(seed, env):
    return PPO("CnnPolicy", env, n_steps=128, batch_size=64, n_epochs=2,
               gamma=0.99, gae_lambda=0.95, ent_coef=0.01, learning_rate=3e-4,
               seed=seed, verbose=0)


def probe2_lambda0_bitwise():
    print("=== probe2: λ=0 与原版 PPO 逐位一致 ===")
    env_a = _make_vec(seed=42)
    env_b = _make_vec(seed=42)
    ppo = _mk_ppo(42, env_a)
    ppo.learn(256, reset_num_timesteps=True)
    ppobc = PPOBC("CnnPolicy", env_b, n_steps=128, batch_size=64, n_epochs=2,
                  gamma=0.99, gae_lambda=0.95, ent_coef=0.01, learning_rate=3e-4,
                  seed=42, verbose=0, bc_lambda=0.0, bc_data_dir=BC_DIR,
                  bc_batch=256)
    assert ppobc._current_lambda() == 0.0
    ppobc.learn(256, reset_num_timesteps=True)
    n_diff = 0
    max_diff = 0.0
    for (n1, p1), (_, p2) in zip(ppo.policy.named_parameters(),
                                 ppobc.policy.named_parameters()):
        if not th.equal(p1.data, p2.data):
            n_diff += 1
            max_diff = max(max_diff, float((p1.data - p2.data).abs().max()))
    print("  n_params_diff=%d  max_diff=%.3e" % (n_diff, max_diff))
    env_a.close()
    env_b.close()
    assert n_diff == 0, "λ=0 仍污染训练流: %d 个参数不一致 max_diff=%.3e" % (n_diff, max_diff)
    print("probe2 PASS")


def probe3_downsample_align():
    print("=== probe3: 降采样对齐 ===")
    paths = sorted(glob.glob(os.path.join(BC_DIR, "bc_*.npz")))
    assert paths, "无 bc 数据: %s" % BC_DIR
    d0 = np.load(paths[0])
    src = np.asarray(d0["image"])  # N,240,320,3
    n0 = len(src)
    # 小切片走完整管线（避免全量加载拖慢探针，但走同一函数）
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        np.savez(os.path.join(td, "bc_00000.npz"),
                 image=src[:8], steer=np.asarray(d0["steer"][:8]),
                 throttle=np.asarray(d0["throttle"][:8]),
                 meta=np.asarray(d0["meta"][:8]))
        obs, acts = load_bc_dataset(td, out_hw=(OUT_H, OUT_W))
    assert obs.shape == (8, 3, OUT_H, OUT_W), obs.shape
    assert obs.dtype == np.uint8
    assert acts.shape == (8, 2), acts.shape
    # 值域与通道均值对应（HWC->CHW 转置正确）
    frame0 = obs[0].transpose(1, 2, 0)  # H,W,3
    src0 = src[0]
    ch_mean_obs = frame0.reshape(-1, 3).mean(axis=0)
    ch_mean_src = src0.reshape(-1, 3).mean(axis=0)
    print("  obs=%s  ch_mean_obs=%s  ch_mean_src=%s"
          % (obs.shape, np.round(ch_mean_obs, 1), np.round(ch_mean_src, 1)))
    assert np.allclose(ch_mean_obs, ch_mean_src, atol=6.0), \
        "通道均值偏离过大，疑似布局错位"
    # 空间对应：左半/右半亮度相对关系应与源图一致
    left_obs, right_obs = frame0[:, : OUT_W // 2].mean(), frame0[:, OUT_W // 2:].mean()
    left_src, right_src = src0[:, : src0.shape[1] // 2].mean(), src0[:, src0.shape[1] // 2:].mean()
    print("  left/right obs=%.1f/%.1f src=%.1f/%.1f"
          % (left_obs, right_obs, left_src, right_src))
    # 符号一致即可（允许源图本身近似均匀）
    if abs(left_src - right_src) > 2.0:
        assert np.sign(left_obs - right_obs) == np.sign(left_src - right_src), "左右空间翻转"
    # 帧间有差异（真图像非灰图）
    diff01 = np.abs(obs[0].astype(np.int16) - obs[1].astype(np.int16)).mean()
    print("  frame0-frame1 mean|d|=%.2f" % diff01)
    assert diff01 > 1.0, "连续帧几乎相同，数据可能有问题"
    # 动作值域
    assert float(acts[:, 0].min()) >= -1.01 and float(acts[:, 0].max()) <= 1.01
    assert float(acts[:, 1].min()) >= -0.01 and float(acts[:, 1].max()) <= 1.01
    print("  n_source_frames=%d  probe n0=%d" % (len(src), n0))
    print("probe3 PASS")


def main():
    th.manual_seed(0)
    np.random.seed(0)
    probe1_grad_nonzero()
    probe2_lambda0_bitwise()
    probe3_downsample_align()
    print("ALL PROBES PASS")


if __name__ == "__main__":
    main()
