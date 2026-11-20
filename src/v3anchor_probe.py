#!/usr/bin/env python3
"""[V3-ANCHOR] 探针三件套（起训前硬门）：
  ① λ=0 resume 逐位一致：vision3 载入 + MethodType 绑 PPOBC 方法 + λ=0，
    train() 与原版 PPO.train 在同一 rollout buffer 上参数逐位一致
    （覆盖 resume 特有的 super() TypeError 坑——显式 PPO.train(self) 修复验证）。
  ② BC loss 梯度非零：CNN action_net / features_extractor 均有非平凡 grad
    （自锚数据 outputs/bc_v3anchor 同管线）。
  ③ 自锚数据管线：120×160 源图过 load_bc_dataset 后 shape/值域/通道对齐。

用法：python3 src/v3anchor_probe.py
全绿 exit 0；任一红打印 FAIL 并 exit 1。
"""

import copy
import glob
import os
import sys
import tempfile

import numpy as np
import torch as th
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.buffers import RolloutBuffer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_train import PPOBC, load_bc_dataset  # noqa: E402

ANCHOR_DIR = os.environ.get(
    "V3_ANCHOR_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "outputs", "bc_v3anchor"))
VISION3 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "outputs", "rl_vision3.zip")
OUT_H, OUT_W = 120, 160
RESUME_METHODS = ("train", "_current_lambda", "_ensure_bc_data",
                  "_sample_bc_batch", "_bc_mean_loss", "save")


class _ChwImageEnv(gym.Env):
    """合成 CHW 图像环境（对齐 vision3 obs space (3,120,160) uint8）。"""

    metadata = {"render_modes": []}

    def __init__(self):
        self.observation_space = spaces.Box(0, 255, (3, OUT_H, OUT_W), np.uint8)
        self.action_space = spaces.Box(
            np.array([-1.0, 0.0], dtype=np.float32),
            np.array([1.0, 1.0], dtype=np.float32))
        self._rng = np.random.default_rng(0)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        obs = self._rng.integers(0, 255, (3, OUT_H, OUT_W), dtype=np.uint8)
        return obs, {}

    def step(self, action):
        obs = self._rng.integers(0, 255, (3, OUT_H, OUT_W), dtype=np.uint8)
        return obs, 0.0, False, False, {}


def _mk_vec(seed=0):
    env = DummyVecEnv([lambda: _ChwImageEnv()])
    env.seed(seed)
    return env


def probe1_lambda0_resume_bitwise():
    print("=== probe1: λ=0 resume 与 PPO.train 逐位一致 ===")
    env = _mk_vec(seed=7)
    model = PPO.load(VISION3, env=env, device="cpu")
    assert model.observation_space.shape == (3, OUT_H, OUT_W), model.observation_space

    # 收集一份 rollout buffer(单次)
    from stable_baselines3.common.callbacks import BaseCallback

    class _Noop(BaseCallback):
        def _on_step(self):
            return True

    model._setup_learn(model.n_steps, False)
    cb = _Noop()
    cb.init_callback(model)
    model.collect_rollouts(model.env, cb, model.rollout_buffer, model.n_steps)
    buf = model.rollout_buffer

    # 臂 A:原版 PPO.train(原地)。minibatch 顺序走全局 np.random.permutation,
    # 两臂 train 前重播种保证同序。
    np.random.seed(123)
    model.train()
    params_a = {n: p.detach().clone() for n, p in model.policy.named_parameters()}
    env.close()

    # 臂 B:重载同权重 → resume 路径(MethodType 绑 PPOBC 方法 + λ=0)
    import types as _t
    from stable_baselines3.common.logger import Logger, HumanOutputFormat
    arm_b = PPO.load(VISION3, device="cpu")
    arm_b._logger = Logger("probe", [HumanOutputFormat(sys.stdout)])
    for m in RESUME_METHODS:
        setattr(arm_b, m, _t.MethodType(getattr(PPOBC, m), arm_b))
    arm_b.bc_lambda = 0.0
    arm_b.bc_data_dir = ANCHOR_DIR
    arm_b.bc_batch = 256
    arm_b.bc_anneal_from = 50000
    arm_b.bc_anneal_to = 100000
    arm_b._bc_obs = None
    arm_b._bc_acts = None
    arm_b._bc_rng = np.random.default_rng(0)
    arm_b.rollout_buffer = buf
    np.random.seed(123)
    arm_b.train()

    n_diff = 0
    max_diff = 0.0
    params_b = dict(arm_b.policy.named_parameters())
    for n, p1 in params_a.items():
        p2 = params_b[n]
        if not th.equal(p1, p2.data):
            n_diff += 1
            max_diff = max(max_diff, float((p1 - p2.data).abs().max()))
    print("  n_params_diff=%d  max_diff=%.3e" % (n_diff, max_diff))
    assert n_diff == 0, \
        "λ=0 resume 仍污染训练流: %d 参数不一致 max_diff=%.3e" % (n_diff, max_diff)
    print("probe1 PASS")


def probe2_grad_nonzero():
    print("=== probe2: 自锚 BC loss 梯度非零 ===")
    env = _mk_vec(seed=1)
    model = PPOBC("CnnPolicy", env, n_steps=64, batch_size=32, n_epochs=1,
                  seed=1, bc_lambda=0.2, bc_data_dir=ANCHOR_DIR, bc_batch=64,
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
    print("probe2 PASS")


def probe3_anchor_data_pipeline():
    print("=== probe3: 自锚数据 120×160 管线对齐 ===")
    paths = sorted(glob.glob(os.path.join(ANCHOR_DIR, "bc_*.npz")))
    assert paths, "无自锚数据: %s" % ANCHOR_DIR
    d0 = np.load(paths[0])
    src = np.asarray(d0["image"])  # N,120,160,3 (采集即 120×160)
    assert src.shape[1:3] == (OUT_H, OUT_W), src.shape
    n0 = min(len(src), 8)
    with tempfile.TemporaryDirectory() as td:
        np.savez(os.path.join(td, "bc_00000.npz"),
                 image=src[:n0], steer=np.asarray(d0["steer"][:n0]),
                 throttle=np.asarray(d0["throttle"][:n0]),
                 meta=np.asarray(d0["meta"][:n0]))
        obs, acts = load_bc_dataset(td, out_hw=(OUT_H, OUT_W))
    assert obs.shape == (n0, 3, OUT_H, OUT_W), obs.shape
    assert obs.dtype == np.uint8
    assert acts.shape == (n0, 2), acts.shape
    frame0 = obs[0].transpose(1, 2, 0)  # H,W,3
    src0 = src[0]
    ch_obs = frame0.reshape(-1, 3).mean(axis=0)
    ch_src = src0.reshape(-1, 3).mean(axis=0)
    print("  obs=%s  ch_mean_obs=%s  ch_mean_src=%s"
          % (obs.shape, np.round(ch_obs, 1), np.round(ch_src, 1)))
    assert np.allclose(ch_obs, ch_src, atol=6.0), "通道均值偏离过大,布局错位"
    # 同分辨率时 interpolate 应近似恒等(浮点/antialias 允许小差)
    mad = float(np.abs(frame0.astype(np.int16) - src0.astype(np.int16)).mean())
    print("  same-res roundtrip mean|d|=%.2f (期望<3)" % mad)
    assert mad < 3.0, "120×160→120×160 重采样差异过大:%.2f" % mad
    diff01 = float(np.abs(obs[0].astype(np.int16) - obs[1].astype(np.int16)).mean())
    print("  frame0-frame1 mean|d|=%.2f" % diff01)
    assert diff01 > 1.0, "连续帧几乎相同,数据可能有问题"
    assert float(acts[:, 0].min()) >= -1.01 and float(acts[:, 0].max()) <= 1.01
    assert float(acts[:, 1].min()) >= -0.01 and float(acts[:, 1].max()) <= 1.01
    print("  n_source=%d" % len(src))
    print("probe3 PASS")


def main():
    th.manual_seed(0)
    np.random.seed(0)
    probe1_lambda0_resume_bitwise()
    probe2_grad_nonzero()
    probe3_anchor_data_pipeline()
    print("ALL V3-ANCHOR PROBES PASS")


if __name__ == "__main__":
    main()
