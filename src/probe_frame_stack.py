#!/usr/bin/env python3
"""叠帧布局修复的单元探针(协议门 1:探针不过不许起训)。

断言:
1. 训练链(SubprocVecEnv→VecMonitor→VecTransposeImage→VecFrameStack first)
   观测空间 == Box(0,255,(12,120,160),uint8);
2. 新建 PPO CnnPolicy 的 conv1 输入通道 == 12(不是 3——旧 bug 是 (3,480,160));
3. 实跑数步:4 个通道块两两不同(帧在滚动);最新块(ch 9-11)与底层 env 当前帧
   一致(FrameStackObservation obs[0]=最旧,与 VecFrameStack 旧帧在前对齐);
4. 评测链(gym.make→FrameStackObservation→MergeStack)与训练链逐步一致。

用法:需 9091 窗口 sim 在跑。python3 src/probe_frame_stack.py
"""
import numpy as np
import gymnasium as gym
import gym_donkeycar  # noqa

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from gymnasium.wrappers import FrameStackObservation
import importlib.util
spec = importlib.util.spec_from_file_location("rl_eval_mod", "src/rl_eval.py")
rl_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rl_eval)

CONF = {
    "exe_path": "remote", "host": "127.0.0.1", "port": 9091, "headless": False,
    "level": "mountain_track", "max_cte": 2.5, "sync_mode": True, "sync_time_step": 0.05,
    "state_obs": False, "spawn_cfg": {"node_min": 5, "node_max": 245, "lateral_offset": 1.0, "heading_noise": 8.0}, "start_delay": 3.0,
    "sim_args": ["--rl"], "cam_config": {"img_enc": "JPG"},
    "cam_resolution": (120, 160, 3),
}

def make():
    return gym.make("donkey-mountain-track-v0", conf=dict(CONF))

# ---- 1) 训练链空间 ----
v_train = DummyVecEnv([make])
v_train = VecMonitor(v_train)
v_train = VecTransposeImage(v_train)
v_train = VecFrameStack(v_train, n_stack=4, channels_order="first")
sp = v_train.observation_space
print("[probe] 训练链空间:", sp.shape, sp.dtype)
assert sp.shape == (12, 120, 160) and sp.dtype == np.uint8, "训练链空间错: %s %s" % (sp.shape, sp.dtype)

# ---- 2) CnnPolicy conv1 通道 ----
model = PPO("CnnPolicy", v_train, n_steps=64, batch_size=32, verbose=0)
conv1 = model.policy.features_extractor.cnn[0]
print("[probe] conv1:", conv1)
assert conv1.in_channels == 12, "conv1 输入通道=%d,应为 12" % conv1.in_channels

# ---- 3) 实跑:帧滚动 + 帧序 ----
obs = v_train.reset()
assert obs.shape[1:] == (12, 120, 160), "reset obs 形状错: %s" % (obs.shape,)
# reset 语义:SB3 stackedobs reset = 旧槽补零 + 新槽=当前帧(zeros padding 正确)
blk_reset = [obs[0, i*3:(i+1)*3] for i in range(4)]
print("[probe] reset 块均值:", [round(float(b.mean()),1) for b in blk_reset])
assert float(blk_reset[3].mean()) > 1.0, "reset 新槽无图像"

for _ in range(6):
    obs, _, _, _ = v_train.step([np.array([0.0, 0.3], dtype=np.float32)])
# 步进后:4 块应为真实帧且两两不同(帧滚动)
blocks = [obs[0, i*3:(i+1)*3] for i in range(4)]
diffs = [int(np.abs(blocks[i].astype(int) - blocks[j].astype(int)).sum())
         for i in range(4) for j in range(i+1, 4)]
print("[probe] 步进后 4 块两两差:", diffs)
assert all(d > 1000 for d in diffs), "步进后帧块未滚动!"
# 底层当前帧在 handler(DonkeySim) 的 image_array,channel-last
def find_handler_img(env):
    u = env
    while not hasattr(u, "image_array"):
        nxt = getattr(u, "env", None)
        if nxt is None and hasattr(u, "viewer"):
            nxt = u.viewer
        if nxt is None and hasattr(u, "handler"):
            nxt = u.handler
        if nxt is None:
            raise RuntimeError("找不到 image_array")
        u = nxt
    return u

u = find_handler_img(v_train.envs[0])
cur = np.asarray(u.image_array).transpose(2, 0, 1)  # HWC -> CHW
newest = obs[0, 9:12]
same = int(np.abs(newest.astype(int) - cur.astype(int)).sum())
print("[probe] 最新块(ch9-11) vs 底层当前帧 差:", same)
assert same == 0, "新帧不在通道 9-11,帧序错位!"

# ---- 4) 评测链一致性(空间/形状/帧序) ----
e = make()
e = rl_eval.MergeStack(FrameStackObservation(e, stack_size=4))
sp2 = e.observation_space
print("[probe] 评测链空间:", sp2.shape, sp2.dtype)
assert sp2.shape == (12, 120, 160) and sp2.dtype == np.uint8
o2, _ = e.reset(seed=1)
assert o2.shape == (12, 120, 160), "评测链 reset 形状错"
for _ in range(4):
    o2, _, _, _, _ = e.step(np.array([0.0, 0.3], dtype=np.float32))
blocks2 = [o2[i*3:(i+1)*3] for i in range(4)]
diffs2 = [int(np.abs(blocks2[i].astype(int) - blocks2[j].astype(int)).sum())
          for i in range(4) for j in range(i+1, 4)]
print("[probe] 评测链步进后 4 块两两差:", diffs2)
assert all(d > 1000 for d in diffs2), "评测链帧块未滚动!"
u2 = find_handler_img(e.env)
cur2 = np.asarray(u2.image_array).transpose(2, 0, 1)
same2 = int(np.abs(o2[9:12].astype(int) - cur2.astype(int)).sum())
print("[probe] 评测链新帧对齐差:", same2)
assert same2 == 0, "评测链帧序错位!"

v_train.close(); e.close()
print("==== FRAME-STACK PROBE: ALL PASS ====")
