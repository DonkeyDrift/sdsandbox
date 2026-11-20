#!/usr/bin/env python3
"""crash_penalty 奖励修复探针(起训前必过)。

断言:
1. crash_penalty=50:cte 越界终止步 reward == -50 且 terminated=True;
   非终止步 reward == 旧进度公式(未被污染);
2. crash_penalty 缺省(0):同一动作序列下终止步 reward == -1.0(旧行为),
   全轨迹 reward 与旧公式逐位一致(状态量轨道回归保护);
3. TimeLimit 截断:truncated=True 的步 reward 为正常进度奖励(无 -50)。

需 9091 窗口 sim。python3 src/probe_crash_penalty.py
"""
import math
import sys

import gymnasium as gym
import gym_donkeycar  # noqa
import numpy as np

BASE = {
    "exe_path": "remote", "host": "127.0.0.1", "port": 9091, "headless": False,
    "level": "mountain_track", "max_cte": 2.5, "sync_mode": True, "sync_time_step": 0.05,
    "state_obs": True,
    "spawn_cfg": {"node_min": 60, "node_max": 60, "lateral_offset": 2.4, "heading_noise": 25.0},
    "start_delay": 3.0, "sim_args": ["--rl"], "cam_config": {"img_enc": "none"},
}

def expected_reward(fwd, cte, max_cte):
    if fwd > 0.0:
        return (1.0 - (math.fabs(cte) / max_cte)) * fwd
    return fwd

def run_episode(penalty, max_steps=60):
    conf = dict(BASE)
    if penalty > 0:
        conf["crash_penalty"] = penalty
    env = gym.make("donkey-mountain-track-v0", conf=conf)
    env.reset(seed=42)
    rows = []
    for i in range(max_steps):
        a = np.array([0.0, 0.8], dtype=np.float32)  # 直行全油门,从边界出生必越界
        obs, r, term, trunc, info = env.step(a)
        rows.append((float(r), bool(term), bool(trunc), float(info.get("cte", 0)),
                     float(info.get("forward_vel", 0))))
        if term or trunc:
            break
    env.close()
    return rows

def check(rows, penalty, label):
    term_idx = [i for i, r in enumerate(rows) if r[1]]
    trunc_idx = [i for i, r in enumerate(rows) if r[2]]
    print("[%s] 步数=%d 终止步=%s 截断步=%s" % (label, len(rows), term_idx, trunc_idx))
    assert term_idx, "%s: 未触发终止,探针无效" % label
    t = term_idx[0]
    r_term = rows[t][0]
    if penalty > 0:
        assert r_term == -penalty, "%s: 终止 reward=%s,应为 -%s" % (label, r_term, penalty)
    else:
        assert r_term == -1.0, "%s: 缺省终止 reward=%s,旧行为应为 -1.0" % (label, r_term)
    # 非终止步与旧公式逐位一致(回归保护)
    for i, (r, term, trunc, cte, fwd) in enumerate(rows):
        if i == t:
            continue
        exp = expected_reward(fwd, cte, 2.5)
        assert abs(r - exp) < 1e-6, "%s: 步%d reward=%.6f 公式=%.6f" % (label, i, r, exp)
    # 截断步(若有)不受惩罚
    for i in trunc_idx:
        assert rows[i][0] > -penalty if penalty > 0 else rows[i][0] != -1.0, \
            "%s: 截断步被惩罚" % label
    print("[%s] PASS(终止 reward=%.1f,其余 %d 步与旧公式一致)" % (label, r_term, len(rows) - 1))
    return t

# ---- TimeLimit 截断不扣 ----
def check_truncation_no_penalty():
    conf = dict(BASE)
    conf["crash_penalty"] = 50
    env = gym.make("donkey-mountain-track-v0", conf=conf)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=3)
    env.reset(seed=7)
    last = None
    for i in range(3):
        last = env.step(np.array([0.0, 0.5], dtype=np.float32))
    r, term, trunc = last[1], last[2], last[3]
    print("[trunc] 第3步 reward=%.3f term=%s trunc=%s" % (r, term, trunc))
    assert trunc and not term, "TimeLimit 未截断?"
    assert r != -50.0, "截断步被扣 -50!"
    print("[trunc] PASS(截断无惩罚)")

rows_on = run_episode(50)
check(rows_on, 50, "penalty=50")
rows_off = run_episode(0)
check(rows_off, 0, "penalty=0(回归)")
# 同序列同种子:开/关两轨的非终止奖励应逐位一致(除终止步外)
n = min(len(rows_on), len(rows_off))
for i in range(n):
    if rows_on[i][1] or rows_off[i][1]:
        break
    assert rows_on[i][0] == rows_off[i][0], "步%d 两轨 reward 分叉: %s vs %s" % (i, rows_on[i][0], rows_off[i][0])
print("[cross] 开/关两轨非终止 reward 逐位一致 PASS")
check_truncation_no_penalty()
print("==== CRASH-PENALTY PROBE: ALL PASS ====")
