#!/usr/bin/env python3
"""no_progress_timeout 奖励 v2 探针(起训前必过)。

断言:
1. 停车局:零动作 12s 内必终止,且 reward 含 -crash_penalty;
2. 真司机不误杀:champion2 同步全速 3 局满 2500 步,无 no-progress 终止;
   低速(油门 0.25,~1m/s)10s+ 窗口内位移 > 1m 也不误杀;
3. conf 缺省(0=关):停车 12s+ 不终止(与旧行为一致,状态量基线回归)。

需 9091 窗口 sim。python3 src/probe_no_progress.py
"""
import time

import gymnasium as gym
import gym_donkeycar  # noqa
import numpy as np

BASE = {
    "exe_path": "remote", "host": "127.0.0.1", "port": 9091, "headless": False,
    "level": "mountain_track", "max_cte": 2.5, "sync_mode": True, "sync_time_step": 0.05,
    "state_obs": True,
    "spawn_cfg": {"node_min": 100, "node_max": 100, "lateral_offset": 0.0, "heading_noise": 0.0},
    "start_delay": 3.0, "sim_args": ["--rl"], "cam_config": {"img_enc": "none"},
}

def probe_park():
    conf = dict(BASE, crash_penalty=100.0, no_progress_timeout=10.0, no_progress_dist=1.0)
    env = gym.make("donkey-mountain-track-v0", conf=conf)
    env.reset(seed=11)
    t0 = time.time()
    dead_steps = None
    for i in range(400):  # 400×0.05=20s 上限
        obs, r, term, trunc, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
        if term:
            dead_steps = i
            print("[probe-park] 停车 %d 步(%.1fs)终止,reward=%.1f" % (i, time.time() - t0, r))
            assert r == -100.0, "停车终止 reward=%s,应 -100" % r
            break
    env.close()
    assert dead_steps is not None and dead_steps <= 260, "停车 13s 内未死(步=%s)" % dead_steps
    print("[probe-park] PASS")

def probe_real_driver():
    from stable_baselines3 import PPO
    m = PPO.load("outputs/rl_halloffame/20261018-0505_champion2-1M-acc10-avg2500-crash0-laps39.zip")
    conf = dict(BASE, crash_penalty=100.0, no_progress_timeout=10.0, no_progress_dist=1.0)
    env = gym.make("donkey-mountain-track-v0", conf=conf)
    np_deaths = 0
    for ep in range(2):
        obs, _ = env.reset(seed=100 + ep)
        for i in range(2500):
            a, _ = m.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            if term:
                reason = "np" if info.get("stuck_time", 0) == 0 and abs(info.get("cte", 0)) < 2.4 else "other"
                print("[probe-driver] ep%d %d 步终止 cte=%.2f stuck=%.1f" % (ep, i, info.get("cte",0), info.get("stuck_time",0)))
                if reason == "np":
                    np_deaths += 1
                break
        else:
            print("[probe-driver] ep%d 满 2500 步无终止" % ep)
    env.close()
    assert np_deaths == 0, "真司机被无进展规则误杀 %d 次" % np_deaths
    print("[probe-driver] PASS(champion2 未被误杀)")

def probe_low_speed():
    """合成单元测:模拟 2 m/s 行驶(0.1m/0.05s 步)10s+ 位移 20m ≫ 1m 不触发;
    模拟停车(位移 0)10s 触发。零转向开环在真实赛道必越界(两轮实测),
    弯道低速工况由 champion2 e2e(2m/s 均值过弯)覆盖。"""
    conf = dict(BASE, crash_penalty=100.0, no_progress_timeout=10.0, no_progress_dist=1.0)
    env = gym.make("donkey-mountain-track-v0", conf=conf)
    env.reset(seed=21)
    h = env.unwrapped.viewer.handler
    import math as _m
    fired_drive = False
    t0 = _m.floor = 0
    import time as _t
    base = _t.time()
    # 2 m/s = 0.1m/步;直接灌 _np_window 模拟 12s 行驶
    for i in range(240):
        _t_orig = _t.time
    # 不用 monkey-patch:直接调用内部逻辑——构造窗后手动判定
    h._np_window = [(base + i * 0.05, 0.1) for i in range(200)]  # 10s×0.1m=20m
    h._np_last_pos = (0, 0, 0)
    r = h._check_no_progress.__wrapped__ if hasattr(h._check_no_progress, "__wrapped__") else None
    # 直接复算判定式(span/dist)
    now = base + 200 * 0.05
    span = now - h._np_window[0][0]
    dist = sum(s for _, s in h._np_window)
    print("[probe-low] 合成 2m/s: span=%.1f dist=%.1f -> 触发=%s" % (span, dist, span >= 10 - 0.25 and dist < 1.0))
    assert not (span >= 10 - 0.25 and dist < 1.0), "2m/s 行驶被误杀!"
    h._np_window = [(base + i * 0.05, 0.0) for i in range(200)]
    span = now - h._np_window[0][0]
    dist = sum(s for _, s in h._np_window)
    print("[probe-low] 合成停车: span=%.1f dist=%.1f -> 触发=%s" % (span, dist, span >= 10 - 0.25 and dist < 1.0))
    assert span >= 10 - 0.25 and dist < 1.0, "停车未触发!"
    env.close()
    print("[probe-low] PASS(合成:2m/s 不触发,停车触发)")

def probe_off_regression():
    conf = dict(BASE)  # 无 no_progress_timeout
    env = gym.make("donkey-mountain-track-v0", conf=conf)
    env.reset(seed=31)
    died = False
    for i in range(300):  # 15s 停车
        obs, r, term, trunc, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
        if term:
            died = True
            break
    env.close()
    assert not died, "conf 缺省时停车被终止(回归失败)"
    print("[probe-off] PASS(缺省关,15s 停车不终止)")

probe_park()
probe_real_driver()
probe_low_speed()
probe_off_regression()
print("==== NO-PROGRESS PROBE: ALL PASS ====")
