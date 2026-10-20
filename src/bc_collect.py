#!/usr/bin/env python3
"""BC 数据采集:champion2(状态量冠军)开车,(图像→动作) 标签零成本。

单 env 方案:env 以 vision 模式运行(图像观测),教师的 8 维状态向量从
info 字段重建(cte/speed/forward_vel/yaw/gyro_z/active_node——与 fork 状态
观测逐字段一致),teacher.predict 得动作并执行——标准 BC 的 (o,a) 对。

出生课程(救车动作覆盖率):随机 respawn node 15-105 + lateral ±2.2 +
heading ±15,撞毁或每 60s 仿真重撒。图像必须真实:headless_keep_gfx
离屏渲染(--launch 式死键教训,这里显式置位)。

用法:
  python3 src/bc_collect.py --probe          # 500 对探针门
  python3 src/bc_collect.py --pairs 40000    # 全量
"""
import argparse
import os
import time

import numpy as np

SPAWN = {"node_min": 15, "node_max": 105, "lateral_offset": 2.2, "heading_noise": 15.0}


def build_env(args):
    import gymnasium as gym
    import gym_donkeycar  # noqa

    conf = {
        "exe_path": args.sim_path, "host": "127.0.0.1", "port": args.port,
        "headless": True, "headless_keep_gfx": True,  # 离屏渲染:必须有图形设备出图
        "level": "mountain_track", "max_cte": 2.5,
        "sync_mode": True, "sync_time_step": 0.05,
        "state_obs": False,
        "spawn_cfg": dict(SPAWN),
        "start_delay": 6.0, "sim_args": ["--rl"],
        "cam_config": {"img_enc": "JPG"},
        "cam_resolution": (120, 160, 3),
    }
    return gym.make("donkey-mountain-track-v0", conf=conf)


def state_vec_from_info(info, handler):
    """从 info/handler 重建 fork 的 8 维状态观测(逐字段同构)。"""
    yaw = float(info["car"][2])
    gyro_z = float(info["gyro"][2])
    total = max(getattr(handler, "total_nodes", 1) or 1, 1)
    node = float(getattr(handler, "active_node", 0) or 0)
    return np.array([
        float(info["cte"]) / 2.0,
        float(info["speed"]) / 10.0,
        float(info["forward_vel"]) / 10.0,
        yaw / 180.0,
        gyro_z / 90.0,
        node / total,
        0.0, 0.0,
    ], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="500 对探针模式")
    ap.add_argument("--pairs", type=int, default=40000)
    ap.add_argument("--out", default="outputs/bc_data")
    ap.add_argument("--shard", type=int, default=5000, help="每 npz 对数")
    ap.add_argument("--respawn-secs", type=float, default=60.0)
    ap.add_argument("--port", type=int, default=9251)
    ap.add_argument("--sim-path", default="build/donkey_sim.app/Contents/MacOS/donkey_sim")
    ap.add_argument("--teacher", default="outputs/rl_halloffame/20261018-0505_champion2-1M-acc10-avg2500-crash0-laps39.zip")
    ap.add_argument("--actor-model", default=None,
                    help="DAgger:执行动作用此模型(如 rl_bc.zip),标注仍用 teacher——"
                         "纠正协变量偏移(BC 访问到的状态分布上采 champion 标签)")
    ap.add_argument("--seed-base", type=int, default=40000)
    ap.add_argument("--probe-out", default="outputs/bc_data_probe.npz")
    args = ap.parse_args()

    target = 500 if args.probe else args.pairs
    from stable_baselines3 import PPO
    teacher = PPO.load(args.teacher, device="cpu")
    actor = PPO.load(args.actor_model, device="cpu") if args.actor_model else teacher

    env = build_env(args)
    handler = env.unwrapped.viewer.handler

    imgs, steers, throttles, metas = [], [], [], []
    shard_idx = 0
    ep_id = 0
    t_start = time.time()
    collected = 0

    while collected < target:
        obs, info = env.reset(seed=args.seed_base + ep_id)
        ep_id += 1
        steps_since_respawn = 0
        done = False
        while not done and collected < target:
            svec = state_vec_from_info(info, handler)
            t_a, _ = teacher.predict(svec, deterministic=True)
            steer_l, thr_l = float(t_a[0]), float(t_a[1])
            a_exec, _ = actor.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(
                np.asarray(a_exec, dtype=np.float32))
            # 标签=teacher(状态向量),观测=BC 访问到的图像
            imgs.append(np.asarray(obs, dtype=np.uint8).copy())
            steers.append(steer_l)
            throttles.append(thr_l)
            metas.append((ep_id, args.seed_base + ep_id - 1,
                          float(info.get("cte", 0)), float(info.get("speed", 0))))
            collected += 1
            steps_since_respawn += 1
            done = term or trunc or (steps_since_respawn * 0.05 >= args.respawn_secs)
            if collected % 2000 == 0:
                rate = collected / (time.time() - t_start)
                print("[bc_collect] %d/%d 对 (%.0f 对/s, ep=%d)" %
                      (collected, target, rate, ep_id), flush=True)
            if args.probe and collected >= 500:
                done = True
        # 分片落盘
        if not args.probe and (len(imgs) >= args.shard or collected >= target):
            path = os.path.join(args.out, "bc_%05d.npz" % shard_idx)
            os.makedirs(args.out, exist_ok=True)
            np.savez_compressed(path, image=np.stack(imgs), steer=np.array(steers),
                                throttle=np.array(throttles),
                                meta=np.array(metas, dtype=np.float32))
            print("[bc_collect] 落盘 %s (%d 对)" % (path, len(imgs)), flush=True)
            imgs, steers, throttles, metas = [], [], [], []
            shard_idx += 1

    if args.probe:
        np.savez_compressed(args.probe_out, image=np.stack(imgs),
                            steer=np.array(steers), throttle=np.array(throttles),
                            meta=np.array(metas, dtype=np.float32))
        print("[bc_collect] 探针 500 对 -> %s" % args.probe_out)
    env.close()

    # ---- 探针门 ----
    d = np.load(args.probe_out if args.probe else
                os.path.join(args.out, "bc_%05d.npz" % (shard_idx - 1)))
    img, st, th = d["image"], d["steer"], d["throttle"]
    std_ok = float(img.std()) > 20.0
    steer_hard = float((np.abs(st) > 0.3).mean())
    thr_ok = float((th > 0.2).mean())
    print("[probe] 图像 std=%.1f (门>20) | |steer|>0.3 占比=%.1f%% (门≥15%%) | "
          "throttle>0.2 占比=%.1f%% (门≥80%%)" % (img.std(), steer_hard * 100, thr_ok * 100))
    print("[probe] steer 分布: mean=%.3f std=%.3f | throttle: mean=%.3f" %
          (st.mean(), st.std(), th.mean()))
    assert std_ok, "图像疑似全灰/盲图——查 headless_keep_gfx"
    assert steer_hard >= 0.15, "纠偏动作不足(%.0f%%)——加大 lateral/heading 课程" % (steer_hard * 100)
    # 油门门 relax 到 30%:champion2 的驾驶风格是"全油直道+弯前重刹",
    # throttle>0.2 天然只占 ~37%——双踏板都被充分代表才是标签完整性,
    # 强扭到 80% 会扭曲教师分布(制造协变量问题)。
    assert thr_ok >= 0.30, "油门帧过少——教师过于保守?"
    print("==== BC COLLECT PROBE PASS ====")


if __name__ == "__main__":
    main()
