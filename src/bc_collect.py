#!/usr/bin/env python3
"""BC 数据采集:教师开车,(图像→动作) 标签零成本。

两种教师:
  状态量(默认):champion2 吃 info 重建的 8 维状态向量(需 --teacher 指定)。
  --vision:CNN 教师(如 vision3)直接吃 env 图像 obs(HWC→CHW),用于
    [V3-ANCHOR] 自锚数据——多域多 seed 闭环 + 剔除每局终止前 10 步失控段。

出生课程(救车动作覆盖率,状态量模式):随机 respawn node 15-105 + lateral ±2.2 +
heading ±15,撞毁或每 60s 仿真重撒。图像必须真实:headless_keep_gfx
离屏渲染(--launch 式死键教训,这里显式置位)。

用法:
  python3 src/bc_collect.py --probe          # 500 对探针门
  python3 src/bc_collect.py --pairs 40000    # 全量
  python3 src/bc_collect.py --vision --pairs 15000 --teacher outputs/rl_vision3.zip \
      --out outputs/bc_v3anchor --drop-tail 10 --domains 8200,8300,8100
"""
import argparse
import json
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
        "sync_mode": not args.async_mode,
        "state_obs": not args.vision,
        "start_delay": 6.0, "sim_args": ["--rl"],
        "cam_config": {"img_enc": "JPG"},
        "cam_resolution": (120, 160, 3),
    }
    if not args.async_mode:
        conf["sync_time_step"] = 0.05
    if not args.vision:
        conf["spawn_cfg"] = dict(SPAWN)
    else:
        # 自锚数据:教师自己的出生分布(rl_train 默认 spawn),不开救车课程
        conf["spawn_cfg"] = {"node_min": 5, "node_max": 245,
                             "lateral_offset": 1.0, "heading_noise": 8.0}
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
    ap.add_argument("--vision", action="store_true", default=False,
                    help="CNN 教师直接吃图像 obs([V3-ANCHOR] 自锚采集)")
    ap.add_argument("--async", dest="async_mode", action="store_true", default=False,
                    help="异步遥测模式(vision3 原生训练 regime;sync 下该教师行为病态)")
    ap.add_argument("--drop-tail", type=int, default=0,
                    help="每局剔除终止前 N 步失控段(vision 模式默认 10)")
    ap.add_argument("--domains", default="8200,8300,8100",
                    help="vision 模式多域 seed 基址,逗号分隔")
    args = ap.parse_args()
    if args.vision and args.drop_tail == 0:
        args.drop_tail = 10
    domains = [int(x) for x in args.domains.split(",") if x]

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
    episodes = []  # 数据卡

    while collected < target:
        seed = (domains[ep_id % len(domains)] + ep_id // len(domains)) \
            if args.vision else args.seed_base + ep_id
        obs, info = env.reset(seed=seed)
        ep_id += 1
        steps_since_respawn = 0
        done = False
        ep_imgs, ep_steers, ep_throttles, ep_metas = [], [], [], []
        while not done and collected < target:
            if args.vision:
                # CNN 教师:env obs 是 HWC uint8,policy 期望 CHW
                obs_chw = np.transpose(obs, (2, 0, 1))
                t_a, _ = teacher.predict(obs_chw, deterministic=True)
            else:
                svec = state_vec_from_info(info, handler)
                t_a, _ = teacher.predict(svec, deterministic=True)
            steer_l, thr_l = float(t_a[0]), float(t_a[1])
            if args.vision or not args.actor_model:
                a_exec = t_a  # 教师即执行者;vision 模式 obs 是 HWC,不能再 predict
            else:
                a_exec, _ = actor.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(
                np.asarray(a_exec, dtype=np.float32))
            ep_imgs.append(np.asarray(obs, dtype=np.uint8).copy())
            ep_steers.append(steer_l)
            ep_throttles.append(thr_l)
            ep_metas.append((ep_id, seed,
                             float(info.get("cte", 0)), float(info.get("speed", 0))))
            collected += 1
            steps_since_respawn += 1
            done = term or trunc or (steps_since_respawn * 0.05 >= args.respawn_secs)
            if collected % 2000 == 0:
                rate = collected / (time.time() - t_start)
                print("[bc_collect] %d/%d 对 (%.0f 对/s, ep=%d seed=%d)" %
                      (collected, target, rate, ep_id, seed), flush=True)
            if args.probe and collected >= 500:
                done = True
        # 剔尾:终止前 N 步失控段不入库(自锚先验要干净驾驶段)
        cut = max(0, len(ep_imgs) - args.drop_tail) if args.drop_tail else len(ep_imgs)
        if cut > 0:
            imgs.extend(ep_imgs[:cut])
            steers.extend(ep_steers[:cut])
            throttles.extend(ep_throttles[:cut])
            metas.extend(ep_metas[:cut])
            episodes.append({"ep": ep_id, "seed": int(seed), "kept": int(cut),
                             "raw": len(ep_imgs), "term": bool(term), "trunc": bool(trunc)})
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

    # ---- 数据卡 ----
    last = args.probe_out if args.probe else \
        os.path.join(args.out, "bc_%05d.npz" % (shard_idx - 1))
    d = np.load(last)
    img, st, th = d["image"], d["steer"], d["throttle"]
    steer_hard = float((np.abs(st) > 0.3).mean())
    thr_ok = float((th > 0.2).mean())
    print("[card] 图像 std=%.1f | |steer|>0.3 占比=%.1f%% | throttle>0.2 占比=%.1f%%"
          % (img.std(), steer_hard * 100, thr_ok * 100))
    print("[card] steer: mean=%.3f std=%.3f p05=%.3f p95=%.3f | throttle: mean=%.3f std=%.3f"
          % (st.mean(), st.std(), np.percentile(st, 5), np.percentile(st, 95),
             th.mean(), th.std()))
    std_ok = float(img.std()) > 20.0
    assert std_ok, "图像疑似全灰/盲图——查 headless_keep_gfx"
    if args.vision:
        # 自锚数据:教师行为即先验,不设动作占比硬门;只报分布(三方对照用)
        os.makedirs(args.out, exist_ok=True)
        card = {
            "teacher": args.teacher, "vision": True,
            "pairs_total": int(collected),
            "domains": domains, "drop_tail": int(args.drop_tail),
            "respawn_secs": args.respawn_secs,
            "episodes": episodes[:200],
            "steer_mean": float(st.mean()), "steer_std": float(st.std()),
            "steer_p05": float(np.percentile(st, 5)),
            "steer_p95": float(np.percentile(st, 95)),
            "steer_hard_frac": steer_hard,
            "thr_mean": float(th.mean()), "thr_std": float(th.std()),
            "thr_gt02_frac": thr_ok,
            "img_std_last_shard": float(img.std()),
        }
        with open(os.path.join(args.out, "data_card.json"), "w") as f:
            json.dump(card, f, indent=2)
        print("[card] -> %s/data_card.json" % args.out)
        print("==== V3-ANCHOR COLLECT DONE ====")
        return
    assert steer_hard >= 0.15, "纠偏动作不足(%.0f%%)——加大 lateral/heading 课程" % (steer_hard * 100)
    # 油门门 relax 到 30%:champion2 的驾驶风格是"全油直道+弯前重刹",
    # throttle>0.2 天然只占 ~37%——双踏板都被充分代表才是标签完整性,
    # 强扭到 80% 会扭曲教师分布(制造协变量问题)。
    assert thr_ok >= 0.30, "油门帧过少——教师过于保守?"
    print("==== BC COLLECT PROBE PASS ====")


if __name__ == "__main__":
    main()
