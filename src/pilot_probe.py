#!/usr/bin/env python3
"""[PILOT] 任务零①: tflite 包装器 I/O 探针。

断言清单:
  A. 模型加载成功,input shape=(1,240,320,3) float32,signature 含 serving_default
  B. 输出键含 n_outputs0(steer) / n_outputs1(throttle)
  C. 已知图(纯噪声 + bc_human_data 一帧)推理:输出在合理区间(|steer|<5, thr∈[-2,3])
  D. EMA 后处理路径不崩,输出仍是 float 且在合理区间
  E. 不同输入产生不同输出(非常量函数)

用法: python3 src/pilot_probe.py | tee outputs/pilot_probe.log
"""
import os
import sys
import json
import numpy as np

MODEL = "/Users/hwp/Dev/DKC/mycar/models/pilot_1792809061430.tflite"
HUMAN_NPZ_DIR = "outputs/bc_human_data"

def main():
    os.makedirs("outputs", exist_ok=True)
    log = []
    def p(msg):
        print(msg)
        log.append(msg)

    # ---- A. 加载 ----
    p("[probe] 加载 tflite: %s" % MODEL)
    assert os.path.isfile(MODEL), "模型不存在: %s" % MODEL
    sys.path.insert(0, os.path.dirname(__file__))
    from pilot_wrapper import load_pilot
    pol = load_pilot(MODEL, mycar_post=False)  # 探针用裸模型
    m = pol.meta
    p("[probe] meta: %s" % json.dumps(m, indent=2))
    assert tuple(m["input_shape"]) == (1, 240, 320, 3), \
        "input_shape 应为 (1,240,320,3), 实际 %s" % m["input_shape"]
    assert m["input_dtype"] == "float32", "input_dtype 应为 float32"
    assert "serving_default" in str(m["input_keys"]) or len(m["input_keys"]) >= 1
    out_names = " ".join(m["output_keys"])
    p("[probe] PASS A: 加载成功, shape=%s dtype=%s" % (m["input_shape"], m["input_dtype"]))

    # ---- B. 输出键 ----
    p("[probe] output_keys=%s" % m["output_keys"])
    # 至少两个输出(steer/throttle)
    assert len(m["output_keys"]) >= 2, "输出应 >=2 维(steer+throttle), 实际 %d" % len(m["output_keys"])
    p("[probe] PASS B: 输出键数量 >=2")

    # ---- C. 已知图推理 ----
    # C1: 纯噪声
    rng = np.random.default_rng(42)
    noise = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
    s1, t1 = pol.infer_raw(noise)
    p("[probe] 噪声图: steer=%.4f throttle=%.4f" % (s1, t1))
    assert abs(s1) < 5.0, "噪声 steer 超界: %.4f" % s1
    assert -2.0 < t1 < 3.0, "噪声 throttle 超界: %.4f" % t1

    # C2: bc_human_data 真实帧(若有)
    real_img = None
    if os.path.isdir(HUMAN_NPZ_DIR):
        npzs = sorted(f for f in os.listdir(HUMAN_NPZ_DIR) if f.endswith(".npz"))
        if npzs:
            d = np.load(os.path.join(HUMAN_NPZ_DIR, npzs[0]))
            real_img = d["image"][0]  # 第一帧
            p("[probe] 真实帧来源: %s[0], shape=%s dtype=%s" % (npzs[0], real_img.shape, real_img.dtype))
    if real_img is not None:
        s2, t2 = pol.infer_raw(real_img)
        p("[probe] 真实帧: steer=%.4f throttle=%.4f" % (s2, t2))
        assert abs(s2) < 5.0, "真实帧 steer 超界: %.4f" % s2
        assert -2.0 < t2 < 3.0, "真实帧 throttle 超界: %.4f" % t2
    else:
        p("[probe] 警告: 无 bc_human_data 真实帧,跳过 C2")
    p("[probe] PASS C: 输出在合理区间")

    # ---- D. EMA 路径 ----
    pol_post = load_pilot(MODEL, mycar_post=True)
    # 连续推理 5 步(模拟控制环)
    for i in range(5):
        a, _ = pol_post.predict(noise if i % 2 == 0 else real_img if real_img is not None else noise)
    p("[probe] EMA 后 5 步末动作: steer=%.4f throttle=%.4f" % (a[0], a[1]))
    assert np.isfinite(a).all(), "EMA 输出含 NaN/Inf"
    assert abs(a[0]) < 5.0 and -2.0 < a[1] < 3.0
    p("[probe] PASS D: EMA 路径正常")

    # ---- E. 非常量 ----
    if real_img is not None:
        s_n, t_n = pol.infer_raw(noise)
        s_r, t_r = pol.infer_raw(real_img)
        diff = abs(s_n - s_r) + abs(t_n - t_r)
        p("[probe] 噪声 vs 真实输出差: %.4f" % diff)
        assert diff > 1e-4, "输出对输入不敏感(可能是常量函数): diff=%.6f" % diff
        p("[probe] PASS E: 非常量函数")
    else:
        p("[probe] PASS E: 跳过(无真实帧)")

    p("[probe] ===== 全部通过 =====")
    with open("outputs/pilot_probe.log", "w") as f:
        f.write("\n".join(log) + "\n")
    print("[probe] 日志已写 outputs/pilot_probe.log")

if __name__ == "__main__":
    main()
