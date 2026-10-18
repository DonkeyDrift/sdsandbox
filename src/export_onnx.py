#!/usr/bin/env python3
"""冠军策略导出:PPO MlpPolicy -> ONNX(+ 真车推理侧的输入归一化说明)。

观测向量(gym_donkeycar state_obs 模式,8 维 float32,组装见
GymDonkeyCar/gym_donkeycar/envs/donkey_sim.py 的 observe()):
  [0] cte / 2.0          横向偏差(米),赛道中心线坐标系,左正右负
  [1] speed / 10.0       车速(米/秒,标量速度)
  [2] forward_vel / 10.0 前向速度(米/秒)
  [3] yaw / 180.0        偏航角(度)
  [4] gyro_z / 90.0      偏航角速度(度/秒)
  [5] active_node / total_nodes  赛道进度 [0,1](当前节点/总节点数)
  [6] 0.0(保留)
  [7] 0.0(保留)
真车侧必须按同样公式归一化后再喂 ONNX。

输出 2 维:[steering, throttle]——steering ∈ [-1,1](左正),
throttle ∈ [0,1]。ONNX 内部已含 SB3 的 action space 反缩放,
直接输出物理量,无需再线性变换。

用法:python3 src/export_onnx.py --model <冠军.zip> --out outputs/rl_export/champion.onnx
"""

import argparse
import json
import os

import numpy as np
import torch
from stable_baselines3 import PPO

OBS_SPEC = [
    ("cte/2.0", "横向偏差(米),赛道中心线坐标系,左正右负"),
    ("speed/10.0", "车速(米/秒,标量)"),
    ("forward_vel/10.0", "前向速度(米/秒)"),
    ("yaw/180.0", "偏航角(度)"),
    ("gyro_z/90.0", "偏航角速度(度/秒)"),
    ("active_node/max(total_nodes,1)", "赛道进度 [0,1]"),
    ("0.0", "保留"),
    ("0.0", "保留"),
]


class ExportPolicy(torch.nn.Module):
    """包装 SB3 策略:obs -> (action 反缩放到物理量, value)。"""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs):
        feats = self.policy.extract_features(obs)
        latent_pi, latent_vf = self.policy.mlp_extractor(feats)
        mean = self.policy.action_net(latent_pi)
        value = self.policy.value_net(latent_vf)
        # MlpPolicy 高斯分布 mean_actions 即确定性动作;tanh 输出已由
        # action_net 保证,反缩放用 SB3 的 low/high(见 unscale_action)。
        low = torch.as_tensor(self.policy.action_space.low, dtype=torch.float32)
        high = torch.as_tensor(self.policy.action_space.high, dtype=torch.float32)
        action = low + (0.5 * (mean + 1.0) * (high - low))
        return action, value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = PPO.load(args.model, device="cpu")
    assert isinstance(model.observation_space.shape, tuple) and model.observation_space.shape == (8,), \
        "导出只支持状态量 8 维观测模型(MlpPolicy),vision CNN 请走 torch.jit"

    wrapped = ExportPolicy(model.policy).eval()
    obs = torch.zeros(1, 8)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.onnx.export(
        wrapped, obs, args.out,
        input_names=["obs"], output_names=["action", "value"],
        dynamic_axes={"obs": {0: "batch"}},
        opset_version=17,
    )

    # 数值一致性:onnx.reference 参考实现 vs SB3 确定性 predict
    from onnx.reference import ReferenceEvaluator
    rng = np.random.default_rng(0)
    cases = rng.normal(0, 0.5, (16, 8)).astype(np.float32)
    sess = ReferenceEvaluator(args.out)
    ok = True
    for c in cases:
        onnx_out = sess.run(None, {"obs": c[None, :]})[0][0]
        sb3_out, _ = model.predict(c, deterministic=True)
        if not np.allclose(onnx_out, sb3_out, atol=1e-5):
            ok = False
            print("[export] MISMATCH onnx=%s sb3=%s" % (onnx_out, sb3_out))
    assert ok, "ONNX 输出与 SB3 predict 不一致"

    spec = {
        "source_model": os.path.abspath(args.model),
        "obs_vector": OBS_SPEC,
        "action": ["steering [-1,1] 左正", "throttle [0,1]"],
        "num_check_cases": len(cases), "max_abs_diff_check": "1e-5",
    }
    spec_path = os.path.splitext(args.out)[0] + ".spec.json"
    with open(spec_path, "w") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)
    print("[export] done: %s (+ %s),16 组随机观测与 SB3 确定性输出一致" % (args.out, spec_path))


if __name__ == "__main__":
    main()
