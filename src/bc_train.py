#!/usr/bin/env python3
"""BC 预训练:champion2 (图像→动作) 行为克隆,权重灌入 SB3 PPO CnnPolicy。

架构逐层镜像 rl_train.py 视觉 PPO 的 CnnPolicy(SB3 NatureCNN:
conv(3→32,8,s4)→relu→conv(32→64,4,s2)→relu→conv(64→64,3,s1)→relu→
flatten→linear(→512)→relu → action_net Linear(512→2))——任务三热加载的前提。

- 数据切分按局段 90/10(meta ep_id,防时序泄漏);图像 /255 与 SB3 预处理同口径。
- |steer|>0.3 纠偏帧过采样至 ≥30% batch(WeightedRandomSampler,开关 --hard-oversample)。
- 探针门:① overfit-256 loss→~0;② val steer MAE 实测报数;③ 曲线无发散。
- 导出:rl_bc.zip = PPO CnnPolicy 骨架(features_extractor+action_net 权重,
  log_std 保持 SB3 默认),rl_eval 零改动可评测。

用法:python3 src/bc_train.py [--epochs 40] [--batch 256]
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn


class NatureCNN(nn.Module):
    """逐层镜像 SB3 NatureCNN(features_dim=512);扁平尺寸运行时探测(不写死)。"""

    def __init__(self, in_hw=(120, 160)):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = int(np.prod(self.cnn(torch.zeros(1, 3, *in_hw)).shape[1:]))
        self.linear = nn.Sequential(nn.Linear(n_flat, 512), nn.ReLU())

    def forward(self, x):  # x: (B,3,120,160) float 0..1
        return self.linear(self.cnn(x))


class BCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = NatureCNN()
        self.head = nn.Linear(512, 2)

    def forward(self, x):
        return self.head(self.trunk(x))


N_STEER_BIN, N_THR_BIN = 21, 6


class BCCatNet(nn.Module):
    """分类头 BC:共享 NatureCNN trunk,steer 21 箱/throttle 6 箱交叉熵。
    推理解码=softmax 期望(Σ p_i·bin中心)——消 MSE 多峰平均效应(蠕行根因)。"""

    def __init__(self):
        super().__init__()
        self.trunk = NatureCNN()
        self.head_s = nn.Linear(512, N_STEER_BIN)
        self.head_t = nn.Linear(512, N_THR_BIN)
        self.register_buffer("centers_s", torch.linspace(-1.0, 1.0, N_STEER_BIN))
        self.register_buffer("centers_t", torch.linspace(0.0, 1.0, N_THR_BIN))

    def forward(self, x):
        z = self.trunk(x)
        return self.head_s(z), self.head_t(z)

    @staticmethod
    def to_bins(acts):
        s = torch.round((acts[:, 0].clamp(-1, 1) + 1.0) / 2.0 * (N_STEER_BIN - 1)).long()
        t = torch.round(acts[:, 1].clamp(0, 1) * (N_THR_BIN - 1)).long()
        return s, t

    @torch.no_grad()
    def decode(self, x):
        hs, ht = self.forward(x)
        ps = torch.softmax(hs, dim=-1)
        pt = torch.softmax(ht, dim=-1)
        return torch.stack([(ps * self.centers_s).sum(-1),
                            (pt * self.centers_t).sum(-1)], dim=-1)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/bc_data")
    ap.add_argument("--out", default="outputs/rl_bc.zip")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--hard-oversample", type=float, default=0.3,
                    help="纠偏帧(|steer|>0.3)最低 batch 占比(0=关)")
    ap.add_argument("--overfit-probe", action="store_true", help="256 样本过拟合探针")
    ap.add_argument("--head", choices=["reg", "cat"], default="cat",
                    help="cat=分箱分类头(21 steer/6 throttle,softmax 期望解码,默认)")
    ap.add_argument("--seed", type=int, default=7)
    return ap.parse_args()


def load_dataset(data_dirs):
    imgs, acts, epids = [], [], []
    ep_offset = 0
    for data_dir in data_dirs.split(","):
        for f in sorted(glob.glob(os.path.join(data_dir.strip(), "bc_*.npz"))):
            d = np.load(f)
            imgs.append(d["image"])
            acts.append(np.stack([d["steer"], d["throttle"]], axis=1))
            epids.append(d["meta"][:, 0].astype(int) + ep_offset)
            ep_offset += int(d["meta"][:, 0].max()) + 1
    return (np.concatenate(imgs), np.concatenate(acts), np.concatenate(epids))


def split_by_episode(ep_ids, val_ratio=0.1, seed=7):
    """按局段切分:整局进 train 或 val,帧级混合会时序泄漏。"""
    rng = np.random.default_rng(seed)
    eps = np.unique(ep_ids)
    rng.shuffle(eps)
    n_val = max(1, int(len(eps) * val_ratio))
    val_eps = set(eps[:n_val].tolist())
    val_mask = np.array([e in val_eps for e in ep_ids])
    return ~val_mask, val_mask


def evaluate(model, imgs, acts, device, batch=512, head_cat=False):
    model.eval()
    tot_se, n = 0.0, 0
    abs_s_sum, abs_t_sum = 0.0, 0.0
    hard_abs_s, hard_n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(imgs), batch):
            x = torch.as_tensor(imgs[i:i + batch], dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.0
            y = torch.as_tensor(acts[i:i + batch], dtype=torch.float32, device=device)
            p = model.decode(x) if head_cat else model(x)
            tot_se += float(((p - y) ** 2).sum()); n += len(x)
            abs_s_sum += float((p[:, 0] - y[:, 0]).abs().sum())
            abs_t_sum += float((p[:, 1] - y[:, 1]).abs().sum())
            hard = y[:, 0].abs() > 0.3
            hard_abs_s += float((p[hard, 0] - y[hard, 0]).abs().sum())
            hard_n += int(hard.sum())
    model.train()
    return {"mse": tot_se / n, "steer_mae": abs_s_sum / n, "thr_mae": abs_t_sum / n,
            "hard_steer_mae": hard_abs_s / max(hard_n, 1), "hard_n": hard_n}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("[bc] device=%s" % device)
    imgs, acts, ep_ids = load_dataset(args.data)
    print("[bc] 数据 %d 对,局 %d 个" % (len(imgs), len(np.unique(ep_ids))))
    tr, va = split_by_episode(ep_ids, seed=args.seed)
    print("[bc] train %d / val %d(按局段)" % (tr.sum(), va.sum()))

    model = (BCCatNet() if args.head == "cat" else BCNet()).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    x_all = imgs[tr]
    y_all = acts[tr]
    if args.overfit_probe:
        x_all, y_all = x_all[:256], y_all[:256]
        print("[probe-overfit] 256 样本过拟合测试")

    # 纠偏帧过采样:权重采样使 |steer|>0.3 占比 ≥ 目标
    weights = np.ones(len(x_all), dtype=np.float64)
    hard = np.abs(y_all[:, 0]) > 0.3
    if args.hard_oversample > 0 and hard.sum() > 0:
        cur = hard.mean()
        if cur < args.hard_oversample:
            weights[hard] = (args.hard_oversample / max(cur, 1e-6))
            print("[bc] 纠偏帧 %.0f%% → 目标 %.0f%%, 权重 %.1fx" %
                  (cur * 100, args.hard_oversample * 100, weights[hard][0]))
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.as_tensor(weights), num_samples=len(x_all), replacement=True)
    ds = torch.utils.data.TensorDataset(
        torch.as_tensor(x_all), torch.as_tensor(y_all, dtype=torch.float32))
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, sampler=sampler)

    for ep in range(args.epochs):
        ep_loss, nb = 0.0, 0
        for x, y in dl:
            x = x.to(device).float().permute(0, 3, 1, 2) / 255.0
            y = y.to(device).float()
            opt.zero_grad()
            if args.head == "cat":
                hs, ht = model(x)
                bs, bt = BCCatNet.to_bins(y)
                loss = nn.functional.cross_entropy(hs, bs) + nn.functional.cross_entropy(ht, bt)
            else:
                loss = nn.functional.mse_loss(model(x), y)
            loss.backward()
            opt.step()
            ep_loss += float(loss); nb += 1
        if ep % 5 == 4 or ep == args.epochs - 1 or args.overfit_probe:
            v = evaluate(model, imgs[va], acts[va], device, head_cat=(args.head == "cat"))
            print("[bc] ep%d loss=%.4f | val MSE=%.4f steerMAE=%.4f thrMAE=%.4f "
                  "hardSteerMAE=%.4f(n=%d)" % (
                      ep, ep_loss / nb, v["mse"], v["steer_mae"], v["thr_mae"],
                      v["hard_steer_mae"], v["hard_n"]), flush=True)
        if args.overfit_probe and ep_loss / nb < 1e-3:
            print("[probe-overfit] loss→%.5f,容量/管线通 PASS" % (ep_loss / nb))
            return

    # ---- 探针门 ----
    assert not args.overfit_probe
    v = evaluate(model, imgs[va], acts[va], device, head_cat=(args.head == "cat"))
    print("[bc] 终验: val steerMAE=%.4f hardSteerMAE=%.4f thrMAE=%.4f(首跑标定,无先验门)" % (
        v["steer_mae"], v["hard_steer_mae"], v["thr_mae"]))

    # ---- 导出到 SB3 PPO CnnPolicy 骨架 ----
    import gymnasium as gym
    from stable_baselines3 import PPO

    class Dummy(gym.Env):
        observation_space = gym.spaces.Box(0, 255, (3, 120, 160), dtype=np.uint8)
        action_space = gym.spaces.Box(
            np.array([-1., 0.], dtype=np.float32), np.array([1., 1.], dtype=np.float32))

        def reset(self, **k):
            return np.zeros((3, 120, 160), np.uint8), {}

        def step(self, a):
            return np.zeros((3, 120, 160), np.uint8), 0., False, False, {}

    ppo = PPO("CnnPolicy", Dummy(), verbose=0)
    # trunk→features_extractor(cnn+linear)
    src = model.trunk.state_dict()
    dst = ppo.policy.features_extractor
    assert src["cnn.0.weight"].shape == dst.state_dict()["cnn.0.weight"].shape
    dst.cnn.load_state_dict({k[len("cnn."):]: v for k, v in src.items() if k.startswith("cnn.")})
    dst.linear.load_state_dict(model.trunk.linear.state_dict())
    if args.head == "cat":
        # 分类头:解码头模块寄居 fork 命名空间(gym_donkeycar),rl_eval import 链
        # 天然可反序列化;mean=action_net(latent) 即 softmax 期望,零改动可评测。
        from gym_donkeycar.envs.bc_decode import ExpectDecode
        dec = ExpectDecode(512, N_STEER_BIN, N_THR_BIN)
        dec.head_s.load_state_dict(model.head_s.state_dict())
        dec.head_t.load_state_dict(model.head_t.state_dict())
        ppo.policy.action_net = dec
    else:
        with torch.no_grad():
            ppo.policy.action_net.weight.copy_(model.head.weight)
            ppo.policy.action_net.bias.copy_(model.head.bias)
    ppo.num_timesteps = 0
    ppo.save(args.out)
    print("[bc] 导出 %s(rl_eval 零改动可评测)" % args.out)


if __name__ == "__main__":
    main()
