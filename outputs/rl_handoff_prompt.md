# DonkeySim 强化学习·下一阶段推进提示词（交接另一个 AI）

> 你是接手 DonkeySim（sdsandbox）小车自动驾驶 RL 项目的 AI 工程师。本文档是你的完整任务简报。
> 开始前**必须先读**：仓库根目录 `AGENTS.md`（工程结构/构建/调试/运维铁律）和
> `outputs/rl_journal.md`（全程决策日志，含所有踩坑记录）。两份报告了解现状：
> `outputs/rl_report_final.md`、`outputs/rl_report_vision.md`。

## 1. 项目现状（2026-10-18 收官状态）

- **已有冠军模型**：`outputs/rl_halloffame/20261018-0505_champion2-1M-acc10-avg2500-crash0-laps39.zip`
  （PPO 状态量观测，~1.0M 步，仅 mountain_track 场景）。
- **验收成绩**：held-out 种子 5000-5009 / 6000-6009 各 10 局全部满 2500 步、零事故
  （日志：`outputs/rl_acceptance_1m.log`、`rl_acceptance_6k.log`）。
- **全新种子域长测**：11 局 10 局满 3000 步（91% 全程率）；唯一早退（seed 7104，135 步越界）
  复测 2/2 通过 500 步——属随机出生边缘姿态 + PhysX 位级方差，非死区。
- **视觉轨道基线**：CNN 从 120×160 单目图像可稳定学习，7.5k 步模型 held-out 平均 444 步；
  因吞吐（~15 步/s）与离屏 sim 长时稳定性收束。
- **已知边界**：冠军只在**同步 0.05s 步进**下工作；异步实时（任何帧率/动作重复配置）必撞——
  策略对动作-观测相位差敏感。headless（`-nographics`）下异步观测流 ~1fps，异步训练当前不可行。

## 2. 环境事实（不要重新探索）

- Unity 工程 `sdsim/`（2022.3.49f1）；Python 客户端 `src/`（训练 `rl_train.py`、评测 `rl_eval.py`、
  冒烟 `rl_smoke_test.py`）。gym 适配层在 `/Users/hwp/Dev/DKC/GymDonkeyCar/`。
- 构建：`/Applications/Unity/Hub/Editor/2022.3.49f1/Unity.app/Contents/MacOS/Unity -batchmode \
  -projectPath sdsim -executeMethod BuildCommand.Build -buildTargetStandaloneOSX \
  /Users/hwp/Dev/DKC/sdsandbox/build/donkey_sim.app -logFile build/unity-build.log`
  （报 license 错先 `open -a "Unity Hub"` 等 20s 重试；脚本级改动增量构建 ~15s）。
- 窗口演示 sim：`build/donkey_sim.app/Contents/MacOS/donkey_sim --rl`（端口 9091）。
- 训练命令模板：`python3 src/rl_train.py --resume <模型> --run-name <名> --envs 4 \
  --steps <计数> [--spawn-lateral X --spawn-heading Y --lr Z]`（训练 worker 端口 9101+，
  评测 worker 紧随）。
- 评测命令模板：`python3 src/rl_eval.py --model <m> --seed-base <新种子> --episodes 10 \
  --max-steps 2500 --json --launch-sim --port 9301 --sync --time-step 0.05`。
- **验收一律用同步模式**（异步口径会误杀高增益策略）；PhysX 多线程有位级方差，
  验收按多局/长局口径，不用单局定论。

## 3. 运维铁律（血泪沉淀，违反=事故）

1. 训练运行期间清理 sim **只按端口**：`lsof -ti:<端口们> | xargs kill`；**严禁 `pkill -f donkey_sim`**（两次误杀训练）。
2. 补丁后必须单环境实跑验证，`py_compile` 不算验证。
3. `cam_config.fps` 等协议字段必须传**数值**，字符串会在 fork 子进程里杀死接收线程。
4. tb 假步进识别：`time/fps > 600` 或奖励冻结 = sim 已挂、在陈旧观测上空转，立即查 sim 存活。
5. tensorboard 目录会被多次 run 复用（PPO_N 混写）——分析曲线前按 event 文件时间戳归属，
   勿整目录聚合（上周期因此误判两次）。
6. `CheckpointCallback.save_freq` 是 **vec-step 粒度**（= 步数/envs）；长跑前先换算成墙钟确认
   落盘节奏。EvalCallback 的评测 worker 断连会让整个 `learn()` 抛 EOFError，未保存的产物全失。
7. git 提交纪律见根 AGENTS.md：验证成功即 commit+push 到当前分支上游，不提交凭据，commit 前先 `git status`。

## 4. 下一阶段计划（按序推进，每阶段有验收门）

### Phase 0：训练基建加固（半天，先做，否则长跑还会白跑）

- `src/rl_train.py`：
  - `model.learn(...)` 包 try/finally，异常/中断也 `model.save(save_path + ".crash")`；
  - CheckpointCallback 改为按墙钟可预期的档位（如每 25k timesteps 一档：save_freq=25000//envs）；
  - `model.learn(..., tb_log_name=args.run_name)` 隔离 tb 目录（根治铁律 5）。
- 验证：启动一个 5 分钟短训，确认 ckpt 落盘、tb 目录隔离、kill 训练后有 crash 存档；`rl_smoke_test.py` 24/24。

### Phase 1：鲁棒性续训（消灭 91% → ≥98%）

- 目标：全新种子域全程率 ≥98%，消除边缘出生姿态早退。
- 命令：从 champion2 续训 1M 步，加强出生课程：
  `--spawn-lateral 2.2 --spawn-heading 15 --lr 1e-4 --run-name robust2`
  （lr 从 1e-4 起步；若 ep_len 长期不升，参考 journal 中 v1/v2 的经验调 ent_coef 0.02-0.03）。
- 验收门：held-out 种子 8000-8019（20 局 × 2500 步，同步）crash_rate ≤ 5% 且 avg_steps ≥ 2400；
  加测 seed 7104 同类边缘出生 10 局全过。
- 通过则按 halloffame 命名规范入库（日期-步数-验收指标），更新 journal。

### Phase 2：异步/实时鲁棒性（通往真车的控制语义）

- 目标：策略在异步实时控制下不撞（当前冠军异步必撞）。
- **首选方案（无需改 sim）**：客户端侧控制延迟课程——gym wrapper 里以概率保持上一动作
  1-2 个物理步（随机 action delay），同步模式内模拟实时控制抖动；从 Phase 1 模型续训 500k。
- 备选方案（改动大）：sim 侧异步遥测改为定时器驱动（当前疑依附相机渲染导致 headless 停摆），
  需改 `TcpCarHandler` 异步路径并重新验证 smoke。
- 验收门：`rl_eval.py` 异步默认模式（不挂 `--sync`）连窗口 sim，20Hz 下 10 局 avg ≥ 1200 步、
  crash ≤ 20%；同步口径验收指标不得回退（防止拆东墙）。

### Phase 3：视觉轨道重启（真车迁移的唯一路径）

- 现状卡点：离屏视觉 sim 长时不稳定（AsyncGPUReadback 挂起 ×2、断连 ×1，见视觉报告）；
  窗口遮挡也会挂起回读；视觉必须图形设备，`-nographics` 不可行。
- 推进顺序：
  1. 先给相机回读加超时兜底（挂起 N 秒→重建回读/重启该 worker 的 sim，loud 记录）；
  2. 2-4 个有窗 sim 平铺（不遮挡）跑视觉续训，从基线 7.5k 步模型续到 ≥100k 步；
  3. 吞吐仍受限就降相机分辨率（120×160 → 96×128）或降 cam-fps 到 15。
- 验收门：held-out 10 局 avg ≥ 1500 步（基线 444 的 3 倍以上）。

### Phase 4：泛化与导出（收尾）

- 多场景：EditorBuildSettings 里有 11 个场景；先零样本测冠军在其他赛道的存活，
  差则多场景混合训练（每 worker 绑不同场景端口）。
- 导出：冠军策略（MlpPolicy）转 ONNX/TorchScript，写一份真车推理侧的输入归一化说明
  （观测向量定义在 `/Users/hwp/Dev/DKC/GymDonkeyCar/gym_donkeycar/envs/donkey_sim.py` 的
  state 观测组装处，逐字段列出物理含义与量纲）。

## 5. 工作方式要求

- 每阶段结束：更新 `outputs/rl_journal.md`（时间戳 + 决策 + 证据），验证成功即 commit+push。
- 任何"看起来没问题"不算数：以 smoke 24/24、评测日志数字、Player.log 证据为准。
- 改动 `AGENTS.md` 已文档化的行为时，同步更新 AGENTS.md。
- 用户沟通用中文，结论先行，给出可复核证据（文件:行号、命令输出、日志路径）。
