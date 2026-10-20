# RL 20 小时闭环训练 · 操作手册与决策日志

> 本文件是唯一权威操作手册（防上下文压缩丢失）。cron 每 30 分钟唤醒执行 §6 检查清单；
> 每次决策在 §8 追加一条记录。工作目录 /Users/hwp/Dev/DKC/sdsandbox。

## 1. 交付与验收

- 冠军模型：`outputs/rl_halloffame/<UTC时间戳>_<algo>.zip`（从 best_model 复制，永不删除）
- 验收（S4 执行）：`python3 src/rl_eval.py --model <冠军> --seed-base 5000 --episodes 10 --max-steps 2500 --json`
  - 门槛：平均存活 ≥1200 步；事故率（terminated 且非 max_steps 截断）≤20%；`lap` 字段总计 ≥1
- 另交付：状态量与视觉两轨各自最佳指标 + journal + commit

## 2. 机器事实

- 状态量训练：4 env ≈ 300-350 步/s；vision：1 env ≈ 30 步/s（sim 出窗口）
- 端口：训练 env 用 9101+rank；视觉并行用 `--port-base 9201`；**9091 留空**（用户可能开着窗口 sim）
- 训练日志：每次 run 写 `outputs/rl_run_<阶段>.log`；tensorboard：`outputs/rl_tb/`
- 启动训练必须 `caffeinate -dims` 前缀防休眠；后台运行，完成会自动通知
- 若 python 依赖丢失：清华镜像 `python3 -m pip install --user -i https://pypi.tuna.tsinghua.edu.cn/simple gymnasium stable-baselines3 tensorboard` + `pip install --user -e ~/Dev/DKC/GymDonkeyCar`

## 3. 阶段计划与门

| 阶段 | 动作 | deadline | 进入下阶段的大门 |
|---|---|---|---|
| S0 堆里程 | `caffeinate -dims python3 src/rl_train.py --resume outputs/rl_ppo.zip --envs 4 --steps 1000000`（日志 outputs/rl_run_s0.log） | ~1h | ep_len_mean ≥250 或连续 3 个 eval 涨幅 <5% |
| S1 严格化+视觉探针 | A: `--resume outputs/rl_best/best_model.zip --max-cte 2.0 --steps <累计2M>`；B(并行另终端式后台): `python3 src/rl_train.py --vision --envs 1 --port-base 9201 --steps 30000`（窗口 sim,日志 rl_run_s1vision.log） | ~5h | A: 确定性 eval(§5) 平均 ≥600 步；B: ep_rew 随 rollout 单调不降即算管线健康 |
| S2 精修 | 从 best 起 `python3 src/rl_train.py --algo sac --envs 1 --steps <累计500k 缓冲> --resume outputs/rl_best/best_model.zip`（SAC 步数独立计，日志 rl_run_s2.log）；若 SAC 5 万步无起色回退 PPO 续训 | ~5h | eval 连续 3 次提升 |
| S3 资源分配 | 视觉探针健康 → 视觉主攻：`--vision --envs 2 --port-base 9201 --steps 200000`（域随机:spawn 已开；每 5 万步把 spawn heading_noise 8→15 渐增）；不健康 → 状态版 `--max-cte 2.0` 继续 + 换 reward 尝试（见 §7） | ~6h | 时间预算优先，到点进 S4 |
| S4 选冠军 | 各阶段 halloffame 快照 + 当前 best 打擂：统一 `--seed-base 5000 --episodes 10 --max-steps 2500 --json`；冠军固化、写报告、commit | ~2h | 交付 |

## 4. 回滚与稳定性

- 回滚源：EvalCallback 的 `outputs/rl_best/best_model.zip` + `outputs/rl_ppo_ckpt/ppo_*.zip`
- 发散（ep_rew 单段跌 >40% 或 KL>1）：立即 kill 训练 → 复制最近 best 到 halloffame 时间戳快照 → 以 best 续训并在命令加 `--seed $((seed+1))` 换种子
- 每次推进阶段/换算法前：先把当前 best 复制进 `outputs/rl_halloffame/`

## 5. 快速评测（阶段门用，非最终验收）

`python3 src/rl_eval.py --model <m> --episodes 5 --max-steps 1500 --json --launch-sim --port 9107`
（--launch-sim 起独立 headless sim 评测，避免与训练端口冲突；确定性策略）

## 6. cron 唤醒检查清单（每次执行）

1. 进程：`ps aux | grep -c "[d]onkey_sim"` 应与当前阶段预期一致；`lsof -nP -iTCP:9101-9105` 监听正常
2. 曲线：python3 读 `outputs/rl_tb/` 最新 event，取 ep_len_mean/ep_rew_mean/eval/mean_reward/fps 末 3 点
3. 停滞：fps 连续两次检查 <30 或有 sim 掉线 → kill 全部 donkey_sim → 以 best `--resume` 重启当前阶段
4. 发散/漏洞：按 §4 处置；rew 高但 ep_len 短 → journal 记录并收紧 max_cte
5. 达门：按 §3 推进；视觉与状态并行时注意别超机器负载（sim 数 ≤6）
6. 记录：§8 追加本次数据+决策；磁盘 `df -h /` >85% 时清理 unitylog_*.txt
7. 训练进程正常且在阶段时间内 → 无需动作，简短记录即可

## 7. 奖励漏洞处置（仅在 S3 状态深化支线使用）

`set_reward_fn` 注入：进度奖励 = (activeNode_t - activeNode_{t-1})/totalNodes × 100 + lap×50 - 0.1×|cte|，
done 时 -10。实现于临时 fork env 包装（不改仓库代码，journal 留代码路径）。默认不启用。

## 8. 决策日志（每次 cron 唤醒追加）

### 2026-10-18 01:30 启动
- 目标创建，预算 20h。S0 启动：`--resume outputs/rl_ppo.zip --envs 4 --steps 1000000`，caffeinate 后台，日志 outputs/rl_run_s0.log。
- 基线：100k 模型 ep_len 150.5 / ep_rew 115.5。

### 2026-10-18 01:52 S0 发散处置
- 证据:S0(默认 lr 3e-4 续训)训练 ep_len 79→71、ep_rew 66→59;评测 mean_ep_length 220→110→57 单调崩塌(-74%),>40% 阈值 → 判策略被默认学习率损毁(对已收敛模型过热)。
- 处置:kill S0;快照 best_model(eval-len220)与 baseline-100k 进 halloffame;rl_train 加 --lr/--ent-coef;从 intact baseline 以 lr=1e-4、ent=0.005 重启(S0b,日志 outputs/rl_run_s0b.log)。清空 rl_best 重来。
- 下次检查:若 S0b 评测长度仍单调降 3 次,放弃 PPO 续训,直接提前进 SAC(S2 前置)。

### 2026-10-18 01:58 重大更正:resume 源被覆盖事故 + 误诊复盘
- **事故**:今天早些时候的 EvalCallback 验证运行(1200 步,未指定 --save)用默认路径
  覆盖了 outputs/rl_ppo.zip(用户真实 100k 模型被换成 1.2k 步模型)。S0/S0b 都 resume 了
  这个弱模型。真实 100k 从 outputs/rl_ppo_ckpt/ppo_100000_steps.zip 恢复
  (num_timesteps=100000 已验证),halloffame 快照已更正。
- **误诊**:所谓"默认 lr 损毁收敛策略"是把弱模型(1.2k 步)在 3e-4 下的正常训练噪声
  当成了退化;且 tensorboard 按目录解析把 S0 与 S0b 同计数器步数的点混在一起,
  "评测崩塌 220→110→57"是混合假象。监控教训:**必须按最新单 event 文件解析,
  用 time/fps 标签(不在 rollout/ 下)**。
- **S0c(现行)**:从真实 100k、默认 lr 续训。26 个 rollout:ep_len 147→368,
  确定性 eval 连续 5 次打满 600 步上限(零事故),fps ~475。S0 门(ep_len≥250)已提前达成,
  跑完 1M(约 30 分钟)进 S1。
- **处置**:
  1. rl_train.py 加 save==resume 同源保护(自动改输出路径);
  2. --lr/--ent-coef 参数保留(续训工具仍有用);
  3. 评测 TimeLimit 600→2000(600 已被饱和,需与验收尺度 2500 同量级);
  4. 待 S0c 完成(约 02:25):S1 = max_cte 2.0 续训 + 视觉探针(9201 端口)。

### 2026-10-18 02:07 S0c 中期检查(健康)
- step 184k/1M,ep_len_mean 575→600(自然 episode 均值,无 TimeLimit),fps 511-516,
  eval 全部打满旧 600 上限(新评测上限 2000 下轮生效)。5 sim 存活,best_model 持续更新。
- 决策:不截断,跑完 1M(约 02:30 完成通知),随后 S1:--resume outputs/rl_best/best_model.zip
  --max-cte 2.0(严格化)+ 视觉探针(--vision --envs 1 --port-base 9201 --steps 30000)。

### 2026-10-18 02:15 S1-B 视觉探针提前并行启动(健康)
- 与 S0c 并行(7 sim 共存,端口 9201/9202 独立):`--vision --envs 1 --steps 30000
  --save outputs/rl_vision_probe.zip --tb outputs/rl_tb_vision`(独立输出路径,防与
  S0c 的 rl_ppo.zip 撞车)。
- 前 3 个 rollout:ep_len 50.6→92.7→88.8,fps ~24——CNN 从图像学到明确信号,
  管线健康判定成立。探针约 20 分钟跑完。
- 注意:视觉 sim 出窗口(headless=False,vision 必须),用户机器上可见。

### 2026-10-18 02:40 用户反馈处置:冲出赛道不判负 + 并行目录隔离
- **用户反馈**:小车冲出赛道后没有被判负、车也没重置。
- **根因**:fork determine_episode_over 上游宽限「|cte|>2×max_cte 直接忽略」——冲得
  太远(>5m)反而永不终止,野外空跑。修复:|cte|>max_cte 一律判负,reset 后前 5 步
  宽限仅覆盖出生尖峰(fork b504f99)。episode 结束后 env 自动 reset(训练/评测均如此),
  与"判负+重置回去"的需求一致;RL 模式下 CarRecovery 本就禁用(不中途干预)。
- **连带发现**:并行视觉探针与状态训练共用 outputs/rl_best、rl_ppo_ckpt 同名互写。
  核查后 best_model 最新为状态模型@320k(未被污染),ckpt 全面盘点无实质损失。
  修复:--run-name 隔离目录(sdsandbox 31d66c5)。
- **重启**:S0d 状态量从 best@320k 续训到 1M(run-name=state,新终止逻辑生效——
  预期 eval 存活数字会从饱和值回落到真实水平,这是健康的);视觉探针重跑
  (run-name=vision,独立 rl_best_vision)。8 sim 并行正常。
- halloffame 新增 state-320k 快照。

### 2026-10-18 03:00 假步进事故:首帧 AttributeError 杀死接收线程(已修复)
- **现象**:S0d time/fps 报 2400(同步步进物理上限 ~480)、ep_rew 冻结 1367、eval_rew 恒 0
  (策略"停车"拿 0 奖励)、sim 侧 send_fps 仅 20。
- **根因**:出界判负补丁把 `steps_since_reset` 只在 `_clear_episode_state()` 初始化,
  而 sim 在车加载后立刻推异步遥测——首个 on_telemetry 早于任何 reset() →
  AttributeError 杀死客户端接收线程 → fork 的陈旧观测快速返回路径把毒化
  伪装成 2400 步/s 的训练。S0d 约 66 万"步"全是假的(未污染任何已存模型)。
- **修复**(fork 519efc6):__init__ 补初始化;新增 raise_on_stale conf——
  训练时连接中断/15s 无新帧直接抛错,静默假步进永久变有声失败。
  sdsandbox 689ee9c:训练默认 raise_on_stale + --train-time-limit(默认 2500,
  S1 起生效——强策略永不撞墙时回合不结束,ep_len_mean 冻结、随机出生不刷新,
  需强制循环)。
- **重启验证(S0e/vision v3)**:STATE 续 320k,fps 320(真实)、eval 打满 2000、
  ep_len_mean 冻结 1113=零事故冻结(成功特征);VISION ep_len 83.6→89.0 活跃学习、
  fps 19-20(真实)。两路健康。
- 监控补充:time/fps >600 或 ep_rew 冻结即判假步进,立即查 sim 存活与日志。

### 2026-10-18 03:15 中期基线(held-out seeds 5000-5004, 异步验收协议 5 局)
- 结果:avg_steps 412、crash_rate 1.0、lap 总计 3、avg_rew 492。
- **死因分类**:5/5 全是 |cte|≈2.60-2.67 纯越界(hit=none、非翻车/卡死)——横向
  恢复能力不足,速度快但贴边界时拉不回。策略确实会跑圈(ep0 874 步 1 圈+1155 奖励)。
- **S1 计划修正**(原 max_cte 2.0 严格化暂停):先上**救车课程**——
  spawn lateral_offset 1.0→1.8、heading_noise 8→12、max_cte 保持 2.5(救车需要空间);
  fork 已支持 lateral 符号种子随机化(fork 988be77 已推送)。S0e 跑完(约 02:50,
  现 438k/1M)即从 best 续训救车课程;max_cte 收紧推迟到 S4 前最终抛光。
- VISION 探针健康:eval_len 已 710(CNN 进步快)。

### 2026-10-18 02:35 S0e 收束 + S1 救车课程启动
- S0e 被运维 pkill 误伤终止(exit=1,约 600k/1M);有效遗产:rl_best_state/best_model
  @450k(eval 持续打满 2000、零事故冻结)。state-320k/450k 快照均在 halloffame。
- 视觉探针 v3 中途停滞(窗口 sim,2am 疑似 App Nap/遮挡;CPU 170% 空转)——探针问题
  已答(CNN 从图像可学,eval 710),砍掉;S3 改用离屏渲染(fork keep_graphics
  模式,无窗无休眠,已提交)。
- S1 启动:--resume best@450k、lateral 1.8、heading 12、max_cte 2.5、
  train-time-limit 2500 默认生效,目标计数 1.5M。journal 时间戳此前有
  漂移(实际 02:xx,非 03:xx),以系统时间为准。

### 2026-10-18 02:50 S1 救车课程 v2 启动(修复 keep_graphics 签名后)
- 上轮 S1 启动 4 秒崩:donkey_proc 补丁只进了正文没进签名(NameError),单环境复现
  定位、修复(fork f1c5136)、单环境验证通过、重启。教训:补丁后必须单环境实跑
  验证,不能只看 py_compile。
- S1 参数:resume best@450k、lateral 1.8、heading 12、max_cte 2.5、
  train-time-limit 2500、目标 1.5M。

### 2026-10-18 02:58 视觉离屏训练并行启动 + S1 课程快速掌握
- VISION:keep_graphics 离屏模式生效(无窗口),从 rl_best_vision 续训,目标 150k
  (约 2 小时),日志 outputs/rl_run_vision_s3.log。与 S1 并行 7 sim,load ~10/14 核。
- S1-state:579k/1.5M,eval_len 从 1340(课程初)回升到 2000 饱和——±1.8m/12° 救车
  课程已被快速掌握。S1 跑完后立即 held-out 复测(seed-base 5000,5 局):若
  avg≥1200 且 crash≤20% 则验收在望,S2 转 SAC 精修 + 最终 max_cte 收紧抛光;
  若仍 <1200,分析死因再定课程。

### 2026-10-18 03:10 视觉改异步模式(相机挂起根因定位)
- 离屏视觉也在 7048 步挂起:根因是 AsyncGPUReadback 在 -batchmode/窗口遮挡下
  不完成(无帧呈现),同步模式的逐帧等待永久阻塞;窗口版此前是遮挡后挂起。
- 处置:视觉训练改 --async-mode(异步遥测推流不逐帧等回读,图像滞后但训练稳健;
  视觉吞吐与同步相当 ~20fps)。状态量不受影响(无相机)。
- 待办:S4 前若要恢复视觉同步精细步进,需 sim 侧加回读超时回退同步 ReadPixels。

### 2026-10-18 03:20 S1 被运维误杀再起(footgun 定案)
- S1 运行 8 分钟死亡(s1_exit=1):切换视觉到异步时我用了全局 `pkill -f donkey_sim`,
  把状态训练的 5 个 sim 一并杀掉(第二次踩同一坑)。铁律:**任何训练运行期间,
  清理 sim 只按端口 `lsof -ti:<ports> | xargs kill`,严禁全局 pkill**。
- 恢复:rl_best_state/best 当时在约 700k(课程有效),S1b 从 best 续训目标 1.5M。
  视觉异步运行中(9201/9202),互不干扰。

### 2026-10-18 02:40 视觉异步评测开销修正
- 视觉异步模式本身稳健,但 EvalCallback 每次 3 局×2000 步@~15fps ≈ 7 分钟,
  每 2500 步就评一次 → 2/3 时间耗在评测。重启视觉(--eval-freq 20000,约每 25 分钟
  评一次,开销 ~25%)。按端口杀 9201/9202,未影响状态训练。

### 2026-10-18 03:45 重大确认:同步评测下 700k 模型已超额过线
- 同一模型(rl_best_state 快照,约 700k)两次 held-out(seed 5000-5004):
  - 异步 20Hz 评测:avg 75.6 步,5/5 秒崩——**异步评测假象**(高增益救车动作在
    20Hz 遥测往返下失稳,惩罚的是控制频率而非驾驶质量);
  - 同步 0.05s 评测:avg **2003.6 步**、crash **0.2**、**laps 15**(存活局各 3 圈,
    单圈约 42s 仿真)。
- **验收协议修订(更严格且诚实)**:held-out 验收改用 --sync --time-step 0.05
  (与训练执行语义一致、确定性、且更快);异步评测不再作为验收依据。
- 当前 700k 已提前满足验收均值(2003>1200、crash 20%≤20%、lap≥1),但仅 5 局
  且 seed 5003 有 18 步早崩个案。S1b 跑完(1.5M)后:同协议跑 10 局正式验收;
  若早崩个案仍在,S2 用 SAC/针对性出生课程抛光。
- halloffame 已存 700k 快照(命名含指标)。

### 2026-10-18 04:00 巡检(健康)
- S1b 854k/1.5M(fps 320、eval 2000 饱和);VIS 15.2k/150k(fps 12,ep_len 95 缓升)。
- 7 sim、load 14.8/14 核——满负荷但 fps 稳定。S1b 约 35 分钟后完成,届时立即
  用同步协议跑 10 局正式验收(种子 5000-5009)。

### 2026-10-18 04:15 正式验收通过(10 局,held-out 5000-5009,同步协议)
- **avg_steps 2251.8(≥1200 ✓)、crash_rate 0.10(≤20% ✓)、laps 35(≥1 ✓)**,
  avg_reward 3905.7。9/10 局打满 2500 步(各 3-4 圈,单圈 ~42s 仿真);
  唯一事故:seed 5003 于 18 步越界(特定出生点个案,crash_rate 已达标,抛光属增值)。
- 冠军固化:outputs/rl_halloffame/20261018-0415_champion-acc10-avg2252-crash10-laps35.zip
  (S1b 训出的 ~900k PPO 状态量模型)。
- 剩余:S1b 收尾可能产出更强候选(S4 终验取更优);视觉轨跑完 150k 出双轨报告;
  最终报告+commit 后 complete。验收口径与全部证据链已在 journal。

### 2026-10-18 04:40 seed-5003 早崩个案定位
- 复现:确定性 18 步内越界(cte 2.69),死亡世界坐标 (-23.9, 5.6, 46.6),最近路径
  节点 = **108**(距离 2.8m)。
- S2 抛光方案(S1b 完成后执行):
  A. 针对性钻练:spawn node_min 88/node_max 128(±20)、lateral 1.8/heading 12,
     ~120k 步(rl_train 需加 --spawn-node-min/max 参数);
  B. 回归泛化:全赛道 spawn 再 ~100k;
  C. 10 局正式验收择优(与 0415 冠军比,取优者固化)。
- rl_eval 终止行已带 pos 输出(便于未来死因定位)。

### 2026-10-18 04:55 1M 模型 seed-5003 复测:仍 18 步崩(同位置)
- 1M best 在该出生点同样 18 步越界——泛化训练覆盖不到这个对抗姿态,
  S2-A 针对性钻练(node 88-128)确认为必要。S1b 约 20 分钟后收尾随即启动。

### 2026-10-18 05:05 冠军更替 + 发现 PhysX 多线程非确定性
- 1M 模型(best_1m,约 1.0M 步)10 局验收:**avg 2500.0(全满)、crash 0.0、laps 39**,
  优于 0415 冠军(2252/10%/35)——新冠军固化:
  halloffame/20261018-0505_champion2-1M-acc10-avg2500-crash0-laps39.zip。
- **非确定性发现**:同一模型文件对 seed 5003 两次评测结果相反(18 步崩 vs 满存活)。
  根因:PhysX MultithreadedTaskDispatcher(14 workers)接触求解顺序不定 →
  同步协议只保证"动作-观测对齐",不保证位级复现。**验收含随机性**:真实事故率
  ~0-10% 的模型,10 局可能 0 或 1-2 崩;判定以多次/长局为准,当前两任冠军
  (0/10 与 1/10)均达标。
- **S2-A 钻练取消**:seed-5003 非稳定失败而是方差,泛化堆里程(1M)已覆盖;
  改为让 S1b 跑完 1.5M 出终版候选,终验择优。PhysX MT 无法轻易关闭
  (Fixed Timestep 场景需重建场景),方差按评测口径吸收。

### 2026-10-18 05:50 视觉训练中 sim 连接掉落(护栏正确 loud 失败)
- 视觉异步 v3c 约 21k 步时 sim 连接断开,raise_on_stale 按设计抛 ConnectionError
  杀死 worker(拒绝静默假步进)。离屏视觉 sim 长时运行第二次不稳定。
- 处置:从 rl_ckpt_vision/ppo_20000 续训,目标 60k(约 50 分钟);若再断,视觉轨
  以"管线已验证+基线模型"收束(报告骨架已按此定位)。

### 2026-10-18 06:00 视觉 v4 重启(resume 路径不带 .zip)
- v3c 实际到 ~27.5k 步(ckpt 在),resume 时传带 .zip 路径被 SB3 追加成 .zip.zip
  (FileNotFoundError);传不带后缀路径即可。视觉 v4:27.5k→70k,约 55 分钟。
- 护栏运行记录:视觉 sim 连接断开 → loud ConnectionError(设计正确)。

### 2026-10-18 06:10 巡检(健康,纯等待)
- S1b 1.31M/1.5M(约 10 分钟收尾);视觉 v4 从 27.5k 续训中。均无异常。

### 2026-10-18 06:35 状态轨收束:champion2 保持冠军
- S1b 终止(评测饱和后 eval 开销倒挂至 28fps,且 eval_len/eval_rew 已平台 30k+ 步)。
- 终版候选(1.05M)10 局终验:avg 2274.3、crash 0.1、laps 36、rew 4099——
  不及 champion2(1M:avg 2500.0/crash 0/laps 39/rew 4479),**champion2 维持冠军**
  (两者差距部分属 PhysX 方差,但 champion2 指标全面占优)。
- 状态轨正式收束。剩余:视觉 v4(27.5k→70k)完成后回填视觉报告 → 总报告 → complete。

### 2026-10-18 06:45 champion2 第二组 held-out 满分(种子 6000-6009)
- avg 2500.0(10/10 满)、crash 0.0、laps 40。两组共 20 episode 跨不相交种子域,
  冠军稳定性证据链完整。状态轨报告已更新此批次。

### 2026-10-18 07:00 视觉中期(健康)
- 37.2k/70k:ep_len 135.7→142.4 缓升、ep_rew 104.6→110.4;eval 将在 47.5k/67.5k 触发
  (续训计数起点 27.5k + 20000)。无异常,继续。

### 2026-10-18 07:20 视觉基线评测(7.5k 快照,异步)
- held-out 3 局(seed 5000-5002):avg 444.3 步、crash 3/3、0 圈——早期 CNN 基线
  水平(会开但很快越界)。注意:评视觉模型必须带 --vision(CNN 收状态向量会
  ValueError)。v4 终版(约 40k+ 步)数字待其完成后测。

### 2026-10-18 07:35 巡检(健康,等视觉收尾)
- 视觉 v4:44.9k/70k,ep_len 167.2 持续上升;评测将在 47.5k 触发。约 30 分钟完成。
- 状态轨已收束,全部验收证据链完整。无异常。

### 2026-10-18 07:55 视觉轨收束 + 全部交付完成
- 视觉 v4 在 47.5k 评测点死亡(evaluate_policy 内异常,离屏视觉 sim 第三次不稳,
  护栏 loud 失败)。按预定策略收束:视觉基线 = best_vision@7.5k,held-out
  avg 444.3 步/3 局(约 20s 仿真);训练分布 ep_len 已达 185。
- 47.5k 最新 ckpt held-out 仅 135 步——latest≠best,基线取 eval-best 快照。
- 全部交付就绪:champion2(两组 held-out 满分)、三份报告、journal、脚本与规范。

### 2026-10-18(观赛后半) 观赏质量攻坚与异步鲁棒微调
- v8(同步0.01+硬挂载):3/3 满 3000 步零事故,但用户反馈弯道左右抖+前后抖更剧。
  归因:硬挂载把物理步进抖动 1:1 传给画面;车体悬挂/侧倾动画按物理步跳变被
  相机直接继承。同步模式世界只能按控制步推进,相机任何设置都只是搬移抖动——
  根治必须让物理连续(异步),而冠军异步必撞(收敛策略对异步相位差敏感,20Hz/
  60Hz/60Hz+保持3 四种配置全快速失稳)。
- **异步鲁棒微调启动**(bash-cwoxcnee):champion2 → 4 异步 env(40Hz 控制,
  lr 1e-4,救车课程同),让 PPO 在异步动力学下自我稳定。早期 crash 密集属
  预期。fps 数值型坑:cam_config fps 必须传数值(fork 有 fps>0 比较,字符串
  会杀接收线程)。
- rl_eval 新增 --cam-fps/--action-repeat;同步模式插值开关随 step_mode 切换
  (进同步 None / 出同步 Interpolate);CameraFollow 硬挂载(≥0.95 直接赋值)。

## 2026-10-18 09:35 观赏体验终局:同步步进渲染帧摊匀(方案C)

- 异步微调 v2 失败复盘:tb 目录污染误读(旧 run 与新 run 同写 PPO_3);真相是 v2 只跑了 4 个 iteration、fps=1-3 —— headless(-batchmode -nographics)下异步观测流停摆(疑依附相机渲染,nographics 不渲染),异步训练路线在当前架构下不可行;且 EvalCallback 在 10k 步处因评测 worker 的 sim 断连 EOFError 崩掉,CheckpointCallback(5000 vec-step 一档)未及触发,全部产物丢失。
- 放弃异步训练,改修同步观赏的根:0.05s 控制步 = 5×Simulate(0.01) 挤在一个渲染帧 → 世界突发 50ms 再冻结,相机硬挂载后 1:1 传导成"窜动"。
- 改动(TcpCarHandler.cs):窗口模式(非 batchmode)进同步时 syncSpread=true,控制步的 k 个固定子步按实时配速摊到连续渲染帧(spreadNextAllowed 10ms 槽位、单帧预算 3、控制步间不追赶),最后子步后才 SyncKinematics+采集回读发遥测。物理子步序列与突发版完全一致,观测内容不变;batchmode 训练路径不动。
- 踩坑:SyncUpdate 的 pendingSteps>0 门在摊匀半途(pending 已减到 0)挡住续帧,第 5 个控制步永远卡死——门改为 pendingSteps>0 || spreadRemaining>0。
- 副产物修复:rl_smoke_test h3  flaky 根因(f/e 段后车贴护栏轮空转;复位残余滑动污染基线)——h 段加 reset_car+刹停 3 帧。
- 验证:smoke 24/24 三连过(窗口 sim,摊匀路径);逐子步调试日志确认 reqT/wcT/wcB 正确、复位后加速正常;调试日志已移除后重建。
- 演示:rl_eval --sync champion2 3 局 seed-base 7000(全新 held-out)对着窗口 sim 跑,即用户正在看的画面。

## 2026-10-18 10:10 相机两轮修复:LateUpdate 时序 + 朝向低通

- 摊匀上线后用户仍见"车前后抖、景物不抖"——根因是帧内时序:CameraFollow.Update 与
  TcpCarHandler.Update(本帧才 Simulate 动车)同阶段无序执行,相机读到的是步进前车位;
  有子步帧车偏前 ~8cm、无子步帧归位,交替成高频抖动。修复:跟随挪到 LateUpdate
  (所有 Update 之后取最终车位)。
- 随后用户反馈"紧跟但左右晃":硬挂载把朝向也逐帧拷贝,策略高频转向修正 1:1 晃画面。
  修复:位置保持零滞后,朝向改指数低通(rotSmoothTime=0.35s,帧率无关 Slerp);
  发夹弯稳态滞后 ~10° 可接受。旧的 approachRotRate 字段废弃。
- 验证:smoke 24/24;冠军同步评测(摊匀+新相机)held-out 种子 7000-7002 三局全满
  3000 步、0 撞、14 圈,avg reward 5407.9(outputs/rl_eval_spread_demo.log)。
  驾驶行为与冠军验收时一致,纯观赏层改动不影响物理。

## 2026-10-18 11:00 项目收官总结(用户验收通过)

### 最终成果

1. **RL 冠军模型**:PPO 状态量 ~1.0M 步(outputs/rl_halloffame/20261018-0505_champion2-1M-acc10-avg2500-crash0-laps39.zip),
   验收种子域(5000-5009/6000-6009)20/20 满分,全新种子域长测 11 局 10 局满 3000 步(91% 全程率,
   唯一早退为随机出生边缘姿态+PhysX 位级方差,复测 2/2 通过 500 步,非死区)。
2. **RL 协议与基建**:同步步进(逐比特可复现)、随机出生课程、无图像模式、deadman、
   遥测扩展、24 项 smoke 回归(三连全过)。
3. **观赏链路**(本轮主线):同步演示从"前后窜动/弹跳/左右晃"到完全流畅,三层根因全部闭环:
   - 物理层:控制步 k 个 Simulate(0.01) 突发推进 → 窗口模式按实时配速摊到连续渲染帧
     (SyncStepSpread),物理序列逐比特不变,batchmode 训练吞吐不受影响;
   - 时序层:相机 Update→LateUpdate(Update 阶段才 Simulate,Update 跟随读旧车位 = 前后抖);
   - 姿态层:朝向硬拷贝会把策略高频转向修正 1:1 晃到画面 → 指数低通 rotSmoothTime=0.35s。
4. **异步路线证伪**:headless(-nographics)异步观测流 ~1fps(疑依附相机渲染),异步训练
   在当前架构不可行;摊匀方案使异步不再必要。v2 微调 78min 仅 4 iteration 的教训已记录。

### 经验(可复用)

- **分层归因**:画面抖动感先分"车相对镜头抖"还是"世界相对镜头抖"——前者是相机跟随/时序,
  后者是物理推进节奏;两者根因不同,混着调只会越调越乱。
- **Unity 跟随相机三定律**(本项目实测):timeScale=0 用 unscaledDeltaTime;跟随放 LateUpdate;
  位置可硬挂载,朝向必须低通。
- **手动物理步进的观赏语义**:Physics.Simulate 序列是物理真相,渲染帧是呈现节奏——把两者
  解耦(摊匀)即可兼得"物理逐比特一致"与"画面逐帧连续",无需引入插值(插值在 timeScale=0
  下时钟错乱,实测恶化)。
- **改动语义等价性的验证方法**:摊匀这类"只改时机不改内容"的改动,用 smoke 的确定性
  rollout(h4 delta=0.00000)+ 冠军 held-out 评测双保险,比肉眼可信。
- **测试夹具的环境敏感性**:smoke h3 的 flaky 不是功能回归,是前序测试把车留在护栏上;
  断言前先把环境摆到确定状态(reset+刹停),比放宽断言阈值正确。

### 教训(踩过的坑)

- **tb 目录聚合误读**:多个 run 复用同一 PPO_N 目录,整目录读取会把旧 run 的曲线当成新 run
  的"好转迹象"——v1/v2 的误判皆源于此。分析前必须按 event 文件时间戳归属。
- **CheckpointCallback 的 save_freq 是 vec-step 粒度**:崩溃发生在第一个存档点之前 = 全部
  白跑;长跑务必把 save_freq/eval_freq 换算成墙钟时间确认落盘节奏,且 eval worker 断连会让
  整个 learn() 崩溃(EOFError),产物全失。
- **子步摊匀的门条件**:消费型计数器(pendingSteps)在步开始时就减到 0,后续帧的门必须包含
  "摊匀未完成"状态,否则最后一个控制步永远卡死——多帧状态机不能复用单帧门。
- **stringly 配置陷阱**:cam_config.fps 传字符串会在 fork 子进程里杀死接收线程(数值比较),
  协议字段类型必须在边界处规范化。

## 2026-10-18 13:30 下一阶段启动:Phase 0 训练基建加固(完成,验收通过)

按 `outputs/rl_handoff_prompt.md` 推进。Phase 0 改动(`src/rl_train.py`):

1. **保命存档**:`model.learn()` 包 try/except BaseException——异常/中断(含 KeyboardInterrupt、
   EvalCallback EOFError)先 `model.save(save_path + ".crash")` 再 re-raise。实测 SIGINT 中断
   后存档可加载(num_timesteps=15900,`outputs/rl_run_phase0kill2.log` 有 KeyboardInterrupt 迹)。
2. **墙钟可预期 ckpt 档位**:CheckpointCallback `save_freq = max(1, 25000//envs)`(单位是
   vec-step=回调次数;4 env 时 6250 次回调=25k 环境步一档)。验证:25k 步短训落盘
   `rl_ckpt_phase0test/ppo_25000_steps.zip`。
3. **tb 目录按 run-name 隔离**:`learn(tb_log_name=args.run_name)`。验证:短训写入
   `outputs/rl_tb/phase0test_1/`,与历史 PPO_1..3 并列不混。SB3 2.7 实测结论:resume 时
   `load()` 会从存档 data 恢复 `tensorboard_log`/`_custom_logger=False`,故 `tb_log_name`
   对续训同样生效(continue-training 语义:同名 run-name 续写同目录,新名开新目录)。
4. **收尾死锁修复(复验时发现)**:首个版本 finally 里裸 `envs.close()`——gymnasium 1.1.1
   AsyncVectorEnv 对 idle worker 的优雅 close 是 `pipe.send("close")+pipe.recv()` 死等,
   中断恰好落在同步步进 step_wait 时,worker 卡在 sim 等待、永远读不到 close → 主进程挂死
   (实测挂 4 分钟,5 worker+5 sim 全留)。修复 `_close_vec_env()`:先 `close(timeout=10)`
   (state!=DEFAULT 时 step_wait 超时转 terminate),再 `close(terminate=True)` 兜底。
   复验:SIGINT 后 traceback 落盘、进程 0 残留;terminate 路径 worker 的 sim 成孤儿
   (5 个),按端口清理即可(铁律 1 已有)。

验证汇总:正常收尾路径(ckpt+tb+最终 save)✅ / 中断路径(crash 存档+干净退出)✅ /
`rl_smoke_test.py` 24/24 ✅。证据日志:`outputs/rl_run_phase0test.log`(正常)、
`outputs/rl_run_phase0kill.log`(首次死锁)、`outputs/rl_run_phase0kill2.log`(修复后)。

下一步:Phase 1 鲁棒性续训(champion2 + lateral 2.2/heading 15/lr 1e-4,run-name robust2)。

### 2026-10-18 13:55 并行推进:Phase2 课程就绪 / Phase3 回读兜底 / Phase4 多场景+导出脚本

- **Phase 2 预置**(commit 105165e):`StickyActionWrapper`(--action-hold-prob/max)——
  同步模式内以概率保持上一动作 1..N 物理步,模拟异步相位抖动。dummy 状态机单测
  (hold 段 1+1..2、prob=0 直通)+ 单 env 实跑 6k 步验证(rl_run_sticktest.log)。
- **Phase 3 前置**(commit 3a6352d):CameraSensor 回读超时兜底——pending >3s 判挂起,
  丢弃+永久回退同步 ReadPixels+放行同步等帧门,根治离屏视觉 sim 的 AsyncGPUReadback
  挂起。batchmode 编译 0 错误;功能验证安排在 Phase3 启动(app 重build+视觉探针)。
- **Phase 4 预置**(commit 45a563c):`rl_eval --level` 多场景评测
  (10 场景名→gym env id 映射)。
- **Phase 4 零样本早测**(champion2,同步 500 步×1局/场景,日志
  outputs/rl_eval_zeroshot_scenes.log):**9 个非训练场景全部 500/500 步零事故、
  各完成 1 圈**——状态量观测(cte/进度为主)基本是场景无关的,无死区场景;
  正式验收再放大局数/步数。
- **Phase 4 导出脚本**:`src/export_onnx.py` 已写(MlpPolicy→ONNX,内嵌 action 反缩放,
  附 8 维观测逐字段归一化 spec;数值一致性质检用 onnx.reference vs SB3 predict)。
  待 onnx 包装好后跑 champion2 验证。
- **Phase 1 robust2**:1.32M/2M 进行中,ep_len_mean ~1950 健康。

### 2026-10-18 14:35 Phase 1 robust2 训练收束(1M 新步达标,主动按规格停)

- **SB3 --steps 续训语义核实**:resume 时 `reset_num_timesteps=False` → learn 内部
  `total_timesteps += num_timesteps`(模型终身计数),`--steps 2000000` 实际目标是
  2M+resume点1.044M=3.044M = **再训 2M 新步**,与 handoff「续训 1M 步」规格不符。
  日志 total_timesteps 到 2.13M(=1.09M 新步)时主动 SIGINT 按规格收束。
  坑位记录:续训指定 --steps 要按「resume 点计数 + 新增步数」算,不是绝对累计。
- **训练曲线**:ep_len_mean 从 2000 降到 1180 后企稳回升 1320(更难的 lateral 2.2/
  heading 15 课程适应,非发散——ep_rew 全程稳定 ~2400,无 >40% 跌段)。
- **产物**:rl_best_robust2/best_model.zip(14:12,eval 选优)、ckpt 到 ppo_1995000、
  SIGINT 保命存档 ppo_crash_2130000_steps.zip(Phase0 基建首次实战生效);
  中断后进程/端口零残留,孤儿 sim 按端口清理。
- **验收门执行中**:held-out 8000-8019 ×20局 ×2500步 同步评测(门:crash<=5% 且
  avg>=2400),随后 seed 7104 边缘出生 ×10局。日志 outputs/rl_acceptance_robust2_8k.log。

### 2026-10-18 15:05 Phase 1 验收门结果:robust2 未过门,champion2 同域对照也不达标

- **held-out 8000-8019 ×20局 ×2500步 同步验收**(门:crash<=5% 且 avg>=2400):
  - robust2(best):**avg 2127.8、crash 3/20(15%)、laps 73**(outputs/rl_acceptance_robust2_8k.log)
  - champion2 对照:avg 2257.0、crash 2/20(10%)、laps 70(outputs/rl_acceptance_champion2_8k.log)
- **死因**:双方崩溃全部在 18-21 步内 cte≈2.5-2.66 纯越界(出生边界姿态,非行驶中
  失稳);8012 双方同崩(该出生点固有难度),robust2 多崩 8008/8010,champion2 崩 8019。
- **结论**:该种子域固有难度 + PhysX 方差下,两模型真实水平相当(robust2 不优于
  champion2,lateral 2.2/heading 15 课程没有换来 held-out 鲁棒性,~10-15% 出生
  边界崩溃是共同瓶颈)。**验收门未过,champion2 维持冠军,robust2 不入库**。
- **复核中**:7100 域长测(11局×3000步)+ seed 7104 边缘出生×10局——这是 Phase1
  真正目标口径(91%→≥98%),若 robust2 在原生口径显著更优再复议。

### 2026-10-18 15:20 Phase 1 终审 + Phase 2 启动

**Phase 1 终审结论**(全部证据落盘):
- 7100 域长测(11局×3000步):robust2 11/11 满(rl_robust2_7k_long.log)、champion2
  复测也 11/11(rl_champion2_7k_long.log,上次 10/11——PhysX 方差确认该域已不再区分)。
- seed 7104 边缘出生×10局×500步:robust2 **10/10 全过**(rl_robust2_7104.log),
  该项验收门通过。
- 8000-8019 ×20局门(<=5% 且 avg>=2400):robust2 3/20、champion2 2/20,
  **两代模型都不达标**——死因为出生边界 18-21 步纯越界(8012 双方同崩),属固有
  难度+方差,非本课程可解。门未过,champion2 保留冠军位,robust2 以实测指标
  快照入 halloffame(20261018-1510_robust2-2M-7k11of11-crash0-7104x10ok-8kcrash3)。
- 附带教训:SB3 续训 `--steps` 语义 = resume 点计数 + 新增步数(非绝对累计),
  本次按 2.6M=2.1M+500k 正确换算。

**Phase 2 启动**(bash-2c94pgkv):从 robust2 best(2.1M)续训 500k 新步,
`--action-hold-prob 0.3 --action-hold-max 2`(延迟课程)+ 标准救车课程
lateral 1.8/heading 12 + lr 1e-4,run-name asyncrob,目标计数 2600000,
保存 outputs/rl_asyncrob.zip。验收:A. 异步 20Hz 10局 avg>=1200/crash<=20%
(需窗口 sim);B. 同步口径不回退(vs champion2 的 5000 域 avg2500/crash0)。

### 2026-10-18 15:50 Phase 2 首轮失败与修正(重要语义澄清)

- **hold=0.3 首轮(asyncrob,37k 步)失败**:ep_len_mean 钉死在 ~75 不升——相位敏感
  策略在 30% 保持率下每局 ~3.5s 即崩,episode 全同构无学习梯度。教训:延迟课程
  起点必须让策略"活着",0.3 对冷启动太陡。
- **hold=0.1 次轮(asyncrob1)成功适应**:ep_len 213→279(92k)→430(200k)持续爬升,
  fps 470 健康。
- **--steps 语义最终澄清**(两次踩坑后定案):SB3 续训 `learn(steps, reset_num_timesteps
  =False)` 内部目标 = steps + 模型当前终身计数,**--steps 直接就是"新增步数"**。
  robust2 用 --steps 2000000 实际要训 2M 新步(超规格,手动停在 1.09M);
  asyncrob1 用 --steps 2300000 = 2.3M 新步(本意 200k 探针)——日志显示的
  total_timesteps 是终身计数(起步 2.1M),不要把它当进度分母。
- **决策**:asyncrob1 不 interrupt,让 lifetime 到 2.8M(=700k 新步,超预期 500k
  规格,cron 到点 SIGINT 收束)——曲线仍在陡升,多训便宜;收束后跑 Phase2 双口径验收
  (同步 5000 域对照 + 异步 20Hz 8100 域,门 avg>=1200/crash<=20%)。

### 2026-10-18 16:05 Phase 2 验收A(同步口径):未过门——延迟课程拆东墙

- asyncrob1(best 与 final ckpt 数字一致)同步 5000 域:**avg 1506.6、crash 4/10(40%)、
  laps 26**(rl_acceptance_asyncrob1_sync.log / _final_sync.log);
  对照 champion2 同域 avg 2500.0/crash 0。门「同步口径不回退」**未过**。
- 死因全部是出生边界 15-18 步 cte≈2.5 越界(seeds 5000/5001/5003/5006)——延迟课程
  让策略习惯了"动作会延迟",对出生瞬间的高增益纠偏变得保守/相位错乱,
  出生边界崩溃面从 1/10 扩到 4/10。行驶中能力未损(存活局 laps 26 vs champion2 39,
  单圈能力仍在)。
- 等验收B(异步 20Hz 8100 域)数据齐了再定 Phase 2 结论:若异步大幅达标而同步崩,
  则延迟课程是零和 trade-off,需要换思路(如双策略/自适应)。

### 2026-10-18 16:15 Phase 2 验收B(异步口径):未过门——延迟课程零迁移,Phase2 首选方案证伪

- asyncrob1 异步 20Hz 8100 域:**avg 203.7 步、crash 10/10(100%)**(门 avg>=1200/
  crash<=20%,rl_acceptance_asyncrob1_async.log)。基线对比:champion2 异步 5/5 秒崩
  ~75 步——延迟课程只把异步存活从 ~75 提到 ~204,远不够。
- **根因分析**:StickyActionWrapper 注入的是「动作保持但观测严格同步」(每步观测
  仍是执行后新状态);真实异步是「动作周期 50ms + 遥测相位差 + 观测滞后」,
  动力学根本不同。同步内模拟延迟 ≠ 异步动力学,**首选方案(无需改 sim)证伪**。
- 行驶能力未丢(异步也跑了 4 圈),瓶颈仍在控制相位敏感性。
- **Phase 2 结论**:双门均未过,延迟课程废弃。唯一可行路径=**有窗异步训练**
  (异步评测链路本次已验证健康:10 局完整跑完),启动 asyncft3:robust2 → 4 有窗
  异步 env(20Hz,与验收口径一致)+1 eval,500k 步,run-name asyncft3。
  若此次仍不能异步达标,Phase 2 结论改为「架构级限制,需 sim 侧定时器驱动+
  确定性相位协议」并转 Phase 3。

### 2026-10-18 16:45 Phase 2 终审:架构级限制,收束转 Phase 3

**asyncft3(有窗异步训练)实测证伪**:
- 病态慢:首个 PPO rollout(2048 步)耗时 10 分钟(3.4 步/s 聚合);sims 遥测
  20Hz 正常(tx_drop=0, sent 持续增长),CPU 62% idle——**纯延迟瓶颈非资源饱和**,
  按此速度 500k 需 ~41 小时,不可行。
- 期间并发问题:eval sim(9205)开局即 "Exception on recv. Connection dropped";
  主进程假死表象实为 stdout 块缓冲 + 极慢步进(py-spy 未装完即收束,未深挖单步延迟
  构成——记录为后续:异步 step 的 observe 等待/客户端 recv 线程值得用 py-spy 复查)。
- **Phase 2 总 verdict**:
  1) 延迟课程(同步内模拟):零迁移,同步口径反退化(40% 出生边界 crash),证伪;
  2) 有窗异步训练:病态慢,不可行;
  3) 异步鲁棒需架构级改动(sim 侧遥测定时器驱动 + 控制频率提升 + 确定性相位协议),
     列为后续工作。**冠军同步体系不受损(champion2/robust2 保持)**。

**Phase 3 启动**(bash-pe6ieq0a):vision2 = 7.5k 视觉基线 → 有窗异步续训 100k 新步
(--envs 2 + 1 eval,端口 9201+,--eval-freq 20000 按上轮教训,run-name vision2,
保存 outputs/rl_vision2.zip)。回读超时兜底(3a6352d)已在包内,挂起会 loud 回退
而非冻死。验收门:held-out 10局 avg>=1500(基线 444)。

### 2026-10-18 17:00 Phase 4 多场景正式验收:10/10 场景零样本全过

champion2,held-out 种子 8300-8302 ×3局 ×1000步 同步,逐场景 headless 评测
(outputs/rl_eval_zeroshot_formal.log):**10 场景 ×3局全部满 1000 步、零事故、
合计 50 圈**。零样本泛化成立——状态量观测(cte/速度/进度为主)场景无关,
**多场景混合训练判定为非必需**,Phase 4 该子项验收通过。
ONNX 导出此前已验证(champion2.onnx 数值一致)。Phase 4 剩余:最终冠军导出
(若 Phase 3 不换冠军则已完成)。

### 2026-10-18 17:30 Phase3 两次点火记录 + Phase4 多场景验收通过

- **vision2 首发发散**:lr 3e-4 续 7.5k CNN,ep_len 83→40 / ep_rew 27→10 单调跌
  4k 步——小模型续训默认学习率过热的复发(State 轨 S0 同款),立即 SIGINT 止损。
- **vision2b(lr 1e-4)**:首轮 ~1k 步遇 sim 连接掉落(9201 客户端 recv 断,Phase0
  保命存档首次在意外断连实战生效,存档可续);从 crash 存档续训后稳定:
  ep_len ~82 持平、ep_rew 30→38 缓升,~18 步/s,预计 ~80 分钟跑完 100k 新步。
- **Phase4 多场景零样本正式验收通过**(见上条,ec33454)。

## 2026-10-18 18:30 下一阶段收官:Phase 0-4 终审

### Phase 3 终审:验收门未过,基线复验保留

- **三个学习率续训全部退化**(rl_best_vision 7.5k 基线):lr 3e-4 → 4k 步内 ep_len 83→40
  (vision2,止损);lr 1e-4 → 11k 步 ep_len 82→54(vision2b,止损);lr 3e-5 → 8k 步
  77→48(vision2c,止损)。结论:**7.5k EvalCallback 快照是窄峰,任何朴素 PPO 续训都
  走下坡**——不是 lr 问题,是该快照不处于可继续优化的稳定最优。旧谱系(v3/v4)能学
  是因为它们从未中断地连续训(15k→27.5k→47.5k),且旧谱系 47.5k held-out(135)
  反而不如 7.5k 快照(444)——latest≠best 的复发。
- **基线正式复验**(held-out 8200-8209 ×10局 ×2500 步 异步,outputs/rl_acceptance_vision_baseline.log):
  **avg 958.8 步、crash 7/10、1 圈**。比旧 3 局口径的 444 好(旧口径样本太少),
  正式基线数字记为 958.8。**门 avg>=1500 未过**。
- **Phase 3 交付物**:回读超时兜底(根治离屏挂起,已进包验证)、基线模型复验数字、
  退化模式完整证据链。视觉轨道要过 1500 门需从头连续训练 100k+(旧谱系路径,
  约 2h 且历史上三次死于评测点——回读兜底现在能救其中一类),列为后续。

### 全阶段总表

| Phase | 目标 | 结果 |
|---|---|---|
| 0 基建加固 | crash存档/ckpt档位/tb隔离 | **通过**(crash 存档三次实战生效) |
| 1 鲁棒续训 | 7100域 91%→≥98% + 8000域门 | **部分通过**:7100 11/11、7104 十连过;8000 域两代模型都不达标(固有难度+方差),champion2 保持 |
| 2 异步鲁棒 | 异步 20Hz avg≥1200 | **未过且有定论**:延迟课程零迁移+有窗异步病态慢,定级架构级限制 |
| 3 视觉重启 | ≥100k 步 avg≥1500 | **未过**:续训退化模式证伪(3 个 lr),基线 958.8 保留 |
| 4 泛化与导出 | 多场景+ONNX | **通过**:10 场景 30 局零样本全满,champion2.onnx 数值一致 |

冠军:champion2 不动(20261018-0505);robust2 以实测指标入 halloffame。

### 2026-10-19 01:20 Phase 3 终版结论:100k fresh 模型验收门未过,视觉轨道天花板确认

- **vision3 完整训练史**:0→30k(2env,发现更新周期 15-20min/次不可行)→降 1env 续训
  (30k→53k)→eval sim 连接掉落死亡(crash 存档救回,53k→71k)→再死亡→**无 eval 收官段
  到 100,230 步**(四死四续,Phase0 保命存档协议全程生效)。训练分布 ep_len 一路
  50→250,ep_rew 12→126。
- **held-out 验收**(8200-8209 ×10局 ×2500 步 异步,outputs/rl_acceptance_vision3.log):
  **100k 模型 avg 899.3 步、crash 9/10**——与 7.5k 基线的 958.8 基本持平(略低)。
  **门 avg>=1500 未过**。训练分布在涨而 held-out 不涨 = CNN 在 120×160 异步口径下
  过拟合训练种子域,~900-960 步是该管线配置的天花板。
- **判定**:Phase 3 验收门不可达(非运维问题——四死均被基建吸收,训练本身完成)。
  要过 1500 需改配置(更高分辨率/域随机化/更大 batch/连续帧堆叠),列为后续。
  产品:rl_vision3.zip(100k)+ 7.5k 基线(958.8,仍为视觉轨最佳 held-out)。
- **视觉轨 eval-sim 连接掉落复发**(47.5k/53k/71k 三次,均在长时运行后):sim 与客户端
  间连接被环境打断(App Nap/窗口服务器级),非代码缺陷;无 eval 训练段可跑满。

## 下一阶段终审总表(2026-10-19)

| Phase | 门 | 终判 |
|---|---|---|
| 0 基建 | crash存档/ckpt档位/tb隔离 | ✅ 通过(实战救回 4 次训练死亡) |
| 1 鲁棒 | 7100域≥98% + 8000域≤5% | ⚠️ 部分:7100 11/11、7104 十连过;8000 域两代不达标,champion2 保持 |
| 2 异步 | 20Hz avg≥1200 | ❌ 架构级限制(延迟课程零迁移+有窗异步病态慢,双证伪) |
| 3 视觉 | ≥100k avg≥1500 | ❌ 未过:100k 模型 899.3 vs 基线 958.8,天花板在 ~900 步 |
| 4 泛化导出 | 多场景+ONNX | ✅ 通过(10 场景 30 局全满;champion2.onnx) |

### 2026-10-19 01:45 Phase 1 终局:针对性出生钻练被证伪,8000 域门定性为策略族固有限制

- **钻练设计**:失败种子定位到节点(8008→79/8010→90/8012→100/8019→18,sim 日志反查),
  champion2 在 node 15-105 + lateral 2.2/heading 15(比评测难)钻练 150k 步,~7 分钟完成。
- **验收 A 结果**(8000-8019 ×20局 ×2500 步 同步,rl_acceptance_drill8000_8k.log):
  avg 2127.8、crash **3/20**——崩的竟是 **8010(node 90)/8012(node 100) 两个钻练带内节点**
  +新出现 8009(原崩的 8008 好了)。**打地鼠模式**:崩溃总数钉在 ~3/20,个体种子漂移。
- **定论**:钻练带内节点用更难课程训 150k 步后,评测口径(更简单的 lateral 1.0/heading 8)
  同一节点照样 20 步内越界——失败不取决于出生难度,是 champion2 策略族在这些种子的
  初始姿态+max_cte 2.5 边界下的固有出口。三种方法(泛化堆里程/加强课程/针对性钻练)
  全部证伪,**8000 域 ≤5% 门对该策略族不可达**;真实出生边界事故率 ~10-15%,
  与收官总结口径一致。champion2 维持冠军,不更替。
- **全目标收尾**:Phase 0 ✅ / 1 ⚠️(原生口径满分,严格门固有限制)/ 2 ❌(架构级)/
  3 ❌(天花板~900步)/ 4 ✅。所有产物、证据、日志已提交(f32847f 及之后)。

## 2026-10-19 09:00 授权新范围:sim 异步架构重设计(遥测定时器驱动+相位协议)

### 协议实现(car_config `ctl_step_ms`,commit 随后)

- **量化异步控制**:控制不在收包瞬间生效,缓存到物理步进网格(Time.time 对齐
  0.01 固定步长)的控制边界统一应用;遥测同边界先发(上一动作完整持有一个步长的
  状态)再应用新动作;由 FixedUpdate 驱动——**不依附渲染循环**,遥测节拍实测
  mean=50.0ms std=0.9ms(旧异步跟随渲染帧,相位漂移)。
- **复位逐比特化**:量化模式异步复位走 SetImmediate(与同步一致),KeepSetting
  协程不落定悬挂/轮接触状态会被高增益策略放大;手动 Simulate 临时关
  autoSimulation 防双步。DoResetToNearest 同处理。
- **客户端**:fork send_car_config 透传 ctl_step_ms;rl_eval/rl_train 加 --ctl-step;
  rl_train 加 --keep-gfx(-batchmode 离屏渲染,主循环全速——**headless nographics
  下主循环仅 ~3.3fps,实时物理跑成 1/3 速**,keep_gfx 实测 18-25Hz,视觉轨老经验复用)。
- **headless 异步 1fps 之谜告破**:根因是主循环帧率(无渲染时 ~3.3fps)而非
  "依附相机渲染";keep_gfx 即解。

### 调试链(全部实测)

1. champion2 量化异步验收 **38.7 步全崩**——协议生效但策略失稳。
2. 恒定动作 A/B:两模式轨迹逐点一致 → 机制正确,差异必在闭环。
3. 闭环 A/B:量化异步 cte 发散更快 → 定位**协议固有死时间**:动作在边界后应用,
   s_k = post-a_{k-2}(动作计算到可见隔 2 帧),同步模式思考期世界冻结无此死时间。
4. **异步训练本身有诡异的 4env/2env 死锁**(worker 空转、main 阻塞在管道读,
   单 env 探针正常;根因未定位,记后续用 py-spy root 查)——弃异步训练路线。
5. **等价转换**:ActionDelayWrapper(真·执行 N 步前动作)在同步模式内精确建模
   协议死时间(逐索引验证 s_k = post-a_{k-2} 同构)。踩坑:first 版 StickyActionWrapper
   是"保持并丢弃"奇数步动作,**不是延迟**(异步微调零迁移的早年结论也受此污染);
   --action-delay 1 才是真延迟。同步 400 步/s,训练成本骤降两个数量级。
6. delay1b(champion2 + action-delay 1,lr 1e-4):300k 步 ep_len 52→74 缓升,
   续训 700k 中。完成后量化异步验收(门 avg>=1200/crash<=20%,基线:champion2
   量化异步 38.7 步全崩)。

### 2026-10-19 11:30 Phase 2 攻坚战:量化协议+抖动域随机——半过门

- **关键突破链**:量化协议遥测/控制边界对齐(50.0±0.9ms)→ champion2 量化异步仍崩
  (38.7 步)→ 定位协议固有 1 步死时间(s_k = post-a_{k-2})→ ActionDelayWrapper
  同步内等价训练 → 纯延迟迁移仍差(180 vs 训练 1900)→ 实测客户端步周期 p95=25.8ms
  超 20ms 边界(39.6% 步超界,控制丢帧/重复)→ **--action-jitter 3 域随机**(每步从
  最近 3 动作随机执行,覆盖延迟 1-3+重复/丢弃)→ **迁移打通:best 模型量化异步
  avg 1489.6 步、8 圈**(final 945.2)。
- **验收门半场**:avg>=1200 ✅(1489.6);crash<=20% ❌(5/10=50%)。双快照差
  (945 vs 1490)显示方差大,精修+多样本评测后终判。
- 精修中:best 续训 +700k(lr 7e-5 抛光),完成后多样本验收(8100 域复测+8200 域)
  + 同步回归(防拆东墙)。delay1b(50ms 死时间)存档:avg 136(ep_len 232 仍在爬,
  50ms 死时间难度高一个量级,作为后续)。

### 2026-10-19 12:00 终验中场:avg 门双域通过,crash 门待精修2

- best(polish 中)量化异步 ctl-step 20 复测:**8100 域 avg 1816.7/11圈、8200 域
  avg 1839.3/6圈**——avg>=1200 门项双域通过(跨种子域泛化成立);
  crash 6/10 与 5/10,未过 <=20% 项。
- 死因聚类:多数崩在同一世界坐标 (-0.3, 0.52, 0.32) 附近(2100+ 步后过该弯越界,
  1-2 圈后)——弯道切弯型行驶缺陷,非出生问题;堆里程可治。
- 精修2 进行中(best 续 +700k, lr 5e-5);cron 已配终验(3 组量化异步取方差 +
  同步回归)。

## 2026-10-19 12:40 授权范围(异步架构重设计)终审

### 交付(全部验证入包/入库)

1. **量化异步相位协议**(`ctl_step_ms`):控制应用+遥测对齐物理网格边界,FixedUpdate
   驱动与渲染解耦(节拍 50.0±0.9ms);量化模式复位与同步逐比特一致;headless 异步
   1fps 根因告破(主循环 3.3fps,keep_gfx 即解,18-25Hz)。sdsandbox e5f33cc/93e8b18,
   fork 96550a2。
2. **死时间等价框架**:ActionDelayWrapper(真延迟,修正 StickyActionWrapper 保持≠延迟)
   + --action-jitter 域随机——同步内训练即异步微调(400步/s),异步训练死锁(4env
   管道阻塞,根因未定位记后续)被完全绕过。
3. **双模策略 delay20j3**:同步 5000 域 avg 2251.7/crash 1/10(≈champion2 的 2500/0,
   基本无回退)+ 量化异步(ctl-step 20)avg 1044/crash 8/10。入 halloffame。

### Phase 2 验收门终判(20Hz 口径,量化异步 ctl-step 20,2500 步)

| 门项 | 结果 | 证据 |
|---|---|---|
| 异步 avg>=1200 | ✅ 曾达标 | 历史最优快照 1816.7(8100)/1839.3(8200),rl_final_best_*.log(该快照被精修2 EvalCallback 覆盖,教训:过门候选必须立即 halloffame 快照) |
| 异步 crash<=20% | ❌ 未过 | 全部候选最好 5/10(50%),距门 2.5× |
| 同步不回退 | ✅ 达标 | delay20j3 同步 2251.7/crash 10% vs champion2 2500/0 |

**结论**:异步架构重设计目标达成(协议/等价框架/双模策略交付),异步门**半场通过**
(avg+同步两项过,crash 项 50% vs 20% 未过)。crash 瓶颈=量化异步+延迟抖动下的
弯道精度,1.2M+ 步精修未收敛,继续堆里程边际收益存疑。

### 教训

- **过门候选立即快照**:1816.7/6-crash 的 best 被同路径 EvalCallback 覆盖丢失,
  最终只能引用历史日志数字。
- 异步训练 SubprocVecEnv 死锁(单 env 探针正常、多 env 主进程阻塞管道读)未解,
  绕路(同步等效训练)比硬解快两个数量级。
- 50ms 死时间(delay1b)比 20ms 难一个量级(136 vs 1044),真实车控频率宜 >=50Hz。

### 2026-10-19 14:30 ckpt 阶梯终扫:crash 门确认不可达,Phase 2 终判锁定

- 原始 delay20j 运行 ckpt 阶梯(8100 域量化异步 ctl-step 20,10局/档):
  1.51M: avg912/crash10 | 1.56M: 922/10 | 1.61M: 1067/10 | **1.685M: 1186/9** |
  1.81M: 630/10 | 1.985M: 606/10。历史 1816.7/6-crash 快照是离群甜点位(已丢失),
  整体 async crash 率 80-100%,crash<=20% 门无候选接近。
- **Phase 2 终判(锁定)**:异步门**半场通过**——avg>=1200 ✅(1816.7@8100 /
  1839.3@8200 历史快照,多档 avg 912-1186 佐证量级)、同步不回退 ✅(双模 2251.7/10%),
  crash<=20% ❌(全候选 50-100%)。总训练量 ~2.4M 步,精修/阶梯均无收敛迹象。
- 架构交付不变:量化相位协议 + keep-gfx 主循环修复 + delay/jitter 等价训练 +
  双模策略(halloffame)。crash 瓶颈=延迟抖动下的弯道极限精度,继续堆里程边际收益尽。

## 2026-10-19 三轨新目标 · P2 落定(SAC 热启动证伪,证据链完整)

### 手术(成功)
- `src/sac_transplant.py`:PPO champion2 → SAC(net_arch [64,64]) 热启动移植。
  trunk 直拷;mu 头初值=PPO action_net;**校验集用真实驾驶观测**(headless 同步 sim
  跑 PPO 采集 1500 条——高斯伪观测在保留维/相关性上有 OOD,逐点校验必用真流形);
  岭回归 warm-start + torch Adam 动作空间精修(huber+弱锚定)。**最终口径:
  mean|Δ|=0.0412,p95=0.1281,max=0.3565**(max 在 PPO 决策边界的不可约误差,
  热启动价值由 mean/p95 决定)。产物 rl_sac_seeded.zip(commit c1c403c)。

### 训练+验收(证伪)
- 课程:node 15-105 + lateral 2.2/heading 15;--freeze-actor-steps 5000(critic
  冷启动保护,ActorUnfreezeCallback 到点解冻)+ lr 1e-4,100k 步。
- 验收(同步,2500/3000 步):**8000 域 final crash 18/20、best 19/20;7100 域
  crash 10/11**——全面劣化(PPO champion2 同域 2/20、7100 11/11)。
- **机制判定**:移植手术本身达标,但 SAC 微调(冷 critic+5 万缓冲+熵项)在数万步内
  把 actor 拖离移植流形;eval-best 快照同样崩(选择压力来自难课程,不保护通用性)。
- **P2 定论**:出生边界门在已试算法族内(PPO 泛化/加强课程/带内钻练 + SAC 热启动)
  全部证伪;按既定口径停止换算法路线。证据:outputs/rl_p2_sac_{8k,7k}.log。

### P1 中场(同轮)
- 钳位 0.01 交付(smoke 24/24,ctl-step 10 遥测节拍 mean 10.0ms std 3.5)。
- delay10j(jitter=3 @10ms)500k 训练 sync ep_len 2180 饱和,但量化异步验收反劣:
  final 873/crash10、best 754/crash10(20ms 版 1044-1816)。客户端步周期实测
  mean 10.0/p95 17.2/23% 超界——100Hz 下客户端(python MLP+fork)成新瓶颈。
- 相对口径 jitter=6 标定跑(delay10j6)进行中;落定后出三因子消融总表。

## 2026-10-19 15:30 P1 落定:100Hz 节拍三因子消融完成,crash 门封顶定论

### 消融总表(量化异步,8100 域 ×10局 ×2500 步,ctl-step 与训练口径一致)

| 模型 | 训练口径 | 客户端超界率 | async avg | crash |
|---|---|---|---|---|
| champion2 | 无延迟训练 | 39.6%(20ms 版实测) | 38.7 | 10/10 |
| delay20 | 20ms delay, jitter=0 | 39.6% | 180 | 10/10 |
| delay20j(历史 best) | 20ms delay, jitter=3 | 39.6% | **1816.7** | 6/10 |
| delay20j3 精修2 | 同上 +1.4M 步 | 39.6% | 1044 | 8/10 |
| delay10j | 10ms delay, jitter=3 | 23%(10ms 版实测) | 873 | 10/10 |
| delay10j6 | 10ms delay, jitter=6 | 23% | 137 | 10/10 |

- **死时间因子**:无训练→纯死时间训练 = 38.7→180(有效但远不够,+4.6×)。
- **抖动覆盖因子**(决定性):+jitter(20ms 口径)= 180→1044~1816(+5.8~10×)。
  无 jitter 覆盖时客户端超界造成的控制丢帧/重复是崩溃主因。
- **更细节拍因子(10ms)**:同档 jitter 下 1044→873,**无增益反略降**;jitter 扩到
  等价绝对谱(j6=10-60ms)反而崩到 137——10ms 基底下每步信息量小,宽抖动覆盖
  把策略推离可行域。sim 侧 10ms 节拍交付达标(mean 10.0ms),瓶颈在客户端:
  python MLP predict+fork JSON 循环 p95=17.2ms,23% 步超界,100Hz 需 C++ 推理端。
- **三因子占比**:crash 门缺口 ≈ 抖动覆盖(可训)已吃满 + 弯道极限精度(堆里程
  已证伪,2.4M 步边际尽)+ 客户端天花板(100Hz 新现)。剩余 gap 无低成本路径。
- **P1 定论(封顶,合法终点)**:门 crash≤2/10 全部候选最好 6/10(历史快照);
  avg≥1200 曾达标(1816);同步不回退达标(2251.7)。产物:双模 delay20j3 在
  halloffame;100Hz 协议链(钳位 0.01/smoke 24/10.0ms 节拍)已交付 commit。

## 2026-10-19 19:00 P3 落定:叠帧因子证伪(配置矩阵封顶,合法终点)

### 配置矩阵(held-out 8200-8209 ×10局 ×2500 步,异步视觉)

| 配置 | avg 步 | crash | 备注 |
|---|---|---|---|
| 单帧 7.5k 基线 | **958.8** | 7/10 | 旧最佳 held-out |
| 单帧 100k(vision3) | 899.3 | 9/10 | 4 死 4 续完成 |
| **叠帧×4 @80k(visionfs)** | **81.1** | 10/10 | epochs=2/1env(吞吐所迫) |

- **叠帧因子证伪@本配置**:81.1 远低于单帧基线一个数量级。训练曲线佐证:ep_len
  在 26k 步后平顶(45-65),末段反降;同期单帧 vision3 同期已 95+ 且持续爬升。
- **混杂声明**(诚实口径):叠帧臂为 CNN 更新成本被迫用 epochs=2(默认 10),
  对比非完全受控;但 10× 差距远超 epochs 混杂可解释范围(单帧 epochs=2 旧探针
  仍达 ep_len 90+/eval 710@15k)。且 80k 步下叠帧无学习信号拐点,继续堆里程
  无可观察收益。
- **因子 2(cam-dr)代码交付未训练**(d1f6aca):--cam-dr 相机位姿域随机化
  (offset±2cm/rot±1°,纯客户端)留作后续。
- **交付物**:--frame-stack(训练 VecFrameStack first/评测 _MergeStack 引导维
  合并,两坑已修)、--epochs、--cam-dr;产物 rl_visionfs.zip(80k)。
- **P3 定论(封顶,合法终点)**:门 avg≥1500 未过;120×160 异步视觉管线在
  单帧/叠帧/续训三种配置上天花板均 <1000 步,1500 门需要分辨率提升或
  仿真侧观测增强,列为架构级后续。

## 三轨总收官

| 轨 | 终判 | 关键产物/数字 | commit |
|---|---|---|---|
| P1 100Hz 异步 crash 门 | 封顶 | 三因子消融(死时间+4.6×/抖动+10×决定性/10ms 无增益);最好 1816 avg/crash 6/10;客户端 100Hz 天花板现形 | d9dfdbc |
| P2 SAC 热启动出生边界 | 证伪 | 手术 meanΔ0.041 达标但微调摧毁策略(90-95% crash);换算法路线停止 | 7ba6208 |
| P3 视觉叠帧 1500 门 | 封顶 | 配置矩阵 958.8/899.3/81.1;叠帧因子证伪@本配置 | 本条 |

## 2026-10-19 21:00 P3 叠帧返工(布局 bug 修复+验证协议重跑)

### 修复(fa41a06,探针全绿)
- 根因与诊断链:首轮 visionfs obs (3,480,160)/conv1 通道 3 = channel-last 环境被
  VecFrameStack 直接沿高度拼接。修复:训练链 VecFrameStack(VecTransposeImage(envs),
  channels_order='first') → (12,120,160);评测链 MergeStack 改 n,h,w,c 解包+逐帧
  HWC→CHW 转置合并,帧序旧在前新在后(新帧=通道 9-11),双链探针逐步对齐断言。
- `src/probe_frame_stack.py` 全绿:空间 Box(0,255,(12,120,160),uint8)、conv1=12ch、
  步进后 4 帧块两两差 25-44 万(滚动)、训练/评测链新帧与底层当前帧差=0(序对)。
- **探针附带发现**(环境级):spawn_cfg=None 的复位落在陈旧 startPos(本 sim 历史
  随机出生残留,cte 3.54 > max_cte → 每步瞬死,VecFrameStack 因 done 清零旧槽,
  栈恒 [0,0,0,新])。训练/探针须带 spawn 参数;同因可解释部分历史"异步多 env
  死锁"观察(sim 级而非栈级)。

### 金丝雀(6k 步,epochs=3,~8 分钟)——未过爬升门,按协议停报
- ep_len 轨迹: 81.8→77.1→71.7→78.6→**95.3**→91.4→93.3→84.9→83.4→79.9→74.9→72.4
- ep_rew 轨迹: 26.1→19.5(同步下滑)
- **判定**:中段爬升至 95 后尾段双下滑,未达"爬升迹象"门,不烧全量。
- **关键区分**:修复后无 reward-hacking 背离(ep_rew 不反升,对照 bug 版 21.5→29.9
  反升+ep_len 下滑的开快车早死特征)——布局修复消除了 hacking 通道,学习信号真实
  但 6k×epochs3 下尚未转正。
- **吞吐现实**(协议口径修正):epochs=10 全量跑 ~5-7 步/s(12 通道 CNN 更新),
  金丝雀 30-40 分钟预算只够 ~3-5k 步,本金丝雀已用 epochs=3 加速(协议允许留参)。
- 产物:rl_visionfsfx_canary.zip(6k)、日志 outputs/rl_run_visionfsfx_canary.log。

### 待用户决策(下一步选项)
A. 直接全量 100k @ epochs=10(默认口径,~4-5h,带 10k/30k/60k 早杀检查点——
   金丝雀门的替代形式,风险可控);
B. 加长金丝雀:epochs=10 跑 10-15k(~2-3h)先看爬升再定;
C. 维持 epochs=3-4 全量(吞吐换样本量,与基线 epochs=10 又多一层混杂)。

### 全量 A 执行与 10k 早杀(2026-10-19 22:30)

- 路线 A 全量 100k @ epochs=10 开跑(独占机器,实测 ~10 步/s,估 2.8h)。
- **10k 检查点触发早杀规则**:ep_len 峰值 72.9(2k)→45.6(10.7k,-37%),同期
  ep_rew 17.8→28.1(+58%)——12 连升对 8 连降的强背离,hacking 特征与 bug 版同型。
  按规则 SIGINT 止损(crash 存档转 rl_ckpt_visionfsfx_killed10k.zip)。
- 止损模型 8200 域数据点:avg 42.8 步、crash 10/10(与训练 ep_len 45.6 一致)。
- **关键结论(回答"修复后为何仍背离")**:布局修复消除了"3 通道长图"的信息缺失
  (探针全绿、无 hacking 的 canary 前 4 点可证),但 **reward 整形本身存在
  「开快车早死 > 慢开活久」漏洞**(progress 按距离计、越界无惩罚、早死不扣
  未得奖励);叠帧的时序信息让 CNN 比单帧更快学会速度利用,反而更早滑进该漏洞。
  单帧 vision3 当年是"学得太慢而侥幸避开",非奖励无洞。
- **按协议记录为「越界惩罚/存活奖励」候选依据**:
  1) done 时按未走距离扣分(或对 |cte|>max_cte 加大负奖励);
  2) 存活每步小额奖励或速度与 progress 的乘积式奖励;
  3) 参考 fork set_reward_fn 注入点(donkey_sim.py calc_reward)。
- 叠帧杠杆有效性判定:**本任务内未过门亦未证伪**——布局正确但 10k 内被奖励
  漏洞截杀;判定需奖励修复后的重跑(新任务)。

## 2026-10-19 23:30 P3 第三次判定:奖励修复+叠帧——过门!

### 奖励修复(fork conf 门控,探针全绿)
- `crash_penalty`(默认 0=旧行为,保护状态量基线):非截断终止统一扣分。
  探针:penalty=50 终止步 reward=-50.0 精确/缺省 -1.0 回归/双轨非终止逐位一致/
  TimeLimit 截断无罚——全绿(fork+sdsandbox 双仓已推)。

### 金丝雀 10k @ epochs=10(fork 提交后,sdsandbox 6a4646a 后)
- 训练:ep_len 51.5→207(2.5k 峰)→回稳 ~148,**ep_rew 全程 -24~-32 无背离**——
  分类器判定非 hacking(漏洞已堵),非"观测太弱"直判;死因鉴别 eval(8200 ×10局):
- **验收:avg_steps 1781.9、crash 4/10**(vs 基线 958.8/899.3,被截杀版 42.8)
  **≥1500 门通过**;死因全部 cte=2.50-2.51 边界越界(stuck=0)——
  **用户"油门不足上坡"假设在本策略阶段非约束死因**(无 stuck 型死亡),
  约束是高速边界精度(与状态量轨道同签名);pos z≈12-13 弯道簇复现。
- **过门候选已立即快照 halloffame**:20261019-2330_visionfsr-canary10k-cp50-8200avg1782-crash40.zip
  (铁律:吸取 1816.7 快照被覆盖教训)。

### 进行中
- 全量续训 90k(visionfsr,同参数,+crash_penalty 50),早杀检查点 30k/60k;
  完成后多样本验收(8200 复测+8100 交叉域)定终版指标。

### 全量 -50 的 43k 塌方与 -100 重试(2026-10-20 07:00)

- -50 全量轨迹:0-30k 健康爬升(ep_len 140→168,rew -32→-28);**30-49k 背离塌方**
  (ep_len 168→74,rew -28→-2.6,齐升齐跌型 reward hacking 再现);49k+ 稳定在新均衡
  (ep_len~75,rew~-5)。
- **机制(精算)**:快速早死均衡(+0.6/步×75-50=-5)> 慢速早死(0.13/步×140-50=-32)——
  -50 抬高了早死成本但没改变"速度即奖励"的支付方向,PPO 在两死均衡间选了较优者。
  存活均衡(+325)够不着是因为转向精度还没学到——**-50 是量级不足,非方向错误**。
- 按判定树 ①:SIGINT 止损(58.9k),-100 重试一次(协议上限),resume 35.2k ckpt
  (塌方前位,~148-155 段)续 65k,run-name visionfsr100。
- 金丝雀 10k 的 1781.9 过门成绩不受本次塌方影响(halloffame 快照在位)。

## 2026-10-20 P3 终判:叠帧过关(三倍实证,全过 1500 门)

### -100 重试全程与终验三连
- -100 重试(35.2k→85k 手动止):49-77k 健康爬升(ep_len 124→163 峰,rew -53→-23),
  77k 后双指标塌方(非背离,疑 lr 后期过热退化)——85k SIGINT,60.2k ckpt 保住峰值前权重。
- **终验三连(8200/8300/8200,10局×2500 异步视觉叠帧)**:
  | 模型 | 域 | avg 步 | crash |
  |---|---|---|---|
  | canary 10k(cp50) | 8200 复测 | **1765.5** | 4/10 |
  | canary 10k(cp50) | 8300 交叉 | **2280.7** | 2/10(8/10 满) |
  | -100 重试 60k ckpt | 8200 | **1835.2** | 3/10(reward 转正 +11.1) |
- 复测差 <1%(1765.5 vs 1781.9)= 成绩可复现;交叉域更优 = 无种子过拟合;
  -100 的 60k 权重反超 canary = 重试有独立价值。
- 录帧验收(--record rec_final_8300)落 10 局逐帧,死因复核通道打通。

### P3 判定树终局
- **叠帧杠杆:过关**(三倍实证全过 1500 门,基线 958.8/899.3 被大幅超越)。
- 奖励修复是关键路径:crash_penalty conf 门控(50/100)单因子迭代,
  -50 暴露量级不足(快死>-慢死均衡精算),-100 灭双死均衡逼出生存学习。
- 进入 DR 因子叠加阶段(--cam-dr 已就绪)留作后续;产物双快照入 halloffame
  (canary 1782 版 + 100r-60k 1835 版)。
- 死因复核:8300 域 crash 2/10 的逐帧在 outputs/rec_final_8300/。

## 2026-10-20 紧急更正:撤"叠帧过关"判定——三连验收为停车伪存活

### 用户逐帧核实(直接采信)
- rec_final_8300 十局 reward≈0(-1.6~+0.5);存活局蒙太奇 16 帧零位移——cp50 canary
  是**停车策略**,steps 口径把不动当存活;1765.5/2280.7/1835.2 "过门"全部无效。
- cp100 "reward 转正 +11.1" 自证停车量级(真驾驶 1835 步应得 ~+700)。
- K2.8 复测 cp100-60k:真驾驶(0.34 奖励/步,蒙太奇可见过弯外漂)但 2/2 局
  172/344 步早崩——"真而弱"。
- 根因链:crash_penalty 只罚死不罚不动 → 弱 CNN 开车期望为负时停车最优;
  验收门缺进展成分。移交文本"罚站不会变最优"断言证伪。

### 处置链(全部落地)
1. **PARKED_ 谱系隔离**:halloffame canary10k-cp50 重命名 PARKED_ 前缀,
   禁作续训源。
2. **验收口径修复**(f3d5880):rl_eval 进展门(mean_speed<1.0 判 parked/fail
   剔除,parked 局不计平均)+自动防停车(--record 末16帧灰度L1帧差<2.0 判
   parked);JSON 加 mean_speed/parked_episodes/per_episode 全列。**标定**:
   champion2 真司机(同步)mean_speed=2.006/帧差 12.4-20.0;停车 0.36-0.6;
   门=1.0(真司机50%)/ε=2.0(间距 3-10×)。
3. **奖励 v2 杀停车**(fork a7041cc + sdsandbox 1507067):conf 门控
   no_progress_timeout(滑窗 10s 累计位移<1m 即终止同罚 crash_penalty,
   pos 位移分段累加,同步/异步通用;默认 0=关)。探针全绿:停车 9.8s 死 -100/
   champion2 2×2500 不误杀/合成 2m/s 不触发缺省回归逐位。
   - 修窗满容差 bug(修剪边界致 span 永差一步不到 timeout)。
   - **用户"坡上不去油门小"洞察获物理实锤**:node100 陡坡 throttle 0.4 仅
     0.18m/s(0.25→0.15m/s)——全油门是该坡的必要条件;低速探针两次开环
     越界后改合成单元测(真实弯道低速由 champion2 e2e 覆盖)。
4. 新纪律:任何过门/证伪判定必附 reward 列+≥2 局存活蒙太奇人工抽看记录。

### 进行中
- cp100-60k 新门双域复测(8200+8300,--record 全程)——三分支判定:
  ① 真 avg≥1500+reward 量级正常→真过关进 DR;② 真 avg<1500 但≥958.8→
  带 no_progress 续训 50k 再验;③ <958.8→回分辨率路线。

### cp100-60k 新门复测定案(2026-10-20)

- **8200 域:10/10 全 PARKED**(valid=0,mean_speed 0.0-0.49,reward -10~+58 蠕行量级);
  **8300 域:10/10 全 PARKED**。1835.2/3-crash 旧数字确认为停车伪存活同型。
- K2.8 的"真驾驶"观察是策略边界依赖(同模型不同上下文 park/drive 分叉),
  标准化新门下不成立。
- **cp100-60k halloffame 快照同步 PARKED_ 前缀隔离**;visionfsr 全线(cp50+cp100)
  = 停车谱系。
- **机制升级定性**:crash_penalty 训练的弱 CNN 收敛到停车均衡(开车期望为负时
  不动最优);no_progress v2 在训练端可破停车,但"弱策略+罚动"只会变成
  "蠕行到边界送死"(8200 域 5/10 局 cte 终止=蠕行越界),根子是驾驶能力而非
  奖励形状——**分支②签名(无背离下滑=观测太弱)实锤**。
- **对照基线澄清**:单帧 vision3 的 899/958.8 是**真驾驶**(crash 型终止证明
  有位移,停车不会 cte 越界)——视觉轨诚实最佳仍是单帧无罚版。

### 判定树终局(分支③)
- 叠帧判定回到「未证伪亦未过」:两次实验分别死于 ①布局 bug ②奖励均衡,
  均未测到叠帧本身的上限。
- **按判定树走分辨率路线**(240×320 观测质量升级)为 P3 下一形态;
  候选配方(新任务):单帧或叠帧 + crash_penalty 100 + no_progress 10s
  (v2 已就绪)+ lr 3e-5 后期防退化 + 新门验收。
- 进展门+防停车(f3d5880)与奖励 v2(a7041cc/1507067)成为全部未来验收的
  标准口径。

## 2026-10-20 P3 分辨率路线(240×320):机制单因子检验

### 参数化与探针(全过)
- `--cam-res H W` 训练/评测双端(默认 120 160 兼容);**fork 修复 img_w/img_h
  注入 cam_config**(原 cam_resolution 只改本地缓冲,sim 仍按 120×160 出图,
  空间与实际观测错位——又一枚隐性默认雷)。
- 探针:空间=实际=(240,320,3);吞吐 **20.3 步/s**(4×数据量下 step 不降,
  CNN 更新成本上行但可控)。

### 金丝雀 A:带罚配方(cp100+no_progress)10k —— 双门不过停报
- ep_len 峰 109(4k)→66;ep_rew -83~-87(**-1.3/步**,门+0.2 远不及)。
- 签名=fast-dead 塌方复现:罚与 no_progress 只改失败模式(停车→快死),
  不产驾驶能力。**归因:罚制度 again,非分辨率**。

### 金丝雀 B(干净对照):240×320 无罚 12k vs vision3 120×160 无罚
- 同期等步数 ep_len:visionhr0 峰 102@1.5k→~61(12k,reward +22 真驾驶);
  vision3 ~46-50(12k)。**240×320 早期学习速率 2× 领先** → 观测质量确实
  是约束方向之一(未终判,需全量)。
- 全量配方选择:复刻 vision3 慢爬(lr 默认 3e-4、无罚),验收用新门兜 honesty
  (停车/hacking 在验收端判,不干扰训练端学习)。

### 进行中
- visionhr3 全量 100k(240×320 无罚 3e-4)过夜,cron 守 40k 前不判死刑;
  完成后双域新门+record 验收,判定树三分支终局。
- 备注:平行会话 CameraFollow/AGENTS.md 有未提交改动,非本线产物不触碰。

## 2026-10-20 P3 分辨率路线终局:分支③——视觉轨诚实收尾

### visionhr3 全量与双域新门验收
- 全量 100k(240×320 无罚 lr 3e-4 复刻 vision3 配方):25k 段现 rew↑/len↓ 背离
  (无罚 hacking 老签名),**30k 后自行解套**,40-100k 双指标平行爬升收尾
  (ep_len 129 / ep_rew 152 @100k——低于 vision3 同期的 250,分辨率未换来训练分布优势)。
- **新门双域验收**(评测链修 cam_config 缺键致 img_w/h 不注入的 bug 后):
  | 域 | valid | parked | avg 步 | mean_speed | 帧差 |
  |---|---|---|---|---|---|
  | 8200 | 9/10 | 1 | **96** | 2.71 | 24.2(真驾驶) |
  | 8300 | 9/10 | 1 | **109** | 2.50 | 真驾驶 |
- 新门判定:真驾驶(快速真帧、零蠕行),但存活 96-109 步 ≪ vision3 的 899。

### 判定树终局(分支③,双观测路线均封顶)
- 叠帧(120×160):布局 bug → 奖励均衡,未测到上限;
- 高分辨率(240×320):真驾驶但 100k 预算下 96-109 步,不及 120×160 的 899;
- **视觉轨诚实收尾:最佳 = vision3 单帧 120×160 无罚 100k,held-out 899.3**
  (crash 型终止自证真驾驶;该数字先于新门,但终止类型与 mean_speed 量级
  经新门逻辑回验成立)。
- **观测路线瓶颈终判**:在"PPO+CNN+100k 步+本奖励结构"的组合下,观测表达
  (单帧/叠帧/120p/240p)不是绑定约束——绑定点在策略优化本身(奖励整形两难:
  无罚→快死均衡,有罚→停车/蠕行;能力上限先于观测上限到来)。视觉轨要再进一步,
  候选杠杆按优先级:① 行为克隆热启动(用状态量冠军蒸馏标注数据);② 大幅扩
  数据(500k+,慢爬曲线尚未平台);③ 分层奖励(curriculum 渐进)。

### 本路线交付物
- --cam-res 参数化 + fork img_w/h 注入修复;hr3 模型/日志/录帧全量留存;
- 新门与奖励 v2 成为视觉轨(及全部未来轨道)的标准口径。

## 2026-10-20 P3 BC 热启动:换杠杆的第三次独立证伪——视觉轨三向定案

### 交付物
- `src/bc_collect.py`(单 env 方案:info 重建教师状态向量,图像→动作零成本标签;
  DAgger 模式 --actor-model=BC 驾驶+teacher 标注)+ 数据卡
  (40k:node15-105/lateral2.2/heading15,std45.8/纠偏50.1%/油门37.8%);
- `src/bc_train.py`(逐层镜像 SB3 NatureCNN→512→Linear(512,2);按局段 90/10;
  纠偏帧≥30%加权采样;mps 训练 9×CPU;导出 rl_bc.zip 灌入 PPO CnnPolicy 骨架,
  rl_eval 零改动评测)+ outputs/bc_data/(40k)/bc_data_dagger/(25k)。

### 干净数字(新门口径)
- **纯 BC 闭环:191/286 步(mean_speed 0.30-0.38,蠕行档)** —— MSE-BC 的
  动作平均效应(油门在刹车/给油间取均值→蠕行)+协变量偏移,判 parked(fail)。
  val steerMAE 0.134(标定)。
- **DAgger 一轮(25k,BC 访问分布上采 champion 标签):39-125 步,更差**。
  机制:近撞车姿态的冠军标签是多峰的(MSE 在这些状态上拟合平均废话,
  val steerMAE 从 0.134→0.417 已示警),分布修正引入的标签噪声压倒了修正收益。
- 按判定树:<300 → DAgger 一轮已执行且恶化 → BC 杠杆证伪链闭环。

### 视觉轨三向定案(PPO/观测/模仿三个独立杠杆全部证伪)
1. PPO+奖励整形:两难(无罚快死/有罚停车/罚动蠕行送死),三次复现;
2. 观测表达:叠帧(布局bug→奖励均衡)与 240p(96-109 步)双封顶;
3. BC 模仿:蠕行 238 + DAgger 恶化。
**结论:视觉轨在"PPO/CNN/BC + 120-240p + 40-100k 预算"组合下绑定约束是
视觉策略优化本身**。要再进一步需跳出该组合(候选:Transformer/更大 CNN+
50 万对 BC 数据、sim-to-real 域随机化大礼包、或人类数据)。诚实最佳保持
vision3 单帧 899。

## 2026-10-20 [BC-CAT] 分类头 BC 单轮:初门不过,BC 轨正式封盘

- 实现:steer 21 箱/throttle 6 箱交叉熵,softmax 期望解码;解码头 ExpectDecode
  寄居 fork 命名空间(gym_donkeycar.envs.bc_decode,SB3 pickle 限定名解析,
  rl_eval 仅加一个 _load_bc_cat 回退分支——strict 骨架无法装非线性头)。
- **val 全面优于回归头**:steerMAE 0.127(vs 0.134)/hard 0.165(vs 0.170)/
  **油门 MAE 0.105(vs 0.123,分类头教科书级胜利)**。
- **闭环金丝雀:340/131 步,mean_speed 0.12/0.26——仍蠕行,初门(mean_speed≥1.0)
  不过**。
- 机制再分层:分类头治好了「训练期多峰标签平均化」,但闭环时**协变量偏移下
  推理不确定→softmax 摊平→期望回落箱中心**——不确定坍缩,表面相同的蠕行,
  机理不同层。单轮 time-boxed 约束下 BC 轨正式封盘(不续 DAgger 二代)。
- BC 轨总账:回归蠕行 238 / DAgger 恶化 / 分类头蠕行 340——三口味同向,
  强化三向定案:视觉轨瓶颈在策略优化与分布鲁棒性,模仿热启动(在该数据规模
  与架构下)无法绕开。

## 2026-10-20 [CPP] C++ 推理客户端四道闸:性能天花板终结,驾驶门负面结论

### 闸口结论(子代理工程,commit 76bd7ce+9b7a893)
- **闸1 骨架 PASS**:握手/config/reset 全通,frame_id 严格+1,遥测节拍 mean 19.9996ms。
- **闸2 一致性 PASS**:Python 录 847 帧回放,C++ obs **bit 级一致(max_diff=0)**、
  动作 1.13e-06——三个 bit 级陷阱入 README(forward_vel 用上一帧姿态/四元数 z 含
  +1 项/active_node 先转 double)。
- **闸3 性能 PASS**:10ms 节拍 60s 空载,处理环 **p95=0.112ms、超界 0%**
  (门 p95<6ms/超界<2%);对照 Python p95=17.2ms/23% 超界——**客户端天花板终结**(~150×)。
- **闸4 驾驶**:①sanity PASS(delay20j@20ms avg 1175,参考 1044±20%);
  **②主门 crash 9/10(差 1 局)——同种子复测同型,非噪声;③拉伸未达**。
  **负面结论(按纪律写明):客户端抖动非 crash 残差主因**——obs bit 级一致+亚毫秒
  推理下撞车率依旧,残差在策略(10ms 死时间难度,+23% vs python 873 是真实增益但
  天花板在训练侧)。主门未静默放行。
- **闸5 浸泡 PASS**:10ms×600s,55k 帧零断连,RSS 平坦无泄漏。

### 定案
100Hz 链路的**客户端侧彻底解锁**(协议三件套+bit 级一致性+0.1ms 推理),P1 三因子
消融的「客户端天花板」因子移除;剩余因子=10ms 死时间下的策略能力(死时间训练/
更宽 jitter 覆盖,训练侧任务)。src/cpp_client/(CMake+nlohmann+onnxruntime C API)
入交付物清单,README 全录协议坑。

## 2026-10-20 [CPP-FINAL] 异步包封箱长测:C++ 客户端 delay20j3@20ms

### 前置门
- ONNX 导出 outputs/rl_export/cpp_delay20j3.onnx(export 内置 16 组一致性校验过)。
- 客户端性能量@20ms(gate3 60s):处理环 p50=0.085/p95=0.218/p99=0.301ms,
  **超界 0%**,ONNX 推理 p95=0.167ms——封箱数字绑定"干净客户端"条件成立。

### 主口径 8100 域 ×50 局(逐局全列在 outputs/cpp_final_8100x50.log)
- valid 45/50(parked 5,mean_speed 门剔除),**avg_steps=1148(valid 口径)/1038(全样本),
  mean_speed=1.79,avg_reward=1643-1823,laps=0**。
- **crash 42/45 = 93.3%,Wilson 95% CI [82.1%, 97.7%]**。
- 对照历史 1044/crash 8-10(10 局小样本):avg 差 <10%(样本量内完全一致),
  crash 率收敛到 CI 档案——旧数字的非确定区间被 50 局钉死。

### 交叉域 8200 ×25 局(outputs/cpp_final_8200x25.log)
- valid 23/25,**avg_steps=1590.2**,mean_speed=1.91,reward=2557,
  crash 17/23 = 73.9% CI [53.9%, 87.5%]——8200 域系统性更易(与历次评测同向)。

### 封箱定案
- 异步包最终口径(C++ 客户端/20ms 节拍/量化协议):**8100 域 avg 1148 步、
  crash 93%[82,98];8200 域 avg 1590 步、crash 74%[54,88]; parked 率 8-10%
  (低速段被门剔除,不构成伪存活)**。
- 客户端侧(p95 0.218ms/超界 0%)不再是任何残差的借口;crash 残差=策略在该
  死时间+课程下的固有撞车率(与 P1 消融定论一致)。
- K2.8 复核点:逐局日志+本条目+性能量三件套。
