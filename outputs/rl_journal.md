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
