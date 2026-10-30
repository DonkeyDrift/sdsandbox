# sdsandbox（DonkeySim）项目协作指引

## 工程结构

- Unity 工程在 `sdsim/`（版本固定 2022.3.49f1，见 `sdsim/ProjectSettings/ProjectVersion.txt`），游戏代码在 `sdsim/Assets/Scripts/`。
- Python 客户端在 `src/`（TCP 9091 控制 / 9092 私网 API）。
- 构建产物 `build/`、分发 `dist/` 均被 gitignore，不进仓库。

## 构建与打包（macOS，Unity 路径 `/Applications/Unity/Hub/Editor/2022.3.49f1/Unity.app/Contents/MacOS/Unity`）

- 编译检查：`-batchmode -projectPath sdsim -quit -logFile build/unity-compile-check.log`（增量约 10s，看日志无 `error CS`）。
- 出 app：加 `-executeMethod BuildCommand.Build -buildTargetStandaloneOSX <绝对路径>/build/donkey_sim.app`（`sdsim/Assets/Editor/BuildCommand.cs`，自动编 EditorBuildSettings 里全部 11 个场景）。
- 打包 DMG：`./tools/package_macos.sh build/donkey_sim.app dist`（ad-hoc 深签 + sha256）。
- **注意**：batchmode 编译/构建通过只代表能编译，不代表运行时功能正常；功能验证看玩家日志或实机。

## 调试

- 玩家日志：`~/Library/Logs/DonkeyCar/donkey_sim/Player.log`（app 退出后完整 flush；排查「代码没生效」先 grep 组件日志行）。
- app 场景流程是 menu → 赛道，跨场景存活的运行时对象必须 `DontDestroyOnLoad`。
- 键盘驾驶走 CrossPlatformInput 虚拟轴（`JoystickCarControl.cs`）；硬编码快捷键（如 `Input.GetKeyDown`）需游戏窗口聚焦才收得到。
- RL 协议 smoke 测试：`python3 src/rl_smoke_test.py`（需先起 app；24 项断言覆盖遥测新字段/time_scale/随机出生/seed 确定性/deadman/同步步进/无图像模式）。

## RL 训练协议（2026-10-17 新增）

- 入站帧解析（`tcp/JsonTcpClient.cs`）：按 `'{...}'` 提取对象、可选 `'\n'` 终止符，**两种客户端写法都合法**——纯 `'}` 拼接（gym-donkeycar 风格，值必须字符串化）或 `'\n'` 分隔（新客户端可发未加引号数字）；修复前 `'\n'` 会永久卡死该连接的入站消息。
- `car_config` 新字段：`rl_mode`(0/1)、`control_timeout_ms`(deadman，0=禁用)；CLI `--rl` 全局开关、`--idle_timeout <秒>`(0=禁用)。RlMode 车：不自动删车、CarRecovery 只上报不自动回位、跳过观众分屏相机（多 TCP 车无上限，实测 8 车并发）；新遥测字段 `frame_id/collision_impulse/flipped/stuck_time/lap` 始终发送。
- `reset_car` 新可选字段：`node_index/node_min/node_max/lateral_offset/heading_noise/seed`（带任一即随机出生到赛道节点，`System.Random(seed)` 可复现；全缺省保持原回出生点行为）。
- 新消息 `time_scale`（0.1~4，异步模式直接加速仿真）。
- **同步步进**：`{"msg_type":"step_mode","step_mode":"synchronous","time_step":0.05}`——发一条 control 推进固定时长物理（k=time_step/0.01 次 `Physics.Simulate`）并回一帧（严格对齐，同 seed+同动作序列结果逐位一致，实测两次 rollout delta=0.00000）；reset 也有应答帧。**物理是实例全局的，单实例只连一个 RL 客户端**；退出/断线自动恢复全局物理。实测吞吐：有图 29 步/s、无图 ~120 步/s（8 车共存）。
- **无图像模式**：`cam_config` 的 `img_enc:"none"`——零渲染零编码，遥测省略 image 键（状态量 RL 用），切回 JPG 自动恢复。
- 底层教训：`Car.Set()` 的 KeepSetting 协程依赖 `WaitForFixedUpdate`，**`Physics.Simulate` 不驱动它**，同步模式瞬移必须走 `Car.SetImmediate`（运动学沙盒内写 transform——kinematic 态 transform 才是权威）；掉线检测计时用 `Time.unscaledDeltaTime`（deltaTime 在 timeScale=0 冻结）；**同一 socket 所有发送必须走池化队列单写者**（同步 Send 与异步 BeginSend 并发会内核级字节交错，客户端收到撕裂 JSON）。

## 关键代码地图

- 车控：`sdsim/Assets/Scripts/Car.cs`——Rigidbody + 4 WheelCollider；`Set(pos, rot)` + `KeepSetting` 协程是异步模式物理瞬移的入口；**同步模式用 `SetImmediate`**；`ApplyRequests()`/`SyncKinematics()` 供 FixedUpdate 与手动步进共用；暴露 `IsFlipped`/`StuckTime`/`LastCollisionImpulse` 供遥测上报。
- 赛道中央线：`PathManager`（各场景实例用序列化覆盖选路径来源，如 mountain_track 用 `doLoadPointPath` + `Resources/Track/large_loop_center.txt`）+ `CarPath`（`GetClosestSpanIndex(pos)` 全扫；带 hint 重载做 ±64 窗口局部搜索，`PathNode.rotation` 即该点切线朝向，可用来「回位到赛道中央并摆正」）。
- 防卡护栏 / 翻车自动恢复 / R 键回位：`sdsim/Assets/Scripts/CarRecovery.cs`——`RuntimeInitializeOnLoadMethod` 自举 + `DontDestroyOnLoad`，阈值均为 public 可调（`stuckTime` 2.5s、`flipGraceTime` 1.5s、`liftHeight` 0.4m 等）；RL 车（`TcpCarHandler.RlMode`）跳过自动恢复，由遥测字段上报状态。
- 车辆生成：`CarSpawner.cs`（`cars` 列表；本地玩家车是 `cars[0]`；无 TCP 客户端时由 menu 的 Manual Drive/pace car 流程生成；`--rl` 且批处理/无图形设备（训练实例）才清理观众分屏相机，带窗口的 `--rl` 保留跟车相机）。
- TCP 业务 handler：`sdsim/Assets/Scripts/tcp/TcpCarHandler.cs`——消息注册/遥测组包（快路径零分配 TxAscii + 回退 JSONObject 双实现，加字段要同步两处）；`NumField()` 助手兼容字符串/数字两种 JSON 取值（NUMBER 型 `.str` 为 null，直接 `.str` 会炸）；同步步进状态机（pendingSteps/SyncStep/SyncPostReset）在 Update 的 `!asynchronous` 分支。
- 发送层：`sdsim/Assets/Scripts/tcp/TcpClient.cs`——池化发送队列是唯一出口（CommitPooledSend/SendData 都进队）；`PooledSendCallback` 对 `EndSend` 部分发送补发剩余字节。
- 构建：`sdsim/Assets/Editor/BuildCommand.cs` 支持 `-buildTarget <StandaloneOSX|StandaloneLinux64|StandaloneWindows64>`（本机缺 Linux 模块时需先在 Hub 装 "Linux Build Support"）。

## RL 训练实操沉淀(2026-10-18 闭环实战)

- 命令:`python3 src/rl_train.py --resume <模型> --run-name <名> --envs 4 --steps N`;
  评测:`python3 src/rl_eval.py --model <m> --seed-base 5000 --episodes 10 --max-steps 2500 --json --launch-sim --port 9301 --sync --time-step 0.05`。
  评测加 `--record <目录>` 落现场(末 24 帧+蒙太奇+meta.json,供逐帧复核死因);视觉/录帧评测经 `--launch-sim` 时自动带离屏渲染(依赖 fork >= 1b05bae,此前 headless_keep_gfx 是死键会跑盲图)。
- 文档分工:`outputs/rl_journal.md` 是训练执行方的逐条操作日志;`outputs/rl_review.md` 是监控/解读方(K2.8)维护的**经核验定性结论史+方法论档案**(每条结论标核验等级,含战线终判/诚实门纪律/证据索引)——写新结论前先对齐 review,冲突以带证据者为准并在 review 留痕。
- **--steps 语义(2026-10-18 两次踩坑定案)**:续训时 SB3 内部目标 = `--steps` + 模型终身计数,
  即 **--steps 直接就是"新增步数"**(不是绝对累计)。日志里 total_timesteps 是终身计数,
  别把它当进度分母;进度 = total_timesteps - resume 点计数。
- **续训 lr 铁律(两次复发)**:从小模型/早期模型续训,PPO 默认 lr 3e-4 必过热发散
  (ep_len/ep_rew 数 rollout 内单调跌 >30% 即止损),一律 `--lr 1e-4` 起步。
- **异步架构重设计(2026-10-19 授权范围交付,修正上条"证伪"结论)**:StickyActionWrapper
  零迁移的根因之一是它"保持并丢弃"奇数步动作,**不是延迟**。真正打通的路径:
  ① 量化异步相位协议 `car_config ctl_step_ms`——控制应用+遥测对齐物理网格边界,
  FixedUpdate 驱动与渲染循环解耦(实测 50.0±0.9ms);量化模式复位必须走 SetImmediate
  与同步逐比特一致(KeepSetting 不落定轮接触状态,高增益策略会放大成失稳);手动
  Simulate 需临时关 Physics.autoSimulation。② headless 异步 1fps 根因=无渲染时 Unity
  主循环 ~3.3fps(实时物理跑成 1/3 速),`--keep-gfx` 离屏渲染即解。③ 协议固有 1 步
  死时间(s_k = post-a_{k-2}),`--action-delay N`(真·执行 N 步前动作)在同步内等价
  训练;必配 `--action-jitter K` 覆盖真实异步时钟抖动(客户端步周期 p95 可超边界 40%,
  控制丢帧/重复,纯延迟迁移只有 180/1900)。④ 多 env 异步训练 SubprocVecEnv 有死锁
  (单 env 正常、多 env 主进程阻塞管道读,未解)——同步等效训练快两个数量级,别硬解。
  ⑤ 过门候选立即 halloffame 快照(EvalCallback 同路径覆盖丢过 1816 步模型一次)。
  ⑥ 50ms 死时间比 20ms 难一个量级(异步 136 vs 1044),真实车控频率宜 >=50Hz。
- **Phase 1 结论(2026-10-18)**:lateral 2.2/heading 15 加强课程不换来 held-out 鲁棒性;
  7100 域长测与 7104 边缘出生可达 100%/10 连过,但出生边界 18-21 步越界是个体种子固有
  难度+PhysX 方差,严格门(如 8000 域 crash<=5%)两代模型都不达标——按多域口径综合判定。
- **验收用同步模式**:异步 20Hz 评测会惩罚高增益策略(救车动作失稳),同步 0.05s 才是训练执行语义。
- **PhysX 多线程非确定**:同模型同种子局级结果有位级方差,验收按多次/长局口径判定。
- 铁律:训练运行期间清理 sim 只按端口 `lsof -ti:<ports> | xargs kill`,**严禁 `pkill -f donkey_sim`**(两次误杀训练);
  补丁后必须单环境实跑验证(py_compile 不够);`time/fps >600` 或 ep_rew 冻结=假步进,立即查 sim 存活。
- PPO 续训已收敛模型默认 lr 即可;真正的杠杆是**出生课程**(lateral 1.8±种子随机/heading 12,
  救车能力)与堆里程;评测上限 2000 会被饱和型策略打满,需结合 held-out 种子测。
- 视觉训练:`--vision --async-mode --eval-freq 20000 --lr 1e-4`(同步模式的 AsyncGPUReadback 在离屏/遮挡下挂起;
  异步评测 3×2000 步一次约 7 分钟,频率必须调低);resume 路径不带 .zip 后缀(SB3 会自动补);
  已加回读超时兜底(CameraSensor,挂起 3s→回退同步 ReadPixels loud 记录),离屏长训不再冻死。
- 多训练并行必须 `--run-name` 隔离(best/ckpt 目录),端口分段 9101+ / 9201+。
- **模型交接约定(2026-10-30)**:RL 模型二进制(`outputs/rl_*.zip`、`rl_best_*`、`rl_ckpt_*`、
  `rl_halloffame/`、`rl_ppo_ckpt/`、`rl_export/`,约 5.8G)与 BC 数据集(`outputs/bc_data`、
  `bc_data_dagger`、`bc_data_probe.npz`、`bc_human_probe`、`bc_probe_montage.jpg`、
  `bc_human_data`(2026-10-30 落盘,mycar tub→npz 7884 对/1.0G),合计约 3.7G)**不进 git**
  (remote 为 GitHub,单文件 >100MB 拒收),
  但**必须随交接包(7z 全仓快照)流转**——重建包 = git 跟踪内容 + `.git` + `sdsim/*.csproj`+`sln` +
  上述模型/数据集;解包后 `--resume outputs/rl_<模型>`、`bc_train.py --data outputs/bc_data[_dagger]`
  直接可用,`rl_export/` 的 ONNX 供 C++ 侧部署。tb/日志/录屏(rec_*)/回放(rl_replay_*)/
  崩溃现场(*.crash)仍留本地不随包。
- 相机跟随在同步模式冻结的坑:`CameraFollow` 原为 FixedUpdate + deltaTime 缩放——timeScale=0 时
  FixedUpdate 不跑且 deltaTime 恒 0,相机永久停在出生位。已改 Update + unscaledDeltaTime。
- 观赏链路终版(2026-10-18 三轮攻坚):① 同步步进窗口模式摊匀——`TcpCarHandler.SyncStepSpread`
  把控制步的 k 个 `Simulate(0.01)` 按实时配速摊到连续渲染帧(位置逐帧连续;物理序列与突发版
  逐比特一致,batchmode 仍走突发保吞吐;SyncUpdate 的门必须是 `pendingSteps>0 || spreadRemaining>0`,
  否则摊匀半途被挡、最后一个控制步永远卡死);② 相机必须 `LateUpdate`(Update 阶段才 Simulate,
  Update 跟随会读到步进前车位,有/无子步帧交替成前后抖);③ 朝向必须低通(`rotSmoothTime` 0.35s),
  硬拷贝朝向会把策略高频转向修正 1:1 晃到画面上。
- **第三人称相机稳定化(2026-10-20,无人机视角)**:`CameraFollow` 位置不再整体硬挂载——锚点
  `CameraFollowTm` 是车体刚性子节点(车后 4m),车身左右平移+偏航抖动经杠杆 1:1 横移相机、
  世界横屏致眩晕。现改为平滑 yaw 系内**分轴滤波**:纵向/垂直硬跟随(保住无前后窜动修复),
  仅横向低通(`latSmoothTime` 0.25s),车可在画面内小幅左右滑、背景锁定;朝向 yaw 低通 +
  roll 锁定(`Quaternion.Euler(pitch0, smoothedYaw, 0)`,俯角取锚点 prefab 固定值,悬挂侧倾/
  车身俯仰不进画面);`snapDistance` 8m 或换绑 target 时位姿硬设,reset 瞬移不拖尾。
  车上 `CameraSensor`(策略视觉输入)不受影响。
- **headless 异步训练不可行**(2026-10-18 v2 实测):-batchmode -nographics 下异步观测流 ~1fps
  (疑依附相机渲染);异步路线只能有窗。tb 目录会被多次 run 复用(PPO_N 混写),分析前先按
  event 文件时间戳归属,勿整目录聚合误读。
- **rl_train 基建(2026-10-18 Phase 0 加固)**:learn() 已包 try/except——任何异常/中断自动
  存 `<save>.crash` 保命存档;CheckpointCallback 档位 `25000//envs`(vec-step 粒度);
  `tb_log_name=--run-name` 隔离 tb 目录。收尾关闭必须用 `close(timeout=10)` 再
  `close(terminate=True)` 兜底——裸 `envs.close()` 在中断落在同步 step_wait 时死锁
  (gymnasium 对 idle worker 的优雅 close 无超时);terminate 后 worker 的 sim 成孤儿,按端口清理。
- batchmode 构建报 `No ULF license found / Token not found in cache`:Unity Hub 没在运行,
  `open -a "Unity Hub"` 等 ~20s 刷新令牌后重试即可。
