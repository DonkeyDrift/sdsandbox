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

- 命令:`python3 src/rl_train.py --resume <模型> --run-name <名> --envs 4 --steps <目标计数>`(续训计数累计);
  评测:`python3 src/rl_eval.py --model <m> --seed-base 5000 --episodes 10 --max-steps 2500 --json --launch-sim --port 9301 --sync --time-step 0.05`。
- **验收用同步模式**:异步 20Hz 评测会惩罚高增益策略(救车动作失稳),同步 0.05s 才是训练执行语义。
- **PhysX 多线程非确定**:同模型同种子局级结果有位级方差,验收按多次/长局口径判定。
- 铁律:训练运行期间清理 sim 只按端口 `lsof -ti:<ports> | xargs kill`,**严禁 `pkill -f donkey_sim`**(两次误杀训练);
  补丁后必须单环境实跑验证(py_compile 不够);`time/fps >600` 或 ep_rew 冻结=假步进,立即查 sim 存活。
- PPO 续训已收敛模型默认 lr 即可;真正的杠杆是**出生课程**(lateral 1.8±种子随机/heading 12,
  救车能力)与堆里程;评测上限 2000 会被饱和型策略打满,需结合 held-out 种子测。
- 视觉训练:`--vision --async-mode --eval-freq 20000`(同步模式的 AsyncGPUReadback 在离屏/遮挡下挂起;
  异步评测 3×2000 步一次约 7 分钟,频率必须调低);resume 路径不带 .zip 后缀(SB3 会自动补)。
- 多训练并行必须 `--run-name` 隔离(best/ckpt 目录),端口分段 9101+ / 9201+。
