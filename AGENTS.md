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
- RL 协议 smoke 测试：`python3 src/rl_smoke_test.py`（需先起 app；14 项断言覆盖遥测新字段/time_scale/随机出生/seed 确定性/deadman）。

## RL 训练协议（2026-10-17 新增）

- 入站帧解析（`tcp/JsonTcpClient.cs`）：按 `'{...}'` 提取对象、可选 `'\n'` 终止符，**两种客户端写法都合法**——纯 `'}` 拼接（gym-donkeycar 风格，值必须字符串化）或 `'\n'` 分隔（新客户端可发未加引号数字）；修复前 `'\n'` 会永久卡死该连接的入站消息。
- `car_config` 新字段：`rl_mode`(0/1)、`control_timeout_ms`(deadman，0=禁用)；CLI `--rl` 全局开关、`--idle_timeout <秒>`(0=禁用)。RlMode 车：不自动删车、CarRecovery 只上报不自动回位；新遥测字段 `frame_id/collision_impulse/flipped/stuck_time/lap` 始终发送。
- `reset_car` 新可选字段：`node_index/node_min/node_max/lateral_offset/heading_noise/seed`（带任一即随机出生到赛道节点，`System.Random(seed)` 可复现；全缺省保持原回出生点行为）。
- 新消息 `time_scale`（0.1~4，异步模式直接加速仿真）。

## 关键代码地图

- 车控：`sdsim/Assets/Scripts/Car.cs`——Rigidbody + 4 WheelCollider；`Set(pos, rot)` + `KeepSetting` 协程是物理瞬移的标准入口（TCP `reset_car` 也走它）；暴露 `IsFlipped`/`StuckTime`/`LastCollisionImpulse` 供遥测上报。
- 赛道中央线：`PathManager`（各场景实例用序列化覆盖选路径来源，如 mountain_track 用 `doLoadPointPath` + `Resources/Track/large_loop_center.txt`）+ `CarPath`（`GetClosestSpanIndex(pos)` 找最近节点，`PathNode.rotation` 即该点切线朝向，可用来「回位到赛道中央并摆正」）。
- 防卡护栏 / 翻车自动恢复 / R 键回位：`sdsim/Assets/Scripts/CarRecovery.cs`——`RuntimeInitializeOnLoadMethod` 自举 + `DontDestroyOnLoad`，阈值均为 public 可调（`stuckTime` 2.5s、`flipGraceTime` 1.5s、`liftHeight` 0.4m 等）；RL 车（`TcpCarHandler.RlMode`）跳过自动恢复，由遥测字段上报状态。
- 车辆生成：`CarSpawner.cs`（`cars` 列表；本地玩家车是 `cars[0]`；无 TCP 客户端时由 menu 的 Manual Drive/pace car 流程生成）。
- TCP 业务 handler：`sdsim/Assets/Scripts/tcp/TcpCarHandler.cs`——消息注册/遥测组包（快路径零分配 TxAscii + 回退 JSONObject 双实现，加字段要同步两处）；`NumField()` 助手兼容字符串/数字两种 JSON 取值（NUMBER 型 `.str` 为 null，直接 `.str` 会炸）。
