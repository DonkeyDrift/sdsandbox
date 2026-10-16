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

## 关键代码地图

- 车控：`sdsim/Assets/Scripts/Car.cs`——Rigidbody + 4 WheelCollider；`Set(pos, rot)` + `KeepSetting` 协程是物理瞬移的标准入口（TCP `reset_car` 也走它）。
- 赛道中央线：`PathManager`（各场景实例用序列化覆盖选路径来源，如 mountain_track 用 `doLoadPointPath` + `Resources/Track/large_loop_center.txt`）+ `CarPath`（`GetClosestSpanIndex(pos)` 找最近节点，`PathNode.rotation` 即该点切线朝向，可用来「回位到赛道中央并摆正」）。
- 防卡护栏 / 翻车自动恢复 / R 键回位：`sdsim/Assets/Scripts/CarRecovery.cs`——`RuntimeInitializeOnLoadMethod` 自举 + `DontDestroyOnLoad`，阈值均为 public 可调（`stuckTime` 2.5s、`flipGraceTime` 1.5s、`liftHeight` 0.4m 等）。
- 车辆生成：`CarSpawner.cs`（`cars` 列表；本地玩家车是 `cars[0]`；无 TCP 客户端时由 menu 的 Manual Drive/pace car 流程生成）。
