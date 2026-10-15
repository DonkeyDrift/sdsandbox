# DonkeySim macOS（DonkeyDrift 60Hz 版）

基于 [sdsandbox](https://github.com/DonkeyDrift/sdsandbox) 分支 `feature/telemetry-fps-60`
构建的 Donkey Car Unity 模拟器。Universal 二进制（Apple Silicon 原生 + Intel），
macOS 13+ 通用。

## 本版相对官方版的变化

| 特性 | 说明 |
|---|---|
| **60Hz 遥测推流** | `cam_config` 新增可选 `fps` 键（默认 20 不变，向后兼容）。客户端发 `"fps": 60` 即真实 60fps 图像流，无插帧 |
| **异步采集管线** | AsyncGPUReadback + 工作线程 JPEG 编码，主线程采包 ~0.5ms，遥测天花板 ~58Hz 消除 |
| **命令行开关** | `--host` `--port` `--telemetry_fps N` `--quality N`（0=最低画质）`--vsync N` |
| **自检仪表** | 日志每 5s 输出 `[telemetry-stats] send_fps/render_fps/capture_ms/sent/tx_drop` |
| **修复** | 加速度遥测 fixedDeltaTime 计算错误 |

## 安装

1. 打开 DMG，把 `donkey_sim.app` 拖进 Applications（随便放哪也行）。
2. **首次运行必做**（ad-hoc 签名，未公证）：终端执行
   ```bash
   xattr -cr /Applications/donkey_sim.app
   ```
   然后正常双击打开。不执行会弹“应用已损坏”。
3. 模拟器默认监听 `0.0.0.0:9091`，菜单界面保持开着即可。

## 客户端使用（gym-donkeycar）

```python
conf = {
    "exe_path": "remote",          # 模拟器已手动启动
    "host": "<模拟器所在机器 IP>",
    "port": 9091,
    "cam_config": {                 # 请求 60Hz 推流；旧版模拟器忽略此键
        "img_w": 640, "img_h": 480, "img_d": 3, "img_enc": "JPG",
        "fov": 90, "offset_x": 0.0, "offset_y": 1.2, "offset_z": 0.0,
        "rot_x": 0.0, "fps": 60,
    },
}
```
配合 DonkeyDrift 的 gym-donkeycar fork（含 8MB 接收缓冲 + cv2 解码加固）体验最佳；
官方 gym-donkeycar 客户端同样兼容（fps 键被安全忽略）。

## 已知问题

- **偶发启动崩溃**（约 1/4 概率，SIGSEGV，见 issue #010）：直接再启动一次即可；
  不影响运行稳定性。无人值守场景建议加启动重试。
- 首次启动后若通过 gym 客户端连车，菜单界面会自动进入 generated_track 场景。

## 构建信息

- Unity 2022.3.49f1 · macOS Universal · ad-hoc 签名（未公证）
- 源码：`github.com/DonkeyDrift/sdsandbox` @ `feature/telemetry-fps-60`
- 构建命令：
  ```bash
  Unity -batchmode -projectPath sdsim -executeMethod BuildCommand.Build \
        -buildTargetStandaloneOSX <输出路径>.app -quit
  ```
