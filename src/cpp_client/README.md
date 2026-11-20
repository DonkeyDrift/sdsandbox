# DonkeySim C++ 推理客户端(macOS arm64)

量化异步控制协议(TcpCarHandler `ctl_step_ms` 相位协议)的 C++ 推理客户端。
控制环:收遥测(阻塞等下一帧)→ 构造 8 维 obs → ONNX Runtime 推理 → 发 control。

## 构建

依赖(brew):`cmake nlohmann-json onnxruntime`

```bash
cmake -S src/cpp_client -B build_cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build_cpp -j
```

产物 `build_cpp/dkcpp`。子命令:`gate1` 骨架探针 / `gate2` 一致性回放 /
`gate3` 性能门 / `gate4` 驾驶验收 / `gate5` 浸泡。

## 前置

- sim:`build/donkey_sim.app/Contents/MacOS/donkey_sim --rl --port 9301`(窗口模式,
  物理实时;headless 异步会掉到 ~1fps,不可用)。
- 模型:`python3 src/export_onnx.py --model <冠军.zip> --out outputs/rl_export/<名>.onnx`
  (导出时已对 16 组随机 obs 验证 ONNX reference == SB3 predict,1e-5)。
- 清理 sim 只按端口:`lsof -ti:9301,9302 | xargs kill`,**严禁 pkill -f donkey_sim**。

## obs 构造(与 fork `gym_donkeycar/envs/donkey_sim.py` 逐字段对齐)

`[cte/2, speed/10, forward_vel/10, yaw/180, gyro_z/90, active_node/max(total_nodes,1), 0, 0]`

三个必须逐比特镜像的坑(一致性门实测踩出,详见下「协议坑」):

1. **forward_vel 用上一帧的 roll/pitch/yaw 和 vel**:`on_telemetry` 先算
   forward_vel 再更新 vel/姿态——第一帧恒 0。
2. **quaternion rotate 的 +v 项**:`rotate_vec(q,[0,0,1])` 的 z 分量是
   `1 + uvz*2q3 + uuvz*2`,不是 `0 + ...`(漏掉 v[2]=1 会得到一个 z≈0 的错误
   forward,低速时 obs[2] 差一个量级)。
3. **active_node 归一化是浮点除法**:`an/max(tn,1)`,C++ 里两个 int 相除会先整除
   恒得 0。

## 协议坑与解法(血泪记录)

- **control 消息三个值必须发字符串**:`OnControlsRecv` 用
  `float.Parse(json["steering"].str)`——发 JSON 数字则 `.str` 为 null,整条
  control 被 try/catch 吞掉**静默失效**(车不受控,无任何报错)。
- **cam_config/car_config 字段也必须字符串**:sim 端 `.str` 解析,发数字会在
  协程里抛异常,相机配置不生效(回退默认 JPG,遥测内嵌大图拖慢节拍)。
- **帧定界**:`{...}` 定界、`\n` 可选;TCP 任意分段/粘包都要能切(实现见
  `sim_client.hpp`,与 sim 端 `JsonTcpClient.ExtractJsonFromStream` 同规则)。
- **量化遥测节拍 = ctl_step_ms**:sim 在物理网格边界先 SendTelemetry 再应用
  缓存的 control——观测里上一动作已完整持有一个步长(协议固有 1 步死时间,
  与同步步进同构)。客户端收到帧→立即回 control 即自然对齐,无需主动 pacing。
- **收包要大缓冲 + TCP_NODELAY**:fork client 8MB RCVBUF 的教训(反压卡死
  sim 发送线程);Nagle 会叠 10ms+ 发送延迟。
- **reset 后首个完整步长才发遥测**(边界相位重对齐):reset 后别指望立刻有帧,
  等待即可;随机出生 seed 完全可复现(`System.Random(seed)`)。
- **RlMode 车断连不自动删**:每个客户端会话用全新 sim 实例(或接受赛道上可能
  留有上一会话的静止车)。
- **episode 判定在客户端**:`|cte|>max_cte`(前 5 步宽限仅挡 >2*max_cte)、
  `hit!="none"`、`flipped`、`stuck_time>2.5s`,与 `rl_eval.py` 口径一致。

## 验证闸结果

原始数字均落盘 `outputs/cpp_gate*.log`。

### 闸 1 骨架探针 — PASS(`outputs/cpp_gate1.log`)

- 连接/握手/car_config(ctl_step_ms=20)/cam_config(img_enc none)/reset_car(seed 8100)
  全部正常,遥测字段全 present(frame_id/collision_impulse/flipped/stuck_time/lap 等新字段齐)。
- frame_id 严格 +1 单调。
- 遥测到达间隔(ctl=20ms):mean=19.9996ms std=4.08 p50=16.73 p95=25.07。
- 客户端处理(收遥测→发 control):mean=0.0129ms p50=0.0123 p95=0.0183 p99=0.0246。

### 闸 2 一致性门 — PASS(`outputs/cpp_gate2.log`)

- 录制:fork 环境(ctl_step_ms=20, seed 8100 域, 3 局共 847 帧,含两段 400 步长驾驶+
  一段撞车),逐帧记录原始遥测 + python obs + python 动作(`outputs/cpp_replay.jsonl`)。
- C++ 离线回放:**max_obs_diff = 0**(847 帧逐点 bit 级一致),动作 max_diff=1.13e-06
  (ONNX Runtime vs SB3 predict,容差 1e-4)。
- 修复的两个 obs 构造 bug 见上「三个必须逐比特镜像的坑」:int 整除、quaternion +v 项。

### 闸 3 性能门 — PASS(`outputs/cpp_gate3.log`)

- ctl_step_ms=10 空载 60s,6001 帧(100Hz 满帧),含 ONNX 推理:
  - 处理环(收→obs→ONNX→发)p50=0.064ms **p95=0.112ms** p99=0.168ms
  - 超 10ms 界:**0%**(门:<2%)
  - ONNX 推理 p50=0.047ms p95=0.081ms
  - 遥测节拍 mean=9.9984ms
- 门 p95<6ms:0.112 ≪ 6 → **PASS**。
- 对照 Python 基线(20Hz):p95=17.2ms、23% 超 10ms 界 → C++ 客户端约 150× 收敛,
  客户端抖动从「残差主因候选」降级为可忽略。

### 闸 4 驾驶验收(`outputs/cpp_gate4_*.log`)

① sanity:delay20j@ctl=20ms,8100 域 ×10 局 ×2500 步(`cpp_gate4_sanity.log`):
valid 9/10(parked 1),**avg_steps=1175.3**(参考 avg~1044,±20% 域 835–1253 → **PASS**),
crashes=6/9,laps=0。

② 主门:delay10j@ctl=10ms(`cpp_gate4_main.log`):valid 10/10,
**avg_steps=1091.7 ≥ 1044 → 均值项 PASS**;crashes=**9/10**,门要求 ≤8/10 →
**crash 项差 1 局未达**。③ 拉伸 ≥1500 且 ≤5/10 → 未达。

同种子独立复测(`cpp_gate4_main_rerun.log`):avg_steps=1053.1,crashes=9/10——
两批一致,非单次噪声。判定见下「重标定说明」。

### 闸 5 浸泡 — PASS(`outputs/cpp_gate5.log`)

- 10ms 节拍连续 600s:55,094 帧、52 次自动 reset(撞车后自动换种子续驾)、
  **零断连**。
- RSS 26.80MB → 27.22MB(Δ+0.42MB 为启动预热,150s 后完全平坦)→ 无内存增长。

## 总结论

四道闸:① PASS、② PASS、③ PASS、④ 部分 PASS(sanity/均值项过,crash 项两批
一致 9/10 未达,理由见下)、⑤ PASS。**客户端天花板终结**:obs bit 级一致、
处理环亚毫秒、10 分钟零断连零泄漏。驾驶侧残差已不归客户端——负面结论按判定树
归档。

## 重标定说明(闸 4 主门 crash 一项)

两批独立复测(同 8100 域同种子、全新 sim 实例)crash 均 9/10、avg_steps 1053/1092:
**差 1 局越线是策略在该种子域的固有撞车率,不是客户端噪声或单局方差**——
- 客户端侧已无可整改项:obs 构造 bit 级一致(闸 2),处理环 p95 0.112ms(闸 3),
  sanity 在同一客户端上复现 20ms 参考(闸 4①);
- PhysX 多线程位级方差(AGENTS.md 已载)用两批复测排除:两批同为 9/10;
- 均值项两批均过 1044(1053/1092),相对 Python 基线参考(873)提升 ~+23%。

**结论(负面,按判定树归档):客户端抖动不是 crash 残差主因。**
把 20Hz/Python 客户端换成 10ms 量化+C++ 亚毫秒推理后,avg_steps +23%,
但 crash 率没有越过 8/10 门——残差在策略本身(10ms 死时间比 20ms 难一个量级,
见 rl_review 异步战线结论),不在客户端。记「客户端天花板终结」:**天花板已拆,
驾驶不背锅**。

(若要把主门推过线,杠杆在策略/课程侧——10ms 死时间专项训练或更宽 jitter
覆盖——不在工程客户端,超出本任务范围。)
