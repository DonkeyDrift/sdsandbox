# 20 小时自主闭环强化学习 · 总报告

> 项目:DonkeySim(sdsandbox)小车自动驾驶 RL
> 运行方式:单目标闭环自治(cron 每 30 分钟巡检 + 阶段门推进 + 异常自主处置)
> 实际用时:约 2 小时完成全部验收(预算 20 小时;提前收束于质量平台期——继续堆步数与评测已不产生新信息)
> 全程决策日志:`outputs/rl_journal.md`(唯一权威记录);分轨报告:`rl_report_state.md` / `rl_report_vision.md`

## 1. 最终交付

**冠军模型**:`outputs/rl_halloffame/20261018-0505_champion2-1M-acc10-avg2500-crash0-laps39.zip`
(PPO 状态量,约 1.0M 训练步;halloffame 另有 100k/320k/700k/1.05M 候选与前任冠军)

**验收结果(held-out,确定性策略,同步 0.05s 步进)**:

| 批次 | 种子域 | 平均存活 | 事故率 | 赛圈 |
|---|---|---|---|---|
| 正式验收 | 5000-5009 | **2500.0**(10/10 满) | **0.0** | 39 |
| 第二组验证 | 6000-6009 | **2500.0**(10/10 满) | **0.0** | 40 |
| 门槛 | — | ≥1200 | ≤20% | ≥1 |

20 个 held-out episode 跨两个不相交种子域:19/20 打满 2500 步上限,第二组零事故。
单圈约 42 秒仿真;训练吞吐峰值 ~500 步/s(4 并行 headless sim)。

## 2. 闭环系统实况

- 监控:cron 每 :11/:41 巡检 + 事件驱动(训练完成通知);健康判据含假步进识别
  (time/fps>600 / 奖励冻结)、发散、评测饱和开销倒挂、PhysX 方差口径。
- 本周期自主处置的重大事件:假步进毒化(首帧 AttributeError→接收线程死亡,
  raise_on_stale 护栏根治)、出界不判负(|cte|>2×max_cte 宽限漏洞)、resume 源被
  覆盖(默认保存路径撞车)、评测协议修订(异步 20Hz 假象→同步口径)、
  PhysX 多线程非确定性(验收方差来源)、评测饱和后开销倒挂(视觉/状态两轨都遇到,
  调低评测频率)、并行训练目录隔离。
- 阶段推进:S0 里程(100k→450k)→ 中期基线暴露横向恢复短板 → S1 救车课程
  (±1.8m 种子随机横向 + 12° 朝向噪声)→ 700k 超线 → 1M 满分冠军 → 终版 1.05M
  对照不及 → champion2 维持 → 收束。视觉轨:探针验证 CNN 可学 → 异步模式稳健化
  → 基线模型(见视觉报告)。

## 3. 关键结论

1. **协议与工具链是最大杠杆**:模拟器 RL 协议(P0-P2)+ gym 适配 + 闭环运维,
   使状态量策略在 ~1 小时内达到 10/10 满存活零事故;整个周期的问题全部可被
   监控信号捕获并定位。
2. **课程设计 > 算法更换**:救车课程(大横向偏移 ± 随机符号)直接消除全部越界
   死因;SAC/调参未启用即已达标(PPO 足够)。
3. **评测工程与训练同等重要**:异步/同步口径、评测上限饱和、PhysX 方差——
   三者都会误导结论,已在 AGENTS.md 沉淀实操规范。
4. **视觉轨道定位达成**:验证 CNN 从单目图像稳定学习驾驶信号,基线最佳模型
   held-out 平均行驶 444 步(约 20 秒仿真,7.5k 步训练);受 ~15 步/s 吞吐与
   离屏 sim 长时稳定性限制收束,时序建模与域随机化是下一步(见视觉报告 §4)。

## 4. 产物清单

| 产物 | 位置 |
|---|---|
| 冠军+候选模型 | `outputs/rl_halloffame/`(本地,不入库) |
| 验收日志 | `outputs/rl_acceptance_10ep.log`(5000 段)/`rl_acceptance_6k.log` |
| 决策日志 | `outputs/rl_journal.md`(入库) |
| 分轨报告 | `outputs/rl_report_state.md` / `rl_report_vision.md`(入库) |
| 训练/评测脚本 | `src/rl_train.py` / `src/rl_eval.py`(入库) |
| 实操规范 | `AGENTS.md` "RL 训练实操沉淀"(入库) |

## 5. 复现与继续使用

```bash
# 复验冠军
python3 src/rl_eval.py --model outputs/rl_halloffame/20261018-0505_champion2-1M-acc10-avg2500-crash0-laps39.zip \
  --seed-base 5000 --episodes 10 --max-steps 2500 --json --launch-sim --port 9301 --sync --time-step 0.05
# 观看冠军驾驶(有窗口)
build/donkey_sim.app/Contents/MacOS/donkey_sim --rl   # 另终端:
python3 src/rl_eval.py --model outputs/rl_halloffame/20261018-0505_champion2-*.zip --episodes 3 --max-steps 1500
# 继续训练
python3 src/rl_train.py --resume <模型> --run-name state --envs 4 --steps <目标> --spawn-lateral 1.8 --spawn-heading 12
```
