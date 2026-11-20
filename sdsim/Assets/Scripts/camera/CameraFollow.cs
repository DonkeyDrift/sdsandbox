using UnityEngine;
using System.Collections;

public class CameraFollow : MonoBehaviour
{

    public Transform target;

    public float approachPosRate = 1.0f;
    // 朝向平滑时间常数(秒):对 yaw 低通滤波——
    // RL 策略的转向修正是高频抖动,朝向硬拷贝会让整个画面左右晃
    // (2026-10-18 用户体感);0.35s 足够滤掉抖动,过弯又能平滑跟上
    // (发夹弯 ~30°/s 时稳态滞后 ~10°,车仍保持在画面内)。
    public float rotSmoothTime = 0.35f;
    // 横向位置平滑时间常数(秒,2026-10-20 无人机视角稳定化):
    // 锚点 CameraFollowTm 是车体刚性子节点(车后 4m),车身左右平移与偏航
    // 抖动(经 4m 杠杆)会 1:1 横移相机、整个世界横屏。横向分量走低通后,
    // 车可在画面内小幅左右滑,世界背景锁定;0.25s 对 ~50-100ms 抖动衰减
    // 一个量级,发夹弯 30°/s 稳态横向滞后 ~0.5m,车不出画面。
    public float latSmoothTime = 0.25f;
    // 瞬移吸附阈值(米):reset/换绑车位跳变超过此距离时硬设位姿,避免低通拖尾
    public float snapDistance = 8f;

    float smoothedYaw;
    float pitch0;
    Transform lastTarget;

    void Awake()
    {
        // 跟随下限:同步步进(RL 模式)下物理按控制步突发推进,软跟随(原 0.1,
        // 滞后时间常数 ~0.2s ≈ 车速×0.2m 级摆动)会让车相对画面"前后窜动"。
        // 1.0 = 硬挂载:车像素级锁定画面中心,世界随物理步进微跳
        // (同步语义的诚实呈现,远优于车辆相对画面摆动)。
        approachPosRate = Mathf.Max(approachPosRate, 1.0f);
    }

    // 必须挂在 LateUpdate 且用 unscaledDeltaTime:
    // 1) 同步步进(timeScale=0)下 FixedUpdate 不执行、Time.deltaTime 恒 0,
    //    相机会永久冻结在出生位置(2026-10-18 实测);
    // 2) Update 阶段 TcpCarHandler 才调用 Physics.Simulate 推进车辆;
    //    相机若在 Update 跟随,读到的是本帧 Simulate 之前的车位——有子步的帧
    //    车相对镜头前移一个子步、无子步的帧归位,交替成高频前后抖动。
    //    LateUpdate 保证在所有 Update(含物理步进)之后取最终车位。
    void LateUpdate()
    {
        if (target == null)
        {
            lastTarget = null;
            return;
        }

        // 绑定/换绑首帧:吸附到锚点,记录固定俯角与初始 yaw
        if (target != lastTarget)
        {
            pitch0 = target.localEulerAngles.x;
            smoothedYaw = target.eulerAngles.y;
            transform.SetPositionAndRotation(target.position, Quaternion.Euler(pitch0, smoothedYaw, 0f));
            lastTarget = target;
            return;
        }

        float dt = Time.unscaledDeltaTime;

        // 朝向:yaw 低通 + roll 锁定(地平线恒水平,悬挂侧倾不漏进画面);
        // 俯角取锚点 prefab 的固定值,车身俯仰同样被滤掉
        float targetYaw = target.eulerAngles.y;
        float aRot = 1f - Mathf.Exp(-dt / Mathf.Max(0.01f, rotSmoothTime));
        smoothedYaw += Mathf.DeltaAngle(smoothedYaw, targetYaw) * aRot;
        Quaternion yawRot = Quaternion.Euler(0f, smoothedYaw, 0f);

        Vector3 delta = target.position - transform.position;
        if (delta.magnitude > snapDistance)
        {
            // 瞬移(reset/随机出生):位姿硬设,低通不拖尾
            smoothedYaw = targetYaw;
            yawRot = Quaternion.Euler(0f, smoothedYaw, 0f);
            transform.SetPositionAndRotation(target.position, Quaternion.Euler(pitch0, smoothedYaw, 0f));
            return;
        }

        // 位置:平滑 yaw 系内分轴——纵向/垂直硬跟随(保住"无前后窜动"修复),
        // 仅横向低通,与车的左右摆动解耦
        Vector3 local = Quaternion.Inverse(yawRot) * delta;
        float aLat = 1f - Mathf.Exp(-dt / Mathf.Max(0.01f, latSmoothTime));
        transform.position += yawRot * new Vector3(local.x * aLat, local.y, local.z);
        transform.rotation = Quaternion.Euler(pitch0, smoothedYaw, 0f);
    }
}
