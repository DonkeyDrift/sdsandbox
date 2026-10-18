using UnityEngine;
using System.Collections;

public class CameraFollow : MonoBehaviour
{

    public Transform target;

    public float approachPosRate = 1.0f;
    // 朝向平滑时间常数(秒):位置零滞后硬挂载,但朝向低通滤波——
    // RL 策略的转向修正是高频抖动,朝向硬拷贝会让整个画面左右晃
    // (2026-10-18 用户体感);0.35s 足够滤掉抖动,过弯又能平滑跟上
    // (发夹弯 ~30°/s 时稳态滞后 ~10°,车仍保持在画面内)。
    public float rotSmoothTime = 0.35f;

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
    //    LateUpdate 保证在所有 Update(含物理步进)之后取最终车位,像素级锁定。
    void LateUpdate()
    {
        if (target == null)
            return;
        // 位置:零滞后硬挂载(前后窜动修复,不可再引入延迟)
        transform.position = target.position;
        // 朝向:指数低通(帧率无关),过滤转向抖动
        float a = 1f - Mathf.Exp(-Time.unscaledDeltaTime / Mathf.Max(0.01f, rotSmoothTime));
        transform.rotation = Quaternion.Slerp(transform.rotation, target.rotation, a);
    }
}
