using UnityEngine;
using System.Collections;

public class CameraFollow : MonoBehaviour
{

    public Transform target;

    public float approachPosRate = 0.5f;
    public float approachRotRate = 0.35f;

    void Awake()
    {
        // 跟随下限:同步步进(RL 模式)下物理按控制步突发推进,软跟随(原 0.1,
        // 滞后时间常数 ~0.2s ≈ 车速×0.2m 级摆动)会让车相对画面"前后窜动"。
        // 抬高下限使相机近似刚性挂载,车辆稳定在画面中心,世界随物理步进微跳
        // (配合刚体插值,体感远优于车辆摆动)。异步常规驾驶同样受益。
        approachPosRate = Mathf.Max(approachPosRate, 0.5f);
        approachRotRate = Mathf.Max(approachRotRate, 0.35f);
    }

    // 必须挂在 Update 且用 unscaledDeltaTime:同步步进(timeScale=0)下 FixedUpdate
    // 不执行、Time.deltaTime 恒 0,相机会永久冻结在出生位置(2026-10-18 实测);
    // unscaled 时间保证两种模式下都逐帧跟随。
    void Update()
    {
        if (target != null)
        {
            float dtRate = (Time.unscaledDeltaTime / 0.02f);
            transform.position = Vector3.Lerp(transform.position, target.position, approachPosRate * dtRate);
            transform.rotation = Quaternion.Lerp(transform.rotation, target.rotation, approachRotRate * dtRate);
        }
    }
}
