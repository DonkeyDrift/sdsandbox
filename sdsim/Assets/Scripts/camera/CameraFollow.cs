using UnityEngine;
using System.Collections;

public class CameraFollow : MonoBehaviour
{

    public Transform target;

    public float approachPosRate = 0.1f;
    public float approachRotRate = 0.05f;

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
