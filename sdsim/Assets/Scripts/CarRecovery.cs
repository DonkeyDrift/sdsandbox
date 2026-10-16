using UnityEngine;

// 防卡护栏 + 翻车自动恢复 + R 键快速回位。
// 运行时自举创建,不依赖场景/prefab 配置,所有场景生效。
public class CarRecovery : MonoBehaviour
{
    [Header("R 键回位")]
    public KeyCode resetKey = KeyCode.R;
    public float manualResetCooldown = 1f;

    [Header("卡顿自动恢复")]
    public float stuckTime = 2.5f;
    public float minStuckSpeed = 0.5f;
    public float stuckThrottle = 0.1f;

    [Header("翻车自动恢复")]
    public float flipGraceTime = 1.5f;
    public float flipUpDot = 0.2f;

    [Header("通用")]
    public float autoResetCooldown = 6f;
    public float liftHeight = 0.4f;

    float lastManualResetTime = -999f;
    float lastAutoResetTime = -999f;
    float stuckTimer = 0f;
    float flipTimer = 0f;

    Car playerCar;
    CarSpawner spawner;
    PathManager pathManager;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void Bootstrap()
    {
        GameObject go = new GameObject("CarRecovery");
        go.AddComponent<CarRecovery>();
    }

    Car GetPlayerCar()
    {
        if (playerCar == null)
        {
            if (spawner == null)
                spawner = FindObjectOfType<CarSpawner>();

            if (spawner != null && spawner.cars.Count > 0 && spawner.cars[0] != null)
                playerCar = spawner.cars[0].GetComponent<Car>();
            else
                playerCar = FindObjectOfType<Car>();
        }

        return playerCar;
    }

    PathManager GetPathManager()
    {
        if (pathManager == null)
            pathManager = FindObjectOfType<PathManager>();

        return pathManager;
    }

    void Update()
    {
        if (resetKey == KeyCode.None)
            return;

        if (Input.GetKeyDown(resetKey) && Time.time - lastManualResetTime >= manualResetCooldown)
        {
            Car car = GetPlayerCar();
            if (car != null)
            {
                lastManualResetTime = Time.time;
                ResetToNearestPathNode(car, "manual key");
            }
        }
    }

    void FixedUpdate()
    {
        Car car = GetPlayerCar();
        if (car == null || car.rb == null || car.rb.isKinematic)
        {
            stuckTimer = 0f;
            flipTimer = 0f;
            return;
        }

        // 卡护栏:有油门意图但车几乎不动;松油门即重置计时,正常停车不误伤
        bool wantsMove = Mathf.Abs(car.requestTorque) > stuckThrottle;
        bool barelyMoving = car.rb.velocity.magnitude < minStuckSpeed;
        if (wantsMove && barelyMoving)
            stuckTimer += Time.fixedDeltaTime;
        else
            stuckTimer = 0f;

        // 翻车:车顶明显不朝上
        if (Vector3.Dot(car.transform.up, Vector3.up) < flipUpDot)
            flipTimer += Time.fixedDeltaTime;
        else
            flipTimer = 0f;

        if (Time.time - lastAutoResetTime < autoResetCooldown)
            return;

        if (stuckTimer >= stuckTime || flipTimer >= flipGraceTime)
        {
            lastAutoResetTime = Time.time;
            stuckTimer = 0f;
            flipTimer = 0f;
            ResetToNearestPathNode(car, "auto recovery");
        }
    }

    void ResetToNearestPathNode(Car car, string reason)
    {
        PathManager pm = GetPathManager();
        bool usePath = pm != null && pm.carPath != null && pm.carPath.nodes.Count > 0;

        Vector3 pos;
        Quaternion rot;

        if (usePath)
        {
            int iClosest = pm.carPath.GetClosestSpanIndex(car.transform.position);
            PathNode node = pm.carPath.GetNode(iClosest);
            pos = node.pos + Vector3.up * liftHeight;
            rot = node.rotation;
        }
        else
        {
            pos = car.startPos + Vector3.up * liftHeight;
            rot = car.startRot;
        }

        // 清零运动状态;Set 内部会先置 isKinematic,必须在这之前做
        car.rb.velocity = Vector3.zero;
        car.rb.angularVelocity = Vector3.zero;
        car.RequestThrottle(0f);
        car.RequestSteering(0f);
        car.RequestFootBrake(0f);

        car.Set(pos, rot);

        Debug.Log(string.Format("[CarRecovery] {0}: reset car to {1} via {2}",
            reason, pos, usePath ? "nearest path node" : "spawn point"));
    }
}
