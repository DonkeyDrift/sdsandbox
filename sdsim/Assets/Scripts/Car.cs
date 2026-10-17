using UnityEngine;
using System.Collections;


public class Car : MonoBehaviour, ICar{

	public CarSpawner carSpawner;
	public WheelCollider[] wheelColliders;
	public Transform[] wheelMeshes;

	public float maxSpeed = 30f;
	public float maxTorque = 50f;
	public float maxBreakTorque = 50f;
	public AnimationCurve torqueCurve;

	public Transform centrOfMass;

	public float requestTorque = 0f;
	public float requestBrake = 0f;
	public float requestSteering = 0f;

	public Vector3 acceleration = Vector3.zero;
	public Vector3 velocity = Vector3.zero;
	public Vector3 prevVel = Vector3.zero;

	public Vector3 startPos;
	public Quaternion startRot;
	private Quaternion rotation = Quaternion.identity;
	private Vector3 gyro = Vector3.zero;

	public Rigidbody rb;

	//for logging
	public float lastSteer = 0.0f;
	public float lastAccel = 0.0f;

	//when the car is doing multiple things, we sometimes want to sort out parts of the training
	//use this label to pull partial training samples from a run 
	public string activity = "keep_lane";

    public float maxSteer = 16.0f;

	//name of the last object we hit.
	public string last_collision = "none";
	public float last_collision_impulse = 0f;

	float stuckTimer = 0f;


	// Use this for initialization
	void Awake () 
	{
		rb = GetComponent<Rigidbody>();
		if (rb)
		{
			// 渲染插值:物理步进(尤其同步步进手动 Simulate 的突发推进)之间,
			// 渲染姿态在上/当前两帧物理姿态间平滑过渡——消除车辆"跳步感"。
			// 异步模式(timeScale=1)下等同常规游戏的惯用设置。
			rb.interpolation = RigidbodyInterpolation.Interpolate;
		}

		if(rb && centrOfMass)
		{
			rb.centerOfMass = centrOfMass.localPosition;
		}

		requestTorque = 0f;
		requestSteering = 0f;

		SavePosRot();
		
		// had to disable this because PID max steering was affecting the global max_steering
        // maxSteer = PlayerPrefs.GetFloat("max_steer", 16.0f);
	}

	public void SavePosRot()
	{
		startPos = transform.position;
		startRot = transform.rotation;
	}

	public void RestorePosRot()
	{
		Set(startPos, startRot);
	}

	// 同步步进专用:不依赖 WaitForFixedUpdate(Physics.Simulate 不驱动它),
	// 运动学沙盒内写 transform(运动学态以 transform 为准),再步进落定、释放回动力学。
	public void SetImmediate(Vector3 pos, Quaternion rot, float dt, int settleSteps)
	{
		rb.isKinematic = true;
		rb.position = pos;
		rb.rotation = rot;
		transform.position = pos;
		transform.rotation = rot;
		for (int i = 0; i < settleSteps; i++)
			Physics.Simulate(dt);
		rb.velocity = Vector3.zero;
		rb.angularVelocity = Vector3.zero;
		rb.isKinematic = false;
	}

	public void RequestThrottle(float val)
	{
		requestTorque = val;
		requestBrake = 0f;
		//Debug.Log("request throttle: " + val);
	}

    public void SetMaxSteering(float val)
    {
        maxSteer = val;
		// had to disable this because PID max steering was affecting the global max_steering
        // PlayerPrefs.SetFloat("max_steer", maxSteer);
        // PlayerPrefs.Save();
    }

    public float GetMaxSteering()
    {
        return maxSteer;
    }

	public void RequestSteering(float val)
	{
		requestSteering = Mathf.Clamp(val, -maxSteer, maxSteer);
		//Debug.Log("request steering: " + val);
	}

	public void Set(Vector3 pos, Quaternion rot)
	{
		rb.position = pos;
		rb.rotation = rot;

		//just setting it once doesn't seem to work. Try setting it multiple times..
		StartCoroutine(KeepSetting(pos, rot, 1));
	}

	IEnumerator KeepSetting(Vector3 pos, Quaternion rot, int numIter)
	{
		while(numIter > 0)
		{
			rb.isKinematic = true;
			
			yield return new WaitForFixedUpdate();

			rb.position = pos;
			rb.rotation = rot;
			transform.position = pos;
			transform.rotation = rot;

			numIter--;

			rb.isKinematic = false;
		}
	}

	public float GetSteering()
	{
		return requestSteering;
	}

	public float GetThrottle()
	{
		return requestTorque;
	}

	public float GetFootBrake()
	{
		return requestBrake;
	}

	public float GetHandBrake()
	{
		return 0.0f;
	}

	public Vector3 GetVelocity()
	{
		return velocity;
	}

	public Vector3 GetAccel()
	{
		return acceleration;
	}
	public Vector3 GetGyro()
	{
	  return gyro;
  	}
	public float GetOrient ()
	{
		Vector3 dir = transform.forward;
		return Mathf.Atan2( dir.z, dir.x);
	}

	public Transform GetTransform()
	{
		return this.transform;
	}

	public bool IsStill()
	{
		return rb.IsSleeping();
	}

	public void RequestFootBrake(float val)
	{
		requestBrake = val;
	}

	public void RequestHandBrake(float val)
	{
		//todo
	}
	
	// Update is called once per frame
	void Update () {
	
		UpdateWheelPositions();
	}

	public string GetActivity()
	{
		return activity;
	}

	public void SetActivity(string act)
	{
		activity = act;
	}

	void FixedUpdate()
	{
		lastSteer = requestSteering;
		lastAccel = requestTorque;
		prevVel = velocity;
		velocity = transform.InverseTransformDirection(rb.velocity);
		acceleration = (velocity - prevVel)/Time.fixedDeltaTime; // FixedUpdate 内用 fixedDeltaTime:帧率波动时 deltaTime 让加速度遥测失真
		gyro = rb.angularVelocity;
		rotation = rb.rotation;

		ApplyRequests();

		if (Mathf.Abs(requestTorque) > 0.1f && rb.velocity.magnitude < 0.5f)
			stuckTimer += Time.fixedDeltaTime;
		else
			stuckTimer = 0f;

	}

	// 把 requestTorque/requestSteering/requestBrake 写到 WheelCollider。
	// FixedUpdate 每物理步调用;同步步进(timeScale=0,FixedUpdate 不跑)由
	// TcpCarHandler 在 Physics.Simulate 前显式调用,否则轮上力矩停留在旧值。
	public void ApplyRequests()
	{
		float velMag = transform.InverseTransformDirection(rb.velocity).magnitude;

		// use the torque curve
		float throttle = torqueCurve.Evaluate(velMag / maxSpeed) * requestTorque * maxTorque;
		float steerAngle = requestSteering;
		float brake = requestBrake * maxBreakTorque;

		//front two tires.
		wheelColliders[2].steerAngle = steerAngle;
		wheelColliders[3].steerAngle = steerAngle;

		//four wheel drive at the moment
		foreach(WheelCollider wc in wheelColliders)
		{
			wc.motorTorque = throttle;
			wc.brakeTorque = brake;
		}
	}

	// 同步步进后刷新遥测依赖的运动学缓存(velocity/acceleration/gyro)。
	// dt 为本次手动推进的总仿真时长;FixedUpdate 不跑时这些字段否则全部冻结。
	public void SyncKinematics(float dt)
	{
		prevVel = velocity;
		velocity = transform.InverseTransformDirection(rb.velocity);
		if (dt > 0f)
			acceleration = (velocity - prevVel) / dt;
		gyro = rb.angularVelocity;
		rotation = rb.rotation;
	}

	void FlipUpright()
	{
		Quaternion rot = Quaternion.Euler(180f, 0f, 0f);
		this.transform.rotation = transform.rotation * rot;
		transform.position = transform.position + Vector3.up * 2;
	}

	void UpdateWheelPositions()
	{
		Quaternion rot;
		Vector3 pos;

		for(int i = 0; i < wheelColliders.Length; i++)
		{
			WheelCollider wc = wheelColliders[i];
			Transform tm = wheelMeshes[i];

			wc.GetWorldPose(out pos, out rot);

			tm.position = pos;
			tm.rotation = rot;
		}
	}

	//get the name of the last object we collided with
	public string GetLastCollision()
	{
		return last_collision;
	}

	public void ClearLastCollision()
	{
		last_collision = "none";
	}

	public float LastCollisionImpulse
	{
		get { return last_collision_impulse; }
	}

	public void ClearLastCollisionImpulse()
	{
		last_collision_impulse = 0f;
	}

	public bool IsFlipped
	{
		get { return Vector3.Dot(transform.up, Vector3.up) < 0.2f; }
	}

	public float StuckTime
	{
		get { return stuckTimer; }
	}

	void OnCollisionEnter(Collision col)
	{
		last_collision = col.gameObject.name;
		last_collision_impulse = col.impulse.magnitude;
	}
}
