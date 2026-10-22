using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityStandardAssets.CrossPlatformInput;

public class UnityStandardCarAdapter : MonoBehaviour, ICar {

	public UnityStandardAssets.Vehicles.Car.CarController unityCar;
	public float MaximumSteerAngle = 25.0f; //has to be kept in sync with the car, as that's a private var.
	float steering = 0.0f;
	float throttle = 0.0f;
	float footBrake = 0.0f;
	float handBrake = 0.0f;
	Vector3 vel = Vector3.zero;
	Vector3 accel = Vector3.zero;
	Quaternion rotation = Quaternion.identity;
	Vector3 gyro = Vector3.zero;
	public string activity = "keep_lane";

	Rigidbody rb;

	public Vector3 startPos;
	public Quaternion startRot;

	void Awake()
	{
		rb = unityCar.GetComponent<Rigidbody>();
		SavePosRot();
	}

	//all inputs require 0-1 input except steering which is in degrees, where 0 is center.
	public void RequestThrottle(float val) { throttle = val; }

	public void RequestSteering(float val) { steering = val; }

	public void RequestFootBrake(float val) { footBrake = val; }

	public void RequestHandBrake(float val) { handBrake = val; }


	//query last input given.
	public float GetSteering() { return steering; }

	public float GetThrottle() { return throttle; }

	public float GetFootBrake() { return footBrake; }

	public float GetHandBrake() { return handBrake; }


	//query state.
	public Transform GetTransform() { return this.transform; }

	public Vector3 GetVelocity()
	{
		return rb.velocity;
	}

	public Vector3 GetAccel() { return accel; }
	public Vector3 GetGyro() { return gyro; }
    public void SetMaxSteering(float val)
    {
        MaximumSteerAngle = val;
    }

    public float GetMaxSteering()
    {
        return MaximumSteerAngle;
    }

	//Save and restore State
	public void SavePosRot() 
	{ 
		startPos = transform.position;
		startRot = transform.rotation;
	}

	public void RestorePosRot()
	{
		Set(startPos, startRot);
	}

	public void Set(Vector3 pos, Quaternion rot)
	{
		rb.position = pos;
		rb.rotation = rot;

		//just setting it once doesn't seem to work. Try setting it multiple times..
		StartCoroutine(KeepSetting(pos, rot, 10));
	}

	IEnumerator KeepSetting(Vector3 pos, Quaternion rot, int numIter)
	{
		while(numIter > 0)
		{
			rb.position = pos;
			rb.rotation = rot;
			transform.position = pos;
			transform.rotation = rot;
			rb.velocity = Vector3.zero;
			rb.angularVelocity = Vector3.zero;

			numIter--;
			yield return new WaitForFixedUpdate();
		}
	}

	private void FixedUpdate()
	{
		accel = rb.velocity - vel;
		vel = rb.velocity;

		// 人驾 BC 录制桥:menu Manual Drive 的键盘由 CarUserControl 直调
		// CarController.Move,绕开 ICar 字段——遥测 steering_angle/throttle 回显
		// 恒零(2026-10-20 HBC 探针定案)。这里在 CarUserControl 之后(脚本序)
		// 自读同一虚拟轴回写字段:GetSteering/GetThrottle÷steer_to_angle(16)
		// 即归一化人驾动作;且与真实 Move 参数一致,双写无冲突。门=同车有
		// 启用的 CarUserControl(人驾车);RL 车无该组件,字段仍由 TCP 独占。
		var uc = GetComponent<UnityStandardAssets.Vehicles.Car.CarUserControl>();
		if (uc != null && uc.enabled)
		{
			float h = UnityStandardAssets.CrossPlatformInput.CrossPlatformInputManager.GetAxis("Horizontal");
			float v = UnityStandardAssets.CrossPlatformInput.CrossPlatformInputManager.GetAxis("Vertical");
			steering = h * MaximumSteerAngle;  // 驱动保真(与 CarUserControl 的 Move 参数一致)
			throttle = Mathf.Clamp01(v);
			// 注意:遥测 echo = GetSteering()/steer_to_angle(16) = h×25/16,
			// 客户端需 ×16/25 还原归一化(见 hbc_record.py STERER_ECHO_SCALE)
		}

		unityCar.Move(steering / MaximumSteerAngle, throttle, footBrake, handBrake);
		gyro = rb.angularVelocity;
		rotation = rb.rotation;
	}

	public string GetActivity()
	{
		return activity;
	}

	public void SetActivity(string act)
	{
		activity = act;
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

	void OnCollisionEnter(Collision col)
	{
		last_collision = col.gameObject.name;
	}

	string last_collision = "none";
}
