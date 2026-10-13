using System.Collections;
using System.Collections.Generic;
using System.Threading;
using Unity.Collections;
using UnityEngine;

public class CameraSensor : MonoBehaviour {

	public Camera sensorCam;
	public int width = 256;
	public int height = 256;
	public int depth = 3;
	public string img_enc = "JPG"; //accepts JPG, PNG, TGA
	Texture2D tex;
	RenderTexture ren;
	Rect ImageRect;

	// ---- 异步采集管线(AsyncGPUReadback + 工作线程 JPEG) ----
	// 动机:旧实现每发送一帧在主线程做 sensorCam.Render()(冗余,相机设了
	// targetTexture 本来就每帧自动渲一次)+ ReadPixels(GPU 同步停顿)+
	// EncodeToJPG(主线程编码),合计 ~6-12ms,与渲染/物理互相挤压,把遥测
	// ceiling 压到 ~58Hz。新实现:Update 里向 RT 发异步回读请求(本帧渲染
	// 完成后由 GPU 直接拷出,无停顿)→ 主线程只做 RGBA→RGB 拷包 → JPEG 编码
	// 挪到工作线程(stb) → GetImageBytes 永远立即返回最近一帧。
	// 回退:首帧/请求出错/库缺失时走旧同步路径(ReadPixels+EncodeToJPG)。
	const int JPG_QUALITY = 75; // 与 Unity EncodeToJPG 默认一致,保持码率可比

	byte[] lastJpg;
	readonly object jpgLock = new object();
	// 2022.3 中该 API 位于 UnityEngine.Rendering 命名空间(根命名空间无别名)
	UnityEngine.Rendering.AsyncGPUReadbackRequest pendingReq;
	bool hasPending;
	bool readbackBroken;      // 请求出错后永久回退同步路径(只报一次)
	bool fallbackLogged;
	Thread encodeThread;
	readonly ManualResetEventSlim encodeWake = new ManualResetEventSlim(false);
	readonly Queue<byte[]> jobs = new Queue<byte[]>();
	readonly object jobsLock = new object();
	readonly Queue<byte[]> bufferPool = new Queue<byte[]>();
	volatile bool running = true;

	/// <summary>最近一次采包耗时(ms,主线程段),TelemetryStats 用。</summary>
	public float LastCaptureMs { get; private set; }

	public void SetConfig(float fov, float offset_x, float offset_y, float offset_z, float rot_x, float rot_y, float rot_z, int img_w, int img_h, int img_d, string _img_enc)
	{
		if (img_d != 0)
		{
			depth = img_d;
		}

		if (img_w != 0 && img_h != 0)
		{
			width = img_w;
			height = img_h;

			Awake();
		}

		if(_img_enc.Length == 3)
			img_enc = _img_enc;

		if(offset_x != 0.0f || offset_y != 0.0f || offset_z != 0.0f)
			transform.localPosition = new Vector3(offset_x, offset_y, offset_z);

		if (rot_x != 0.0f || rot_y != 0.0f || rot_z != 0.0f)
		{
			transform.localEulerAngles = new Vector3(rot_x, rot_y, rot_z);
		}

		if(fov != 0.0f)
			sensorCam.fieldOfView = fov;
	}

	void Awake()
	{
		tex = new Texture2D(width, height, TextureFormat.RGB24, false);
		ren = new RenderTexture(width, height, 16, RenderTextureFormat.ARGB32);
		sensorCam.targetTexture = ren;
	}

	void OnDestroy()
	{
		running = false;
		encodeWake.Set();
	}

	/// <summary>主线程每帧调用(TcpCarHandler.Update):轮询回读结果、喂编码线程。</summary>
	public void Tick()
	{
		float t0 = Time.realtimeSinceStartup * 1000f;

		if (hasPending && pendingReq.done)
		{
			hasPending = false;
			if (pendingReq.hasError)
			{
				readbackBroken = true;
				pendingReq = default;
				if (!fallbackLogged)
				{
					fallbackLogged = true;
					Debug.LogWarning("[CameraSensor] AsyncGPUReadback 出错,回退同步 ReadPixels 路径。");
				}
			}
			else
			{
				// 请求完成后 NativeArray 即失效,立刻拷成 RGB24 再放手
				NativeArray<byte> rgba = pendingReq.GetData<byte>();
				byte[] rgb = RentBuffer();
				bool gray = (depth == 1);
				int px = width * height;
				for (int i = 0; i < px; i++)
				{
					int s = i * 4, d = i * 3;
					byte r = rgba[s], g = rgba[s + 1], b = rgba[s + 2];
					if (gray)
					{
						byte lum = (byte)(0.2126f * r + 0.7152f * g + 0.0722f * b);
						r = g = b = lum;
					}
					rgb[d] = r; rgb[d + 1] = g; rgb[d + 2] = b;
				}
				pendingReq = default;

				byte[] jpg = null;
				if (!StbJpeg.Available)
					jpg = EncodeOnMainThread(rgb); // 无原生库:主线程编码(仍无 GPU 停顿)
				else
				{
					lock (jobsLock) { jobs.Enqueue(rgb); }
					encodeWake.Set();
					EnsureEncodeThread();
				}
				if (jpg != null)
					lock (jpgLock) { lastJpg = jpg; }
				// 主线程编码路径的 buffer 在 EncodeOnMainThread 内归还
			}
		}

		LastCaptureMs = Time.realtimeSinceStartup * 1000f - t0;
	}

	/// <summary>发送节流触发新请求(发送频率=传感器相机渲染次数,无额外 Render)。</summary>
	void KickReadback()
	{
		if (readbackBroken || hasPending || ren == null)
			return;
		try
		{
			pendingReq = UnityEngine.Rendering.AsyncGPUReadback.Request(ren);
			hasPending = true;
		}
		catch (System.Exception)
		{
			readbackBroken = true;
		}
	}

	void EnsureEncodeThread()
	{
		if (encodeThread != null)
			return;
		encodeThread = new Thread(EncodeLoop)
		{
			IsBackground = true,
			Name = "CameraSensorJpeg",
		};
		encodeThread.Start();
	}

	void EncodeLoop()
	{
		while (running)
		{
			encodeWake.Wait();
			while (true)
			{
				byte[] rgb = null;
				lock (jobsLock)
				{
					if (jobs.Count == 0) break;
					rgb = jobs.Dequeue();
				}
				byte[] jpg = StbJpeg.Encode(rgb, width, height, JPG_QUALITY);
				if (jpg != null)
					lock (jpgLock) { lastJpg = jpg; }
				ReturnBuffer(rgb);
			}
		}
	}

	byte[] EncodeOnMainThread(byte[] rgb)
	{
		try
		{
			tex.LoadRawTextureData(rgb);
			tex.Apply();
			return tex.EncodeToJPG(JPG_QUALITY);
		}
		finally
		{
			ReturnBuffer(rgb);
		}
	}

	byte[] RentBuffer()
	{
		int need = width * height * 3;
		lock (bufferPool)
		{
			if (bufferPool.Count > 0)
			{
				byte[] b = bufferPool.Dequeue();
				if (b.Length == need)
					return b;
			}
		}
		return new byte[need];
	}

	void ReturnBuffer(byte[] b)
	{
		if (b == null)
			return;
		lock (bufferPool)
		{
			if (bufferPool.Count < 4)
				bufferPool.Enqueue(b);
		}
	}

	// ---------- 旧同步路径(Logger.cs 截图/首帧/回退用) ----------

	Texture2D RTImage()
	{
		var currentRT = RenderTexture.active;
		RenderTexture.active = sensorCam.targetTexture;

		sensorCam.Render();

		tex.ReadPixels(new Rect(0, 0, width, height), 0, 0);

		if(depth == 1)
		{
			//assumes TextureFormat.RGB24. Convert to grey scale image
			NativeArray<byte> bytes = tex.GetRawTextureData<byte>();
			for (int i=0; i<bytes.Length; i+=3)
			{
				byte gray = (byte)(0.2126f * bytes[i+0] + 0.7152f * bytes[i+1] + 0.0722f * bytes[i+2]);
				bytes[i+2] = bytes[i+1] = bytes[i+0] = gray;
			}
		}

		tex.Apply();
		RenderTexture.active = currentRT;

		return tex;
	}

	public Texture2D GetImage()
	{
		return RTImage();
	}

	public byte[] GetImageBytes()
	{
		if(img_enc == "PNG")
			return GetImage().EncodeToPNG();

		if(img_enc == "TGA")
			return GetImage().EncodeToTGA();

		// 新管线:kick 一次请求,返回最近已编码帧(首帧尚未就绪时回退同步路径)
		KickReadback();
		lock (jpgLock)
		{
			if (lastJpg != null)
				return lastJpg;
		}
		return GetImage().EncodeToJPG();
	}
}
