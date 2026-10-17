using System.Collections;
using UnityEngine;
using System;
using UnityEngine.UI;
using System.Globalization;
using UnityEngine.SceneManagement;
using UnityStandardAssets.ImageEffects;
using UnityEngine.AI;
using System.Collections.Generic;

namespace tk
{

    public class TcpCarHandler : MonoBehaviour
    {

        public GameObject carObj;
        public ICar car;
        public CarSpawner carSpawner;
        public PathManager pm;
        public CarConfig conf;

        // Sensors
        public CameraSensor camSensor;
        public CameraSensor camSensorB;
        public Lidar lidar;
        public Odometry[] odom;

        private tk.JsonTcpClient client;
        public Text ai_text;

        public float limitFPS = 20.0f;
        float timeSinceLastCapture = 0.0f;
        public float timeSinceLastMoved = 0.0f;
        public Vector3 lastPos = Vector3.zero;
        public float lastDistanceTraveled = 0.0f;

        float steer_to_angle = 16.0f;

        float ai_steering = 0.0f;
        float ai_throttle = 0.0f;
        float ai_brake = 0.0f;

        int iActiveSpan = 0;

        public bool RlMode { get; private set; }
        float controlTimeoutMs = 0f;
        DateTime lastControlRecvTime = DateTime.UtcNow;
        bool deadmanTriggered = false;
        uint frameId = 0;
        int lapCount = 0;
        int lastActiveNode = -1;
        int resetNodeIndex = -1;
        int resetNodeMin = 0;
        int resetNodeMax = -1;
        float resetLateralOffset = 0f;
        float resetHeadingNoise = 0f;
        int? resetSeed = null;
        Car carComp0;

        bool asynchronous = true;
        float time_step = 0.1f;
        bool bResetCar = false;
        bool bResetToNearest = false;
        bool bExitScene = false;
        float savedTimeScale = 1f;
        float syncFixedDt = 0.01f;
        int pendingSteps = 0;
        bool syncCapturePending = false;
        int syncKickedFrame = 0;

        public enum State
        {
            UnConnected,
            SendTelemetry
        }

        public State state = State.UnConnected;

        void Awake()
        {
            car = carObj.GetComponent<ICar>();
            conf = carObj.GetComponent<CarConfig>();
            carComp0 = car as Car;
            pm = GameObject.FindObjectOfType<PathManager>();
            carSpawner = GameObject.FindObjectOfType<CarSpawner>();
            Canvas canvas = GameObject.FindObjectOfType<Canvas>();
            GameObject go = CarSpawner.getChildGameObject(canvas.gameObject, "AISteering");
            if (go != null)
                ai_text = go.GetComponent<Text>();

            if (pm != null && carObj != null)
            {
                iActiveSpan = pm.carPath.GetClosestSpanIndex(carObj.transform.position);
            }
        }

        public void Init(tk.JsonTcpClient _client)
        {
            client = _client;

            if (client == null)
                return;

            client.dispatchInMainThread = false; //too slow to wait.
            client.dispatcher.Register("get_protocol_version", new tk.Delegates.OnMsgRecv(OnProtocolVersion));
            client.dispatcher.Register("control", new tk.Delegates.OnMsgRecv(OnControlsRecv));
            client.dispatcher.Register("exit_scene", new tk.Delegates.OnMsgRecv(OnExitSceneRecv));
            client.dispatcher.Register("reset_car", new tk.Delegates.OnMsgRecv(OnResetCarRecv));
            client.dispatcher.Register("reset_to_nearest", new tk.Delegates.OnMsgRecv(OnResetToNearestRecv));
            client.dispatcher.Register("step_mode", new tk.Delegates.OnMsgRecv(OnStepModeRecv));
            client.dispatcher.Register("quit_app", new tk.Delegates.OnMsgRecv(OnQuitApp));
            client.dispatcher.Register("regen_road", new tk.Delegates.OnMsgRecv(OnRegenRoad));
            client.dispatcher.Register("car_config", new tk.Delegates.OnMsgRecv(OnCarConfig));
            client.dispatcher.Register("cam_config", new tk.Delegates.OnMsgRecv(OnCamConfig));
            client.dispatcher.Register("cam_config_b", new tk.Delegates.OnMsgRecv(OnCamConfigB));
            client.dispatcher.Register("lidar_config", new tk.Delegates.OnMsgRecv(OnLidarConfig));
            client.dispatcher.Register("set_position", new tk.Delegates.OnMsgRecv(OnSetPosition));
            client.dispatcher.Register("node_position", new tk.Delegates.OnMsgRecv(OnNodePositionRecv));
            client.dispatcher.Register("time_scale", new tk.Delegates.OnMsgRecv(OnTimeScaleRecv));
        }

        public void Start()
        {
            SendCarLoaded();
            state = State.SendTelemetry;
        }

        public tk.JsonTcpClient GetClient()
        {
            return client;
        }

        public void OnDestroy()
        {
            if (client != null && client.dispatcher != null)
                client.dispatcher.Reset();

            if (!asynchronous)
            {
                Physics.autoSimulation = true;
                Time.timeScale = savedTimeScale;
            }
        }

        void Disconnect()
        {
            client.Disconnect();
        }

        public void Boot()
        {
            if (carSpawner != null)
            {
                if (client != null) { Disconnect(); }
                carSpawner.RemoveCar(client);
            }
        }

        void OnProtocolVersion(JSONObject msg)
        {
            JSONObject json = new JSONObject(JSONObject.Type.OBJECT);
            json.AddField("msg_type", "protocol_version");
            json.AddField("version", "2");

            client.SendMsg(json);
        }

        void SendTelemetry()
        {
            if (client == null)
                return;

            frameId++;

            // 非 JPG 编码(Logger/PNG/TGA 场景)不在性能关键路径,走旧 JSONObject 路径
            if (camSensor.img_enc != "JPG")
            {
                SendTelemetryLegacy();
                return;
            }

            byte[] buf = client.RentSendBuffer();
            if (buf == null)
            {
                // 发送池耗尽(客户端严重反压):遥测是实时数据,丢帧优于积压旧帧
                _txDrops++;
                return;
            }
            _txBuf = buf;
            _txLen = 0;
            _txOverflow = false;

            TxAscii("{\"msg_type\":\"telemetry\"");
            TxAscii(",\"steering_angle\":"); TxNum(car.GetSteering() / steer_to_angle);
            TxAscii(",\"throttle\":"); TxNum(car.GetThrottle());

            TxAscii(",\"image\":");
            if (!TxBase64Image(camSensor)) { AbortTx(buf); SendTelemetryLegacy(); return; }

            if (camSensorB != null && camSensorB.gameObject.activeInHierarchy)
            {
                TxAscii(",\"imageb\":");
                if (!TxBase64Image(camSensorB)) { AbortTx(buf); SendTelemetryLegacy(); return; }
            }

            if (lidar != null && lidar.gameObject.activeInHierarchy)
            {
                TxAscii(",\"lidar\":");
                TxAscii(lidar.GetOutputAsJson().ToString());
            }

            foreach (Odometry o in odom)
            {
                TxAscii(",\""); TxAscii(o.Label); TxAscii("\":"); TxNum(o.GetNumberRotations());
            }

            TxAscii(",\"hit\":\""); TxAscii(car.GetLastCollision()); TxAscii("\"");
            car.ClearLastCollision();

            TxAscii(",\"frame_id\":"); TxAscii(frameId.ToString(CultureInfo.InvariantCulture));
            float collImpulse = carComp0 != null ? carComp0.LastCollisionImpulse : 0f;
            if (carComp0 != null) carComp0.ClearLastCollisionImpulse();
            TxAscii(",\"collision_impulse\":"); TxNum(collImpulse);
            TxAscii(",\"flipped\":"); TxAscii(carComp0 != null && carComp0.IsFlipped ? "true" : "false");
            TxAscii(",\"stuck_time\":"); TxNum(carComp0 != null ? carComp0.StuckTime : 0f);

            TxAscii(",\"time\":"); TxNum(Time.timeSinceLevelLoad);

            Vector3 velocity = car.GetVelocity() / 8.0f;
            TxAscii(",\"speed\":"); TxNum(velocity.magnitude);

            Vector3 accel = car.GetAccel() / 8.0f;
            TxAscii(",\"accel_x\":"); TxNum(accel.x);
            TxAscii(",\"accel_y\":"); TxNum(accel.y);
            TxAscii(",\"accel_z\":"); TxNum(accel.z);

            Vector3 gyro = car.GetGyro();
            TxAscii(",\"gyro_x\":"); TxNum(gyro.x);
            TxAscii(",\"gyro_y\":"); TxNum(gyro.y);
            TxAscii(",\"gyro_z\":"); TxNum(gyro.z);

            Transform tm = car.GetTransform();
            Vector3 eulerAngles = tm.rotation.eulerAngles;
            TxAscii(",\"pitch\":"); TxNum(eulerAngles.x);
            TxAscii(",\"yaw\":"); TxNum(eulerAngles.y);
            TxAscii(",\"roll\":"); TxNum(eulerAngles.z);

            if (pm != null)
            {
                float cte = 0.0f;
                pm.carPath.GetCrossTrackErr(tm.position, ref iActiveSpan, ref cte); // get distance to closest node
                if (GlobalState.extendedTelemetry) { TxAscii(",\"cte\":"); TxNum(cte); }

                TxAscii(",\"activeNode\":"); TxNum(iActiveSpan);
                int totalNodes = pm.carPath.nodes.Count;
                TxAscii(",\"totalNodes\":"); TxNum(totalNodes);

                if (totalNodes > 0 && lastActiveNode > totalNodes * 0.9f && iActiveSpan < totalNodes * 0.1f)
                    lapCount++;
                lastActiveNode = iActiveSpan;
            }

            // not intended to use in races, just to train
            if (GlobalState.extendedTelemetry)
            {
                Vector3 pos = tm.position / 8.0f;
                TxAscii(",\"pos_x\":"); TxNum(pos.x);
                TxAscii(",\"pos_y\":"); TxNum(pos.y);
                TxAscii(",\"pos_z\":"); TxNum(pos.z);

                TxAscii(",\"vel_x\":"); TxNum(velocity.x);
                TxAscii(",\"vel_y\":"); TxNum(velocity.y);
                TxAscii(",\"vel_z\":"); TxNum(velocity.z);
            }
            TxAscii(",\"lap\":"); TxNum(lapCount);
            TxAscii("}\n");

            if (_txOverflow)
            {
                // 报文超出发送缓冲(lidar 等可变长字段):回退旧路径,不发送截断包
                AbortTx(buf);
                SendTelemetryLegacy();
                return;
            }

            if (!client.SendRaw(buf, _txLen))
                _txDrops++;
        }

        // ---- 零分配遥测拼包 ----
        // 旧路径 JSONObject.ToString+Convert.ToBase64String+UTF8.GetBytes 每帧产生
        // ~3x 报文大小的临时大对象(640x480@60fps ≈ 25MB/s 垃圾),非增量 GC 全停
        // 主线程,表现为窗口不定时卡顿闪动。新路径:池化发送缓冲(TcpClient)+
        // ToBase64CharArray+手工 ASCII 拼包,每帧仅剩几十字节的数字格式化小对象。
        // 字段名/顺序/类型与旧路径逐一对齐(见 SendTelemetryLegacy)。
        char[] _b64Chars;       // 池化 base64 字符缓冲,按需扩容
        byte[] _txBuf;          // 当前租用的发送缓冲
        int _txLen;
        bool _txOverflow;       // 拼包越界标志(缓冲不足时后续写入变 no-op,整体回退旧路径)
        long _txDrops;          // 拼包侧丢弃的遥测帧数(TelemetryStats 输出)

        void TxAscii(string s)
        {
            if (_txOverflow || _txLen + s.Length > _txBuf.Length) { _txOverflow = true; return; }
            for (int i = 0; i < s.Length; i++) _txBuf[_txLen++] = (byte)s[i];
        }
        void TxNum(double v) { TxAscii(v.ToString(CultureInfo.InvariantCulture)); }
        void TxNum(int v) { TxAscii(v.ToString(CultureInfo.InvariantCulture)); }

        bool TxBase64Image(CameraSensor sensor)
        {
            // 容量上界:JPEG <= w*h*3(RGB 原始大小,实际远小于此)
            int need = 4 * ((sensor.width * sensor.height * 3 + 2) / 3) + 16;
            if (_b64Chars == null || _b64Chars.Length < need)
                _b64Chars = new char[need];
            int n = sensor.GetLatestJpgBase64(_b64Chars);
            if (n < 0 || _txLen + n + 2 >= _txBuf.Length)
                return false;
            _txBuf[_txLen++] = (byte)'"';
            for (int i = 0; i < n; i++)
                _txBuf[_txLen++] = (byte)_b64Chars[i];
            _txBuf[_txLen++] = (byte)'"';
            return true;
        }

        void AbortTx(byte[] buf)
        {
            client.ReturnSendBuffer(buf);
            _txBuf = null;
            _txLen = 0;
            _txDrops++;
        }

        // 旧 JSONObject 发送路径:非 JPG 编码或零分配路径异常时回退,行为不变。
        void SendTelemetryLegacy()
        {
            JSONObject json = new JSONObject(JSONObject.Type.OBJECT);
            json.AddField("msg_type", "telemetry");

            json.AddField("steering_angle", car.GetSteering() / steer_to_angle);
            json.AddField("throttle", car.GetThrottle());

            json.AddField("image", Convert.ToBase64String(camSensor.GetImageBytes()));

            if (camSensorB != null && camSensorB.gameObject.activeInHierarchy)
            {
                json.AddField("imageb", Convert.ToBase64String(camSensorB.GetImageBytes()));
            }

            if (lidar != null && lidar.gameObject.activeInHierarchy)
            {
                json.AddField("lidar", lidar.GetOutputAsJson());
            }

            foreach (Odometry o in odom)
            {
                json.AddField(o.Label, o.GetNumberRotations());
            }


            json.AddField("hit", car.GetLastCollision());
            car.ClearLastCollision();

            json.AddField("frame_id", (int)frameId);
            float collImpulse = carComp0 != null ? carComp0.LastCollisionImpulse : 0f;
            if (carComp0 != null) carComp0.ClearLastCollisionImpulse();
            json.AddField("collision_impulse", collImpulse);
            json.AddField("flipped", carComp0 != null && carComp0.IsFlipped);
            json.AddField("stuck_time", carComp0 != null ? carComp0.StuckTime : 0f);

            json.AddField("time", Time.timeSinceLevelLoad);

            Vector3 velocity = car.GetVelocity() / 8.0f;
            json.AddField("speed", velocity.magnitude);

            Vector3 accel = car.GetAccel() / 8.0f;
            json.AddField("accel_x", accel.x);
            json.AddField("accel_y", accel.y);
            json.AddField("accel_z", accel.z);

            Vector3 gyro = car.GetGyro();
            json.AddField("gyro_x", gyro.x);
            json.AddField("gyro_y", gyro.y);
            json.AddField("gyro_z", gyro.z);

            Transform tm = car.GetTransform();
            Vector3 eulerAngles = tm.rotation.eulerAngles;
            json.AddField("pitch", eulerAngles.x);
            json.AddField("yaw", eulerAngles.y);
            json.AddField("roll", eulerAngles.z);

            if (pm != null)
            {
                float cte = 0.0f;
                pm.carPath.GetCrossTrackErr(tm.position, ref iActiveSpan, ref cte); // get distance to closest node
                if (GlobalState.extendedTelemetry) { json.AddField("cte", cte); }

                json.AddField("activeNode", iActiveSpan);
                int totalNodes = pm.carPath.nodes.Count;
                json.AddField("totalNodes", totalNodes);

                if (totalNodes > 0 && lastActiveNode > totalNodes * 0.9f && iActiveSpan < totalNodes * 0.1f)
                    lapCount++;
                lastActiveNode = iActiveSpan;
            }

            // not intended to use in races, just to train
            if (GlobalState.extendedTelemetry)
            {
                Vector3 pos = tm.position / 8.0f;
                json.AddField("pos_x", pos.x);
                json.AddField("pos_y", pos.y);
                json.AddField("pos_z", pos.z);

                json.AddField("vel_x", velocity.x);
                json.AddField("vel_y", velocity.y);
                json.AddField("vel_z", velocity.z);
            }
            json.AddField("lap", lapCount);
            client.SendMsg(json);
        }

        void SendCarLoaded()
        {
            if (client == null)
                return;

            JSONObject json = new JSONObject(JSONObject.Type.OBJECT);
            json.AddField("msg_type", "car_loaded");
            client.SendMsg(json);
            Debug.Log("car loaded.");
        }

        float clamp(float val, float low, float high)
        {
            float ret = val;
            if (val > high)
                ret = high;
            else if (val < low)
                ret = low;
            return ret;
        }

        void OnControlsRecv(JSONObject json)
        {
            try
            {
                lastControlRecvTime = DateTime.UtcNow;
                deadmanTriggered = false;

                ai_steering = float.Parse(json["steering"].str, CultureInfo.InvariantCulture.NumberFormat);
                ai_throttle = float.Parse(json["throttle"].str, CultureInfo.InvariantCulture.NumberFormat);
                ai_brake = float.Parse(json["brake"].str, CultureInfo.InvariantCulture.NumberFormat);

                ai_steering = clamp(ai_steering, -1.0f, 1.0f);
                ai_throttle = clamp(ai_throttle, -1.0f, 1.0f);
                ai_brake = clamp(ai_brake, 0.0f, 1.0f);

                ai_steering *= steer_to_angle;

                car.RequestSteering(ai_steering);
                car.RequestThrottle(ai_throttle);
                car.RequestFootBrake(ai_brake);

                if (!asynchronous)
                    System.Threading.Interlocked.Increment(ref pendingSteps);
            }
            catch (Exception e)
            {
                Debug.Log(e.ToString());
            }
        }

        void OnExitSceneRecv(JSONObject json)
        {
            bExitScene = true;
        }

        void ExitScene()
        {
            // 启动场景已改为 mountain_track(见 EditorBuildSettings),
            // 返回前端菜单必须按名字加载,不能再用 build index 0。
            SceneManager.LoadSceneAsync("menu");
        }

        // RL 客户端常发未加引号的 JSON 数字(NUMBER 型 .str 为 null),字符串/数字两种都要能收。
        static float NumField(JSONObject json, string name, float defVal)
        {
            JSONObject f = json.GetField(name);
            if (f == null)
                return defVal;
            if (f.str != null)
            {
                float v;
                if (float.TryParse(f.str, NumberStyles.Float, CultureInfo.InvariantCulture, out v))
                    return v;
                return defVal;
            }
            return f.IsNumber ? (float)f.n : defVal;
        }

        void OnResetCarRecv(JSONObject json)
        {
            resetNodeIndex = -1;
            resetNodeMin = 0;
            resetNodeMax = -1;
            resetLateralOffset = 0f;
            resetHeadingNoise = 0f;
            resetSeed = null;

            resetNodeIndex = (int)NumField(json, "node_index", -1f);
            resetNodeMin = (int)NumField(json, "node_min", 0f);
            resetNodeMax = (int)NumField(json, "node_max", -1f);
            resetLateralOffset = NumField(json, "lateral_offset", 0f);
            resetHeadingNoise = NumField(json, "heading_noise", 0f);
            if (json.HasField("seed"))
                resetSeed = (int)NumField(json, "seed", 0f);

            bResetCar = true;
        }

        void OnTimeScaleRecv(JSONObject json)
        {
            float v = NumField(json, "time_scale", 1.0f);
            v = clamp(v, 0.1f, 4.0f);
            Time.timeScale = v;
        }

        // 就近回位到赛道中央并摆正(与本地 R 键同一逻辑,见 CarRecovery)。
        // 旧版模拟器未注册此消息时由 Dispatcher 静默忽略,不崩溃。
        void OnResetToNearestRecv(JSONObject json)
        {
            bResetToNearest = true;
        }

        void OnRegenRoad(JSONObject json)
        {
            //This causes the track to be regenerated with the given settings.
            //This only works in scenes that have random track generation enabled.
            //For now that is only in scene road_generator.
            //An index into our track options. 5 in scene RoadGenerator.
            int road_style = int.Parse(json.GetField("road_style").str);
            int rand_seed = int.Parse(json.GetField("rand_seed").str);
            float turn_increment = float.Parse(json.GetField("turn_increment").str, CultureInfo.InvariantCulture);

            //We get this callback in a worker thread, but need to make mainthread calls.
            //so use this handy utility dispatcher from
            // https://github.com/PimDeWitte/UnityMainThreadDispatcher
            UnityMainThreadDispatcher.Instance().Enqueue(RegenRoad(road_style, rand_seed, turn_increment));
        }

        IEnumerator RegenRoad(int road_style, int rand_seed, float turn_increment)
        {
            TrainingManager train_mgr = GameObject.FindObjectOfType<TrainingManager>();
            PathManager path_mgr = GameObject.FindObjectOfType<PathManager>();

            if (train_mgr != null)
            {
                if (turn_increment != 0.0 && path_mgr != null)
                {
                    path_mgr.turnInc = turn_increment;
                }

                UnityEngine.Random.InitState(rand_seed);
                train_mgr.SetRoadStyle(road_style);
                train_mgr.OnMenuRegenTrack();
            }

            yield return null;
        }

        void OnCarConfig(JSONObject json)
        {
            Debug.Log("Got car config message");

            string body_style = json.GetField("body_style").str;
            int body_r = int.Parse(json.GetField("body_r").str);
            int body_g = int.Parse(json.GetField("body_g").str);
            int body_b = int.Parse(json.GetField("body_b").str);
            string car_name = json.GetField("car_name").str;
            int font_size = 100;

            RlMode = GlobalState.rlMode || (int)NumField(json, "rl_mode", 0f) == 1;
            controlTimeoutMs = NumField(json, "control_timeout_ms", 0f);

            if (json.GetField("font_size") != null)
                font_size = int.Parse(json.GetField("font_size").str);

            if (carObj != null && car_name != "Racer Name")
                UnityMainThreadDispatcher.Instance().Enqueue(SetCarConfig(body_style, body_r, body_g, body_b, car_name, font_size));
        }

        IEnumerator SetCarConfig(string body_style, int body_r, int body_g, int body_b, string car_name, int font_size)
        {
            CarConfig conf = carObj.GetComponent<CarConfig>();

            if (conf)
            {
                conf.SetStyle(body_style, body_r, body_g, body_b, car_name, font_size);
            }

            yield return null;
        }

        void OnCamConfig(JSONObject json)
        {
            ParseCamConfig(json, 0);
        }

        void OnCamConfigB(JSONObject json)
        {
            ParseCamConfig(json, 1);
        }

        void ParseCamConfig(JSONObject json, int iCamera)
        {
            float fov = float.Parse(json.GetField("fov").str, CultureInfo.InvariantCulture.NumberFormat);
            float offset_x = float.Parse(json.GetField("offset_x").str, CultureInfo.InvariantCulture.NumberFormat);
            float offset_y = float.Parse(json.GetField("offset_y").str, CultureInfo.InvariantCulture.NumberFormat);
            float offset_z = float.Parse(json.GetField("offset_z").str, CultureInfo.InvariantCulture.NumberFormat);
            float rot_x = float.Parse(json.GetField("rot_x").str, CultureInfo.InvariantCulture.NumberFormat);
            float rot_y = 0f;
            float rot_z = 0f;
            if (json.HasField("rot_y")) { rot_y = float.Parse(json.GetField("rot_y").str, CultureInfo.InvariantCulture.NumberFormat); }
            if (json.HasField("rot_z")) { rot_z = float.Parse(json.GetField("rot_z").str, CultureInfo.InvariantCulture.NumberFormat); }
            float fish_eye_x = float.Parse(json.GetField("fish_eye_x").str, CultureInfo.InvariantCulture.NumberFormat);
            float fish_eye_y = float.Parse(json.GetField("fish_eye_y").str, CultureInfo.InvariantCulture.NumberFormat);
            int img_w = int.Parse(json.GetField("img_w").str);
            int img_h = int.Parse(json.GetField("img_h").str);
            int img_d = int.Parse(json.GetField("img_d").str);
            string img_enc = json.GetField("img_enc").str;
            // 可选遥测推流帧率(默认 20 维持旧行为;>0 时覆盖 limitFPS)
            float fps = 0f;
            if (json.HasField("fps")) { fps = float.Parse(json.GetField("fps").str, CultureInfo.InvariantCulture.NumberFormat); }

            if (carObj != null)
                UnityMainThreadDispatcher.Instance().Enqueue(SetCamConfig(iCamera, fov, offset_x, offset_y, offset_z, rot_x, rot_y, rot_z, img_w, img_h, img_d, img_enc, fish_eye_x, fish_eye_y, fps));
        }

        IEnumerator SetCamConfig(int iCamera, float fov, float offset_x, float offset_y, float offset_z, float rot_x,
            float rot_y, float rot_z, int img_w, int img_h, int img_d, string img_enc, float fish_eye_x, float fish_eye_y,
            float fps = 0f)
        {
            // cam_config 的 fps 键:遥测推流帧率。必须在主线程应用(发送节流在
            // Update 里读 limitFPS)。<=0 保持默认 20 不变,向后兼容。
            if (fps > 0f)
                limitFPS = fps;

            CameraSensor cam = null;

            if (iCamera == 0)
                cam = camSensor;
            else
            {
                cam = camSensorB;

                if (cam != null && !cam.gameObject.activeInHierarchy)
                {
                    cam.gameObject.SetActive(true);
                }
            }

            if (cam)
            {
                cam.SetConfig(fov, offset_x, offset_y, offset_z, rot_x, rot_y, rot_z, img_w, img_h, img_d, img_enc);

                Fisheye fe = cam.gameObject.GetComponent<Fisheye>();

                if (fe != null && (fish_eye_x != 0.0f || fish_eye_y != 0.0f))
                {
                    fe.enabled = true;
                    fe.strengthX = fish_eye_x;
                    fe.strengthY = fish_eye_y;
                }
            }

            yield return null;
        }

        void OnSetPosition(JSONObject json)
        {
            if (GlobalState.extendedTelemetry)
            {
                float pos_x = float.Parse(json.GetField("pos_x").str, CultureInfo.InvariantCulture.NumberFormat);
                float pos_y = float.Parse(json.GetField("pos_y").str, CultureInfo.InvariantCulture.NumberFormat);
                float pos_z = float.Parse(json.GetField("pos_z").str, CultureInfo.InvariantCulture.NumberFormat);
                Quaternion rot = Quaternion.identity;
                if (json.GetField("Qx") != null && json.GetField("Qy") != null && json.GetField("Qz") != null && json.GetField("Qw") != null)
                {
                    float qx = float.Parse(json.GetField("Qx").str, CultureInfo.InvariantCulture.NumberFormat);
                    float qy = float.Parse(json.GetField("Qy").str, CultureInfo.InvariantCulture.NumberFormat);
                    float qz = float.Parse(json.GetField("Qz").str, CultureInfo.InvariantCulture.NumberFormat);
                    float qw = float.Parse(json.GetField("Qw").str, CultureInfo.InvariantCulture.NumberFormat);

                    rot.x = qx;
                    rot.y = qy;
                    rot.z = qz;
                    rot.w = qw;
                }

                UnityMainThreadDispatcher.Instance().Enqueue(setCarPosition(pos_x, pos_y, pos_z, rot));
            }
        }

        IEnumerator setCarPosition(float pos_x, float pos_y, float pos_z, Quaternion rot)
        {
            carObj.transform.position = new Vector3(pos_x, pos_y, pos_z);
            carObj.transform.rotation = rot;
            yield return null;
        }

        void OnLidarConfig(JSONObject json)
        {
            float offset_x = float.Parse(json.GetField("offset_x").str, CultureInfo.InvariantCulture.NumberFormat);
            float offset_y = float.Parse(json.GetField("offset_y").str, CultureInfo.InvariantCulture.NumberFormat);
            float offset_z = float.Parse(json.GetField("offset_z").str, CultureInfo.InvariantCulture.NumberFormat);
            float rot_x = float.Parse(json.GetField("rot_x").str, CultureInfo.InvariantCulture.NumberFormat);

            float degPerSweepInc = float.Parse(json.GetField("degPerSweepInc").str, CultureInfo.InvariantCulture.NumberFormat);
            float degAngDown = float.Parse(json.GetField("degAngDown").str, CultureInfo.InvariantCulture.NumberFormat);
            float degAngDelta = float.Parse(json.GetField("degAngDelta").str, CultureInfo.InvariantCulture.NumberFormat);
            float maxRange = float.Parse(json.GetField("maxRange").str, CultureInfo.InvariantCulture.NumberFormat);
            float noise = float.Parse(json.GetField("noise").str, CultureInfo.InvariantCulture.NumberFormat);
            int numSweepsLevels = int.Parse(json.GetField("numSweepsLevels").str);


            if (carObj != null)
                UnityMainThreadDispatcher.Instance().Enqueue(SetLidarConfig(offset_x, offset_y, offset_z, rot_x, degPerSweepInc, degAngDown, degAngDelta, maxRange, noise, numSweepsLevels));
        }

        IEnumerator SetLidarConfig(float offset_x, float offset_y, float offset_z, float rot_x,
            float degPerSweepInc, float degAngDown, float degAngDelta, float maxRange, float noise, int numSweepsLevels)
        {
            if (lidar != null)
            {
                if (!lidar.gameObject.activeInHierarchy)
                    lidar.gameObject.SetActive(true);

                lidar.SetConfig(offset_x, offset_y, offset_z, rot_x, degPerSweepInc, degAngDown, degAngDelta, maxRange, noise, numSweepsLevels);
            }

            yield return null;
        }

        void OnStepModeRecv(JSONObject json)
        {
            string step_mode = json.GetField("step_mode").str;
            float _time_step = NumField(json, "time_step", 0.05f);

            if (step_mode == "synchronous")
            {
                asynchronous = false;
                this.time_step = _time_step > 0f ? _time_step : 0.05f;
                syncFixedDt = Time.fixedDeltaTime;
                savedTimeScale = Time.timeScale;
                Time.timeScale = 0.0f;
                Physics.autoSimulation = false;
                pendingSteps = 0;
                syncCapturePending = false;
                Debug.Log(string.Format("[TcpCarHandler] synchronous mode, time_step={0}", this.time_step));
                Debug.LogWarning("[TcpCarHandler] synchronous mode is instance-global; use one RL client per sim instance");
            }
            else
            {
                asynchronous = true;
                Physics.autoSimulation = true;
                Time.timeScale = savedTimeScale;
                pendingSteps = 0;
                syncCapturePending = false;
                Debug.Log("[TcpCarHandler] asynchronous mode");
            }
        }

        public IEnumerator SendCollisionWithStartingLine(int startingLineIndex, float timeStamp)
        {
            if (client != null)
            {
                JSONObject json = new JSONObject(JSONObject.Type.OBJECT);
                json.AddField("msg_type", "collision_with_starting_line");
                json.AddField("starting_line_index", startingLineIndex);
                json.AddField("timeStamp", timeStamp);
                client.SendMsg(json);
            }

            yield return null;
        }

        public void OnNodePositionRecv(JSONObject json)
        {
            int index = (int)NumField(json, "index", -1f);
            if (pm != null && index >= 0 && index < pm.carPath.nodes.Count)
            {
                PathNode node = pm.carPath.nodes[index];
                UnityMainThreadDispatcher.Instance().Enqueue(SendNodePosition(index, node));
            }
        }

        public IEnumerator SendNodePosition(int index, PathNode node)
        {

            JSONObject json = new JSONObject(JSONObject.Type.OBJECT);
            json.AddField("msg_type", "node_position");
            json.AddField("index", index);
            json.AddField("pos_x", node.pos.x);
            json.AddField("pos_y", node.pos.y);
            json.AddField("pos_z", node.pos.z);
            json.AddField("Qx", node.rotation.x);
            json.AddField("Qy", node.rotation.y);
            json.AddField("Qz", node.rotation.z);
            json.AddField("Qw", node.rotation.w);
            client.SendMsg(json);
            yield return null;
        }

        void OnQuitApp(JSONObject json)
        {
            Application.Quit();
        }

        void FixedUpdate()
        {
            if (bExitScene)
            {
                bExitScene = false;
                ExitScene();
            }

            if (state == State.SendTelemetry)
            {
                if (bResetCar)
                {
                    DoResetCar(false);
                    bResetCar = false;
                }

                if (bResetToNearest)
                {
                    DoResetToNearest(false);
                    bResetToNearest = false;
                }

                if (ai_text != null)
                    ai_text.text = string.Format("NN: {0} : {1}", ai_steering, ai_throttle);

                Vector3 currentPos = car.GetTransform().position;
                float distance = Vector3.Distance(currentPos, lastPos);

                if (distance < 1f)
                {
                    timeSinceLastMoved += Time.fixedDeltaTime;
                }
                else
                {
                    timeSinceLastMoved = 0.0f;
                    lastPos = currentPos;
                }
                if (!RlMode && GlobalState.timeOut > 0f && timeSinceLastMoved >= GlobalState.timeOut && carSpawner != null)
                {
                    Boot();
                }
            }
        }

        void DoResetCar(bool immediate)
        {
            CarPath path = pm != null ? pm.carPath : null;
            bool randomReset = resetNodeIndex >= 0 || resetNodeMin > 0 || resetNodeMax >= 0
                || resetLateralOffset != 0f || resetHeadingNoise != 0f || resetSeed.HasValue;

            if (randomReset && path != null && path.nodes.Count > 0 && carComp0 != null)
            {
                int totalNodes = path.nodes.Count;
                int lo = Mathf.Clamp(resetNodeMin, 0, totalNodes - 1);
                int hi = resetNodeMax >= 0 ? Mathf.Min(resetNodeMax, totalNodes - 1) : totalNodes - 1;
                if (lo > hi)
                    lo = hi;

                System.Random rng = resetSeed.HasValue ? new System.Random(resetSeed.Value) : new System.Random();
                int iNode = resetNodeIndex >= 0 ? Mathf.Clamp(resetNodeIndex, 0, totalNodes - 1) : rng.Next(lo, hi + 1);
                PathNode node = path.GetNode(iNode);

                float hNoise = 0f;
                if (resetHeadingNoise != 0f)
                    hNoise = ((float)rng.NextDouble() * 2f - 1f) * resetHeadingNoise;

                Vector3 tangent = node.rotation * Vector3.forward;
                Vector3 right = Vector3.Cross(Vector3.up, tangent).normalized;
                Vector3 pos = node.pos + right * resetLateralOffset + Vector3.up * 0.4f;
                Quaternion rot = node.rotation * Quaternion.Euler(0f, hNoise, 0f);

                if (immediate)
                    carComp0.SetImmediate(pos, rot, syncFixedDt, 2);
                else
                {
                    carComp0.rb.velocity = Vector3.zero;
                    carComp0.rb.angularVelocity = Vector3.zero;
                    carComp0.Set(pos, rot);
                }

                car.RequestSteering(0.0f);
                car.RequestThrottle(0.0f);
                car.RequestFootBrake(10.0f);

                iActiveSpan = iNode;
                lapCount = 0;
                lastActiveNode = iNode;

                Debug.Log(string.Format("[TcpCarHandler] reset_car: random reset node {0} seed {1}",
                    iNode, resetSeed.HasValue ? resetSeed.Value.ToString() : "time"));
            }
            else
            {
                if (immediate)
                {
                    Car cc = car as Car;
                    if (cc != null)
                        cc.SetImmediate(cc.startPos, cc.startRot, syncFixedDt, 2);
                    else
                        car.RestorePosRot();
                }
                else
                {
                    car.RestorePosRot();
                }

                if (carObj != null)
                {
                    //reset last controls
                    car.RequestSteering(0.0f);
                    car.RequestThrottle(0.0f);
                    car.RequestFootBrake(10.0f);

                    // Reset closest point of car path
                    if (pm)
                        iActiveSpan = 0;
                }

                lapCount = 0;
                lastActiveNode = 0;
            }
        }

        void DoResetToNearest(bool immediate)
        {
            int iNearest = -1;
            Car carComp = car as Car;
            if (carComp != null)
                iNearest = CarRecovery.ResetCarToNearestPathNode(carComp, 0.4f, immediate);

            if (carObj != null)
            {
                //reset last controls
                car.RequestSteering(0.0f);
                car.RequestThrottle(0.0f);
                car.RequestFootBrake(10.0f);

                // 遥测的最近路径点跟随复位目标,保持 cross-track 误差连续
                if (pm && iNearest >= 0)
                    iActiveSpan = iNearest;
            }

            Debug.Log("[TcpCarHandler] reset_to_nearest: node " + iNearest);
        }

        // ---- 同步步进(step_mode=synchronous) ----
        // 进入:timeScale=0 + Physics.autoSimulation=false,FixedUpdate 不再运行;
        // 每条 control 在主线程推进 time_step 秒物理,等相机回读完成后回一帧遥测。
        // 物理是实例全局的,单 sim 实例只支持一个 RL 客户端(见进入时的警告日志)。
        void SyncUpdate()
        {
            if (bResetCar)
            {
                DoResetCar(true);
                bResetCar = false;
                SyncPostReset();
                return;
            }

            if (bResetToNearest)
            {
                DoResetToNearest(true);
                bResetToNearest = false;
                SyncPostReset();
                return;
            }

            if (syncCapturePending)
            {
                if (camSensor != null)
                {
                    camSensor.Tick();
                    if (camSensor.LastCompletedFrame >= syncKickedFrame)
                    {
                        syncCapturePending = false;
                        SendTelemetry();
                    }
                }
                else
                {
                    syncCapturePending = false;
                    SendTelemetry();
                }
                return;
            }

            if (pendingSteps > 0)
            {
                System.Threading.Interlocked.Decrement(ref pendingSteps);
                SyncStep();
            }
        }

        void SyncStep()
        {
            if (carComp0 != null)
                carComp0.ApplyRequests();

            int k = Mathf.Max(1, Mathf.RoundToInt(time_step / syncFixedDt));
            for (int i = 0; i < k; i++)
                Physics.Simulate(syncFixedDt);

            if (carComp0 != null)
                carComp0.SyncKinematics(k * syncFixedDt);

            KickSyncCapture();
        }

        void SyncPostReset()
        {
            if (carComp0 != null)
                carComp0.ApplyRequests();

            Physics.Simulate(syncFixedDt);
            Physics.Simulate(syncFixedDt);

            if (carComp0 != null)
                carComp0.SyncKinematics(2f * syncFixedDt);

            KickSyncCapture();
        }

        void KickSyncCapture()
        {
            if (camSensor != null)
            {
                camSensor.KickReadbackNow();
                syncKickedFrame = camSensor.KickCount;
                syncCapturePending = true;
            }
            else
            {
                SendTelemetry();
            }
        }

        void Update()
        {
            // 遥测发送节流放渲染循环:FixedUpdate 物理 100Hz、limitFPS 提到 60
            // 也发不满(相位错开);Update 跟随渲染帧率。帧率优先级:
            // --telemetry_fps(服务端强制,二分用) > cam_config fps 键 > 默认 20。
            if (state != State.SendTelemetry)
                return;

            if (!asynchronous)
            {
                SyncUpdate();
                return;
            }

            if (controlTimeoutMs > 0f && !deadmanTriggered
                && (DateTime.UtcNow - lastControlRecvTime).TotalMilliseconds > controlTimeoutMs)
            {
                car.RequestSteering(0.0f);
                car.RequestThrottle(0.0f);
                car.RequestFootBrake(1.0f);
                deadmanTriggered = true;
            }

            if (camSensor != null)
                camSensor.Tick();
            if (camSensorB != null)
                camSensorB.Tick();

            float effFps = GlobalState.telemetry_fps > 0f ? GlobalState.telemetry_fps : limitFPS;
            timeSinceLastCapture += Time.deltaTime;
            if (timeSinceLastCapture >= 1.0f / effFps)
            {
                timeSinceLastCapture -= (1.0f / effFps);
                SendTelemetry();
                NoteSend();
            }

            TelemetryStats();
        }

        // ---- 遥测自检仪表(二分实验用,每 5s 一行) ----
        float _statSendEma = 20f;   // 发送间隔倒数 EMA(实测推流 fps)
        float _statFrameEma = 1f / 60f; // 主线程帧间隔 EMA(渲染 fps)
        float _statLastSendAt = -1f;
        float _statNextReportAt = 5f;
        int _statSent;

        void NoteSend()
        {
            float now = Time.unscaledTime;
            _statSent++;
            if (_statLastSendAt >= 0f)
            {
                float gap = now - _statLastSendAt;
                if (gap > 0.0001f)
                    _statSendEma = Mathf.Lerp(_statSendEma, 1f / gap, 0.05f);
            }
            _statLastSendAt = now;
        }

        void TelemetryStats()
        {
            _statFrameEma = Mathf.Lerp(_statFrameEma, Time.unscaledDeltaTime, 0.05f);
            float now = Time.unscaledTime;
            if (now < _statNextReportAt)
                return;
            _statNextReportAt = now + 5f;
            float capMs = camSensor != null ? camSensor.LastCaptureMs : 0f;
            long txDrops = _txDrops + (client != null ? client.PooledSendDrops : 0);
            Debug.Log(string.Format(
                "[telemetry-stats] send_fps={0:F1} render_fps={1:F1} capture_ms={2:F1} sent={3} tx_drop={4}",
                _statSendEma,
                1f / Mathf.Max(_statFrameEma, 0.00001f),
                capMs,
                _statSent,
                txDrops));
        }

        public bool IsGhostCar()
        {
            if (client == null) { return true; }
            else { return false; }
        }
    }
}
