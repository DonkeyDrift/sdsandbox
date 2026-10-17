using System.Collections.Generic;
using UnityEngine;


public static class GlobalState
{
    public static string version = "v25.06.10";
    public static string host = "0.0.0.0";
    public static int port = 9091;
    public static int portPrivateAPI = 9092;
    public static int fps = 60;
    public static float timeScale = 1.0f;
    public static int maxSplitScreen = 4;
    public static bool bAutoHideSceneMenu = false;

    // should we create a car even though we don't have a network client?
    public static bool bCreateCarWithoutNetworkClient = false;
    public static string log_path = "default";
    public static bool extendedTelemetry = true;
    public static bool generateTrees = true;
    public static bool generateRandomCones = true;
    public static bool randomLight = true;
    public static bool overheadCamera = false;
    public static bool raceCameras = false;
    public static bool paceCar = false;
    public static bool manualDriving = false;
    public static float kp = 5.0f;
    public static float kd = 5.0f;
    public static float ki = 0.0f;
    public static string privateKey = "";
    public static bool useSeed = false;
    public static int seed = 20432814;
    public static string additionnalContentPath = "";
    public static string[] sceneNames;
    public static List<AssetBundle> bundleScenes = new List<AssetBundle>();
    public static bool drawLidar = true;
    public static float timeOut = 300f;

    // ---- 遥测二分开关(命令行,--host/--port 同款,见 SandboxServer) ----
    // telemetry_fps: >0 时服务端强制遥测推流帧率(cam_config fps 键的 CLI 上位开关,
    // 二分实验用;0=按 cam_config/默认 20)
    public static float telemetry_fps = 0f;
    // quality_level/vsync_count: >=0 时启动即覆盖画质档/垂直同步(-1=不动,用工程默认)
    public static int quality_level = -1;
    public static int vsync_count = -1;
    public static bool rlMode = false;
}
