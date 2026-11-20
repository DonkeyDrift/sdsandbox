using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using tk;
using System.Net;
using System.Net.Sockets;
using System;

[RequireComponent(typeof(tk.TcpServer))]
public class SandboxServer : MonoBehaviour
{
    public string host;
    public int port;

    tk.TcpServer _server = null;

    public GameObject clientTemplateObj = null;
    public Transform spawn_pt;
    public bool spawnCarswClients = true;
    public bool privateAPI = false;
    bool argHost = false;
    bool argPort = false;

    // 9091/9092 各有一个 SandboxServer 实例都会跑 Start;限帧是应用级设置,只应用一次
    static bool fpsCapApplied = false;

    public void CheckCommandLineConnectArgs()
    {
        string[] args = System.Environment.GetCommandLineArgs();


        if (!privateAPI)
        {
            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == "--host")
                {
                    host = args[i + 1];
                    argHost = true;
                }
                else if (args[i] == "--port")
                {
                    port = int.Parse(args[i + 1]);
                    argPort = true;
                }
                else if (args[i] == "--telemetry_fps")
                {
                    GlobalState.telemetry_fps = float.Parse(args[i + 1], System.Globalization.CultureInfo.InvariantCulture.NumberFormat);
                }
                else if (args[i] == "--quality")
                {
                    GlobalState.quality_level = int.Parse(args[i + 1]);
                }
                else if (args[i] == "--vsync")
                {
                    GlobalState.vsync_count = int.Parse(args[i + 1]);
                }
                else if (args[i] == "--max_fps")
                {
                    GlobalState.max_fps = int.Parse(args[i + 1]);
                }
                else if (args[i] == "--rl")
                {
                    GlobalState.rlMode = true;
                }
                else if (args[i] == "--idle_timeout")
                {
                    GlobalState.timeOut = float.Parse(args[i + 1], System.Globalization.CultureInfo.InvariantCulture.NumberFormat);
                }
            }
        }

        if (argHost == false) { host = GlobalState.host; }
        if (argPort == false) { port = GlobalState.port; }
        
        if (privateAPI)
        {
            port = GlobalState.portPrivateAPI;
        }

    }

    private void Awake()
    {
        _server = GetComponent<tk.TcpServer>();
    }

    // Start is called before the first frame update
    void Start()
    {
        CheckCommandLineConnectArgs();

        if (GlobalState.rlMode && !privateAPI)
            Debug.Log("[SandboxServer] rl mode enabled");

        // 画质/垂直同步命令行覆盖(二分开关;工程默认档 5=Ultra,见 QualitySettings)
        if (GlobalState.quality_level >= 0)
        {
            int q = Mathf.Clamp(GlobalState.quality_level, 0, QualitySettings.names.Length - 1);
            QualitySettings.SetQualityLevel(q, true);
            Debug.Log("SDSandbox quality override -> " + q + " (" + QualitySettings.names[q] + ")");
        }
        if (GlobalState.vsync_count >= 0)
        {
            QualitySettings.vSyncCount = GlobalState.vsync_count;
            Debug.Log("SDSandbox vsync override -> " + GlobalState.vsync_count);
        }

        // 前台帧率上限。启动场景是赛道而非 menu(见 EditorBuildSettings 场景序),
        // menu 里 GlobalStateEditor 的 60 帧设置不会运行;工程质量档 vSyncCount=0,
        // 不限帧时空场景可跑数千 fps(渲染空转,物理按 FixedUpdate 固定步长不受影响)。
        // --rl 训练实例跳过,主循环行为保持与历史训练数据一致。
        if (!GlobalState.rlMode && !fpsCapApplied && GlobalState.max_fps > 0)
        {
            fpsCapApplied = true;
            Application.targetFrameRate = GlobalState.max_fps;
            Debug.Log("SDSandbox fps cap -> " + GlobalState.max_fps);
            StartCoroutine(ReportMeasuredFps());
        }
        Debug.Log("SDSandbox Server starting.");
        _server.onClientConntedCB += new tk.TcpServer.OnClientConnected(OnClientConnected);
        _server.onClientDisconntedCB += new tk.TcpServer.OnClientDisconnected(OnClientDisconnected);

        _server.Run(host, port);
    }

    // 限帧生效后的实测回执:跳过启动首秒(场景加载抖动),统计其后 5 秒真实时间的帧数
    IEnumerator ReportMeasuredFps()
    {
        yield return new WaitForSecondsRealtime(1f);
        int frames = 0;
        float start = Time.realtimeSinceStartup;
        while (Time.realtimeSinceStartup - start < 5f) { frames++; yield return null; }
        float elapsed = Time.realtimeSinceStartup - start;
        Debug.Log("SDSandbox measured avg fps over 5s -> " + (frames / elapsed).ToString("F0"));
    }

    // It's our responsibility to create a GameObject with a TcpClient
    // and return it to the server.
    public tk.TcpClient OnClientConnected()
    {
        if (clientTemplateObj == null)
        {
            Debug.LogError("client template object was null.");
            return null;
        }

        if (_server.debug)
            Debug.Log("creating client obj");

        GameObject go = GameObject.Instantiate(clientTemplateObj) as GameObject;

        go.transform.parent = this.transform;

        if (spawn_pt != null)
            go.transform.position = spawn_pt.position + UnityEngine.Random.insideUnitSphere * 2;

        tk.TcpClient client = go.GetComponent<tk.TcpClient>();

        InitClient(client);

        return client;
    }

    private void InitClient(tk.TcpClient client)
    {
        if (privateAPI) // private API client server
        {
            PrivateAPI privateAPIHandler = GameObject.FindObjectOfType<PrivateAPI>();
            if (privateAPIHandler != null)
            {
                if (_server.debug)
                    Debug.Log("init private API handler.");

                privateAPIHandler.Init(client.gameObject.GetComponent<tk.JsonTcpClient>());
            }
        }

        else // normal client server
        {
            if (spawnCarswClients) // we are on in a track scene
            {
                CarSpawner spawner = GameObject.FindObjectOfType<CarSpawner>();
                if (spawner != null)
                {
                    if (_server.debug)
                        Debug.Log("spawning car.");

                    spawner.Spawn(client.gameObject.GetComponent<tk.JsonTcpClient>(), false);
                }
            }
            else //we are in the menu
            {

                tk.TcpMenuHandler handler = GameObject.FindObjectOfType<TcpMenuHandler>();
                if (handler != null)
                {
                    if (_server.debug)
                        Debug.Log("init menu handler.");

                    handler.Init(client.gameObject.GetComponent<tk.JsonTcpClient>());
                }
            }
        }
    }


    public void OnSceneLoaded(bool bFrontEnd)
    {
        spawnCarswClients = !bFrontEnd;

        List<tk.TcpClient> clients = _server.GetClients();

        foreach (tk.TcpClient client in clients)
        {
            if (_server.debug)
                Debug.Log("init network client.");

            InitClient(client);
        }

        if (GlobalState.paceCar && !bFrontEnd) // && clients.Count == 0
        {
            CarSpawner spawner = GameObject.FindObjectOfType<CarSpawner>();
            if (spawner)
            {
                spawner.EnsureOneCar();
            }
        }
    }

    public void OnClientDisconnected(tk.TcpClient client)
    {
        if (privateAPI)
        {
        }
        else
        {
            CarSpawner spawner = GameObject.FindObjectOfType<CarSpawner>();
            if (spawner)
            {
                spawner.RemoveCar(client.gameObject.GetComponent<tk.JsonTcpClient>());
            }
        }
        GameObject.Destroy(client.gameObject);
    }
}
