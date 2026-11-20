using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace DonkeySimUpdate
{
    /// App-icon appearance switcher for the Donkey simulator (macOS only).
    ///
    /// Renders a small settings row (Auto / Light / Dark, persisted via
    /// PlayerPrefs) in the window's top-right corner and drives the running
    /// app's Dock tile through the Objective-C runtime:
    ///   - "auto" polls NSApplication.effectiveAppearance every two seconds
    ///     and re-applies the matching artwork on every flip;
    ///   - "light"/"dark" pin the matching artwork and stop polling.
    /// Artwork files live next to the bundle catalog the packager injects:
    ///   donkey_sim.app/Contents/Resources/icon-day.png / icon-night.png
    /// (light = the shipped white-background artwork, dark = the same
    /// subject on pure #000000, both rounded to the macOS icon squircle).
    /// The Finder/Launchpad icon is served by the bundle's dual-appearance
    /// Assets.car and always follows the OS; this component only controls
    /// the RUNNING app's Dock tile.
    ///
    /// Implementation notes (2026-11-19, all verified on the shipped
    /// 2022.3.49 player base):
    ///   - The base strips the IMGUI event pipeline: OnGUI is never invoked
    ///     for any component (also why the update button never shows there).
    ///     The row is therefore programmatic uGUI with manual RectTransforms
    ///     (LayoutGroup sizing proved unreliable across uGUI versions).
    ///   - The base never runs tk.TcpMenuHandler.Start; activation comes
    ///     from the packager's Cecil injection into CarRecovery.Update
    ///     (idempotent Install, verified firing every frame).
    ///   - Icon paths anchor on this DLL's own location (Managed/), not
    ///     Application.dataPath (its Mac value varies by Unity version).
    ///
    /// Everything is best-effort: missing artwork, native failures, or a
    /// non-macOS player leave the bundle icon untouched.
    public static class SimulatorAppIconHook
    {
        private const string ModeKey = "DonkeySim.AppIconMode";
        private const string DayIconName = "icon-day.png";
        private const string NightIconName = "icon-night.png";
        private const float AutoPollSeconds = 2.0f;

        public static void Install(object menuHandler)
        {
            SimulatorAppIconSwitcher.EnsureCreated();
        }

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void InstallDefault()
        {
            SimulatorAppIconSwitcher.EnsureCreated();
        }

        private sealed class SimulatorAppIconSwitcher : MonoBehaviour
        {
            private static SimulatorAppIconSwitcher instance;
            private static string appliedIconPath;

            private float nextPollTime;
            private string styledMode;
            private readonly Dictionary<string, Image> buttonBackgrounds = new Dictionary<string, Image>();

            public static void EnsureCreated()
            {
                if (instance != null)
                {
                    return;
                }

                if (Application.platform != RuntimePlatform.OSXPlayer)
                {
                    return;
                }

                GameObject gameObject = new GameObject("Simulator App Icon Switcher");
                instance = gameObject.AddComponent<SimulatorAppIconSwitcher>();
                UnityEngine.Object.DontDestroyOnLoad(gameObject);
            }

            private static string ReadMode()
            {
                string mode = PlayerPrefs.GetString(ModeKey, "auto");
                return mode == "light" || mode == "dark" ? mode : "auto";
            }

            private static string IconPath(bool dark)
            {
                try
                {
                    string managed = Path.GetDirectoryName(typeof(SimulatorAppIconHook).Assembly.Location);
                    string resources = Path.GetFullPath(Path.Combine(managed, "..", ".."));
                    return Path.Combine(resources, dark ? NightIconName : DayIconName);
                }
                catch (Exception exception)
                {
                    Debug.LogWarning("App icon: cannot resolve Resources path (" + exception.Message + ")");
                    return null;
                }
            }

            private static bool SystemIsDark()
            {
                try
                {
                    string name = MacAppIconBridge.EffectiveAppearanceName();
                    return !string.IsNullOrEmpty(name) && name.StartsWith("NSAppearanceNameDark", StringComparison.Ordinal);
                }
                catch (Exception exception)
                {
                    Debug.LogWarning("App icon: appearance probe failed (" + exception.Message + ")");
                    return false;
                }
            }

            private static void ApplyMode(string mode)
            {
                bool dark = mode == "dark" || (mode == "auto" && SystemIsDark());
                string path = IconPath(dark);
                if (path == null)
                {
                    return;
                }

                if (!File.Exists(path))
                {
                    Debug.LogWarning("App icon: artwork missing at " + path);
                    return;
                }

                if (appliedIconPath == path)
                {
                    return;
                }

                if (MacAppIconBridge.SetApplicationIcon(path))
                {
                    appliedIconPath = path;
                    Debug.Log("App icon applied: " + Path.GetFileName(path) + " (mode " + mode + ")");
                }
            }

            private void SetMode(string mode)
            {
                PlayerPrefs.SetString(ModeKey, mode);
                PlayerPrefs.Save();
                appliedIconPath = null;
                nextPollTime = 0.0f;
                ApplyMode(mode);
                Restyle(mode);
            }

            private void Start()
            {
                ApplyMode(ReadMode());
                BuildUi();
            }

            private bool hintLogged;

            private void Update()
            {
                // 键盘设置入口：I 键循环 Auto → Light → Dark（IMGUI 在此基座被裁、
                // uGUI 行在某些渲染管线不可见时，这是保底可用且反馈即时的入口——
                // 每次切换 Dock 磁贴当场变色并写日志）。
                if (Input.GetKeyDown(KeyCode.I))
                {
                    string current = ReadMode();
                    string next = current == "auto" ? "light" : current == "light" ? "dark" : "auto";
                    SetMode(next);
                    Debug.Log("App icon mode -> " + next);
                }

                if (!hintLogged)
                {
                    hintLogged = true;
                    Debug.Log("App icon: press I to cycle Auto/Light/Dark (current " + ReadMode() + ")");
                }

                string mode = ReadMode();
                if (styledMode != mode && buttonBackgrounds.Count > 0)
                {
                    Restyle(mode);
                }

                if (mode != "auto" || Time.time < nextPollTime)
                {
                    return;
                }

                nextPollTime = Time.time + AutoPollSeconds;
                ApplyMode("auto");
            }

            private void BuildUi()
            {
                try
                {
                    Canvas canvas = gameObject.AddComponent<Canvas>();
                    canvas.renderMode = RenderMode.ScreenSpaceOverlay;
                    canvas.sortingOrder = 32767;
                    CanvasScaler scaler = gameObject.AddComponent<CanvasScaler>();
                    scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
                    scaler.referenceResolution = new Vector2(1280.0f, 720.0f);
                    gameObject.AddComponent<GraphicRaycaster>();

                    if (UnityEngine.Object.FindObjectOfType<EventSystem>() == null)
                    {
                        GameObject system = new GameObject("AppIcon EventSystem");
                        system.AddComponent<EventSystem>();
                        system.AddComponent<StandaloneInputModule>();
                        UnityEngine.Object.DontDestroyOnLoad(system);
                    }

                    Font font = LoadFont();

                    GameObject row = new GameObject("AppIconRow", typeof(RectTransform));
                    row.transform.SetParent(canvas.transform, false);
                    RectTransform rowRect = (RectTransform)row.transform;
                    rowRect.anchorMin = rowRect.anchorMax = new Vector2(1.0f, 1.0f);
                    rowRect.pivot = new Vector2(1.0f, 1.0f);
                    rowRect.anchoredPosition = new Vector2(-16.0f, -12.0f);
                    rowRect.sizeDelta = new Vector2(360.0f, 34.0f);

                    // Manual absolute layout — no LayoutGroups (their sizing
                    // behavior differs across the uGUI versions shipped in
                    // these player builds).
                    MakeLabel(row, font, "App icon:", 0.0f, 80.0f);
                    MakeButton(row, font, "Auto", "auto", 86.0f, 74.0f);
                    MakeButton(row, font, "Light", "light", 168.0f, 84.0f);
                    MakeButton(row, font, "Dark", "dark", 260.0f, 74.0f);
                    Restyle(ReadMode());
                    Debug.Log("AppIcon uGUI row built; canvas enabled=" + canvas.enabled
                        + " activeInHierarchy=" + canvas.gameObject.activeInHierarchy
                        + " isRoot=" + canvas.isRootCanvas
                        + " pixelRect=" + canvas.pixelRect
                        + " renderMode=" + canvas.renderMode
                        + " scaleFactor=" + (canvas.transform as RectTransform).localScale
                        + " rowPos=" + rowRect.anchoredPosition
                        + " rowSize=" + rowRect.rect.size
                        + " rowWorld=" + rowRect.position
                        + " children=" + rowRect.childCount);
                }
                catch (Exception exception)
                {
                    Debug.LogWarning("App icon: UI build failed (" + exception.Message + ")");
                }
            }

            private static Font LoadFont()
            {
                try
                {
                    return Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
                }
                catch
                {
                    // Unity 2020 and older name the builtin font Arial
                }

                try
                {
                    return Resources.GetBuiltinResource<Font>("Arial.ttf");
                }
                catch
                {
                    return null;
                }
            }

            private void MakeLabel(GameObject parent, Font font, string text, float x, float width)
            {
                GameObject go = new GameObject("label", typeof(RectTransform));
                go.transform.SetParent(parent.transform, false);
                RectTransform rect = (RectTransform)go.transform;
                rect.anchorMin = rect.anchorMax = new Vector2(0.0f, 0.5f);
                rect.pivot = new Vector2(0.0f, 0.5f);
                rect.anchoredPosition = new Vector2(x, 0.0f);
                rect.sizeDelta = new Vector2(width, 34.0f);
                Text label = go.AddComponent<Text>();
                label.text = text;
                label.font = font;
                label.fontSize = 18;
                label.alignment = TextAnchor.MiddleRight;
                label.color = Color.white;
            }

            private void MakeButton(GameObject parent, Font font, string text, string mode, float x, float width)
            {
                GameObject go = new GameObject("btn-" + mode, typeof(RectTransform));
                go.transform.SetParent(parent.transform, false);
                RectTransform rect = (RectTransform)go.transform;
                rect.anchorMin = rect.anchorMax = new Vector2(0.0f, 0.5f);
                rect.pivot = new Vector2(0.0f, 0.5f);
                rect.anchoredPosition = new Vector2(x, 0.0f);
                rect.sizeDelta = new Vector2(width, 30.0f);
                Image background = go.AddComponent<Image>();
                background.color = new Color(0.85f, 0.85f, 0.85f, 0.9f);
                Button button = go.AddComponent<Button>();
                button.targetGraphic = background;
                string captured = mode;
                button.onClick.AddListener(delegate { SetMode(captured); });

                GameObject textGo = new GameObject("text", typeof(RectTransform));
                textGo.transform.SetParent(go.transform, false);
                Text label = textGo.AddComponent<Text>();
                label.text = text;
                label.font = font;
                label.fontSize = 16;
                label.alignment = TextAnchor.MiddleCenter;
                label.color = Color.black;
                Stretch(textGo);

                buttonBackgrounds[mode] = background;
            }

            private static void Stretch(GameObject go)
            {
                RectTransform rect = (RectTransform)go.transform;
                rect.anchorMin = Vector2.zero;
                rect.anchorMax = Vector2.one;
                rect.offsetMin = Vector2.zero;
                rect.offsetMax = Vector2.zero;
            }

            private void Restyle(string mode)
            {
                styledMode = mode;
                foreach (KeyValuePair<string, Image> pair in buttonBackgrounds)
                {
                    pair.Value.color = pair.Key == mode
                        ? new Color(0.35f, 0.72f, 1.0f, 0.95f)
                        : new Color(0.85f, 0.85f, 0.85f, 0.9f);
                }
            }
        }

        /// Minimal ObjC bridge: enough of the runtime to read the effective
        /// appearance name and to push an NSImage onto the Dock tile. All
        /// imports are lazy; none of this type is touched off macOS.
        private static class MacAppIconBridge
        {
            [DllImport("__Internal")]
            private static extern IntPtr objc_getClass(string name);

            [DllImport("__Internal")]
            private static extern IntPtr sel_registerName(string name);

            [DllImport("__Internal", EntryPoint = "objc_msgSend")]
            private static extern IntPtr Object_msgSend(IntPtr receiver, IntPtr selector);

            [DllImport("__Internal", EntryPoint = "objc_msgSend")]
            private static extern IntPtr IntPtr_msgSend_IntPtr(IntPtr receiver, IntPtr selector, IntPtr argument);

            [DllImport("__Internal", EntryPoint = "objc_msgSend")]
            private static extern void Void_msgSend_Object(IntPtr receiver, IntPtr selector, IntPtr argument);

            public static string EffectiveAppearanceName()
            {
                IntPtr application = sharedApplication();
                if (application == IntPtr.Zero)
                {
                    return null;
                }

                IntPtr appearance = Object_msgSend(application, Sel("effectiveAppearance"));
                if (appearance == IntPtr.Zero)
                {
                    return null;
                }

                IntPtr nameString = Object_msgSend(appearance, Sel("name"));
                if (nameString == IntPtr.Zero)
                {
                    return null;
                }

                IntPtr utf8 = Object_msgSend(nameString, Sel("UTF8String"));
                return utf8 == IntPtr.Zero ? null : Marshal.PtrToStringAnsi(utf8);
            }

            public static bool SetApplicationIcon(string path)
            {
                IntPtr application = sharedApplication();
                IntPtr imageClass = objc_getClass("NSImage");
                if (application == IntPtr.Zero || imageClass == IntPtr.Zero)
                {
                    return false;
                }

                IntPtr image = Object_msgSend(imageClass, Sel("alloc"));
                if (image == IntPtr.Zero)
                {
                    return false;
                }

                IntPtr pathPointer = Marshal.StringToHGlobalAnsi(path);
                try
                {
                    IntPtr pathString = IntPtr_msgSend_IntPtr(
                        objc_getClass("NSString"), Sel("stringWithUTF8String:"), pathPointer);
                    if (pathString == IntPtr.Zero)
                    {
                        return false;
                    }

                    IntPtr initialized = IntPtr_msgSend_IntPtr(image, Sel("initWithContentsOfFile:"), pathString);
                    if (initialized == IntPtr.Zero)
                    {
                        return false;
                    }

                    Void_msgSend_Object(application, Sel("setApplicationIconImage:"), initialized);
                    return true;
                }
                finally
                {
                    Marshal.FreeHGlobal(pathPointer);
                }
            }

            private static IntPtr sharedApplication()
            {
                IntPtr applicationClass = objc_getClass("NSApplication");
                return applicationClass == IntPtr.Zero
                    ? IntPtr.Zero
                    : Object_msgSend(applicationClass, Sel("sharedApplication"));
            }

            private static IntPtr Sel(string name)
            {
                return sel_registerName(name);
            }
        }
    }
}
