using System;
using System.Runtime.InteropServices;
using UnityEngine;

namespace DonkeySimUpdate
{
    public static class MacTitleBarHider
    {
        private const string ObjCLibrary = "/usr/lib/libobjc.dylib";

        [DllImport(ObjCLibrary, EntryPoint = "objc_getClass", CharSet = CharSet.Ansi)]
        private static extern IntPtr ObjcGetClass(string name);

        [DllImport(ObjCLibrary, EntryPoint = "sel_registerName", CharSet = CharSet.Ansi)]
        private static extern IntPtr SelRegisterName(string name);

        [DllImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
        private static extern IntPtr SendMessage(IntPtr receiver, IntPtr selector);

        [DllImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
        private static extern ulong SendMessageUlong(IntPtr receiver, IntPtr selector);

        [DllImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
        private static extern IntPtr SendMessageIndex(IntPtr receiver, IntPtr selector, ulong index);

        [DllImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
        private static extern void SendMessageUlongVoid(IntPtr receiver, IntPtr selector, ulong value);

        [DllImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
        private static extern void SendMessageBool(IntPtr receiver, IntPtr selector, [MarshalAs(UnmanagedType.I1)] bool value);

        [DllImport(ObjCLibrary, EntryPoint = "objc_msgSend")]
        private static extern bool SendMessageBoolResult(IntPtr receiver, IntPtr selector);

        private const ulong Titled = 1UL << 0;
        private const ulong FullSizeContentView = 1UL << 15;
        private const ulong NSWindowTitleHidden = 1UL;

        private static readonly bool DeminiaturizeForTests =
            Environment.GetEnvironmentVariable("MAC_TITLEBAR_DEMINIATURIZE") == "1";

        private static TitleBarReapplier reapplier;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        public static void Apply()
        {
            if (Application.platform != RuntimePlatform.OSXPlayer)
            {
                return;
            }

            if (reapplier != null)
            {
                return;
            }

            GameObject gameObject = new GameObject("Mac Title Bar Hider");
            UnityEngine.Object.DontDestroyOnLoad(gameObject);
            reapplier = gameObject.AddComponent<TitleBarReapplier>();
            reapplier.ApplyNow();
        }

        private static int ApplyToWindows(bool report)
        {
            int hidden = 0;
            try
            {
                IntPtr application = SendMessage(ObjcGetClass("NSApplication"), SelRegisterName("sharedApplication"));
                if (application == IntPtr.Zero)
                {
                    if (report)
                    {
                        Debug.LogWarning("MacTitleBarHider: NSApplication.sharedApplication returned null");
                    }

                    return 0;
                }

                IntPtr windows = SendMessage(application, SelRegisterName("windows"));
                if (windows == IntPtr.Zero)
                {
                    if (report)
                    {
                        Debug.LogWarning("MacTitleBarHider: NSApplication.windows returned null");
                    }

                    return 0;
                }

                ulong count = SendMessageUlong(windows, SelRegisterName("count"));
                if (count == 0)
                {
                    if (report)
                    {
                        Debug.LogWarning("MacTitleBarHider: no windows, title bar not hidden");
                    }

                    return 0;
                }

                IntPtr objectAtIndexSelector = SelRegisterName("objectAtIndex:");
                IntPtr styleMaskSelector = SelRegisterName("styleMask");
                IntPtr titleVisibilitySelector = SelRegisterName("titleVisibility");
                IntPtr setStyleMaskSelector = SelRegisterName("setStyleMask:");
                IntPtr setTitlebarAppearsTransparentSelector = SelRegisterName("setTitlebarAppearsTransparent:");
                IntPtr setTitleVisibilitySelector = SelRegisterName("setTitleVisibility:");

                for (ulong i = 0; i < count; i++)
                {
                    IntPtr window = SendMessageIndex(windows, objectAtIndexSelector, i);
                    if (window == IntPtr.Zero)
                    {
                        if (report)
                        {
                            Debug.LogWarning("MacTitleBarHider: window " + i + " is null, skipped");
                        }

                        continue;
                    }

                    ulong mask = SendMessageUlong(window, styleMaskSelector);
                    if ((mask & Titled) == 0)
                    {
                        if (report)
                        {
                            Debug.Log("MacTitleBarHider: window " + i + " has no title bar (styleMask=0x" + mask.ToString("X") + "), skipped");
                        }

                        continue;
                    }

                    bool needsFix = (mask & FullSizeContentView) == 0;
                    ulong titleVisibility = SendMessageUlong(window, titleVisibilitySelector);
                    if (titleVisibility != NSWindowTitleHidden)
                    {
                        needsFix = true;
                    }

                    if (needsFix)
                    {
                        if ((mask & FullSizeContentView) == 0)
                        {
                            SendMessageUlongVoid(window, setStyleMaskSelector, mask | FullSizeContentView);
                        }

                        SendMessageBool(window, setTitlebarAppearsTransparentSelector, true);
                        SendMessageUlongVoid(window, setTitleVisibilitySelector, NSWindowTitleHidden);
                        Debug.Log(
                            "MacTitleBarHider: hid title bar on window " + i +
                            " (styleMask=0x" + mask.ToString("X") + ", titleVisibility=" + titleVisibility + ")");
                    }

                    if (DeminiaturizeForTests &&
                        SendMessageBoolResult(window, SelRegisterName("isMiniaturized")))
                    {
                        SendMessageUlongVoid(window, SelRegisterName("deminiaturize:"), 0);
                        if (report)
                        {
                            Debug.Log("MacTitleBarHider: deminiaturized window " + i);
                        }
                    }

                    hidden++;
                }
            }
            catch (Exception exception)
            {
                Debug.LogWarning("MacTitleBarHider: failed: " + exception.Message);
            }

            return hidden;
        }

        private sealed class TitleBarReapplier : MonoBehaviour
        {
            private int frames;

            public void ApplyNow()
            {
                int hidden = ApplyToWindows(true);
                Debug.Log("MacTitleBarHider: applied to " + hidden + " window(s)");
            }

            private void Update()
            {
                frames++;
                if (frames % 30 != 0)
                {
                    return;
                }

                ApplyToWindows(false);
            }
        }
    }
}
