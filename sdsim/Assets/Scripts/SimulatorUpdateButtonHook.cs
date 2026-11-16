using System;
using System.Diagnostics;
using UnityEngine;
using UnityEngine.SceneManagement;
using Debug = UnityEngine.Debug;

namespace DonkeySimUpdate
{
    public static class SimulatorUpdateButtonHook
    {
        private const string LatestReleaseUrl = "https://github.com/tawnkramer/gym-donkeycar/releases/latest";
        private const string UpdaterModule = "gym_donkeycar.simulator_update";
        private const string ButtonLabel = "Update & Restart";
        private static SimulatorUpdateButton button;

        public static void Install(object menuHandler)
        {
            if (button != null)
            {
                return;
            }

            GameObject gameObject = new GameObject("Simulator Update Button");
            button = gameObject.AddComponent<SimulatorUpdateButton>();
            UnityEngine.Object.DontDestroyOnLoad(gameObject);
        }

        private static void UpdateAndRestart()
        {
            string pid = Process.GetCurrentProcess().Id.ToString();
            string arguments = "update --apply-after-exit --pid " + pid + " --restart";
            if (StartUpdater(arguments))
            {
                Application.Quit();
                return;
            }

            Debug.LogError(
                "Simulator updater not found. Install gym_donkeycar, which provides donkey-sim-update, or update manually.");
            Application.OpenURL(LatestReleaseUrl);
        }

        private static bool StartUpdater(string arguments)
        {
            string customUpdater = Environment.GetEnvironmentVariable("DONKEY_SIM_UPDATER");
            if (!string.IsNullOrEmpty(customUpdater) && System.IO.File.Exists(customUpdater))
            {
                return ShellStart(customUpdater, arguments);
            }

            if (Application.platform == RuntimePlatform.WindowsPlayer)
            {
                if (CommandExists("cmd", "/c where donkey-sim-update"))
                {
                    return ShellStart("cmd", "/c donkey-sim-update " + arguments);
                }

                if (CommandExists("cmd", "/c where python"))
                {
                    return ShellStart("cmd", "/c python -m " + UpdaterModule + " " + arguments);
                }

                return false;
            }

            string home = Environment.GetEnvironmentVariable("HOME");
            string[] candidates =
            {
                "/opt/homebrew/bin/donkey-sim-update",
                "/usr/local/bin/donkey-sim-update",
                "/usr/bin/donkey-sim-update",
                string.IsNullOrEmpty(home) ? null : home + "/.local/bin/donkey-sim-update"
            };

            foreach (string candidate in candidates)
            {
                if (!string.IsNullOrEmpty(candidate) && System.IO.File.Exists(candidate))
                {
                    return ShellStart(candidate, arguments);
                }
            }

            if (CommandExists("donkey-sim-update", "--help"))
            {
                return ShellStart("donkey-sim-update", arguments);
            }

            string[] pythonCandidates = { "/opt/homebrew/bin/python3", "/usr/local/bin/python3", "/usr/bin/python3", "python3" };
            foreach (string python in pythonCandidates)
            {
                if (CommandExists(python, "-m " + UpdaterModule + " --help"))
                {
                    return ShellStart(python, "-m " + UpdaterModule + " " + arguments);
                }
            }

            return false;
        }

        private static bool ShellStart(string fileName, string arguments)
        {
            try
            {
                ProcessStartInfo startInfo = new ProcessStartInfo();
                startInfo.FileName = fileName;
                startInfo.Arguments = arguments;
                startInfo.UseShellExecute = false;
                startInfo.CreateNoWindow = true;
                Process process = Process.Start(startInfo);
                if (process == null)
                {
                    return false;
                }

                Debug.Log("Started simulator updater: " + fileName + " " + arguments);
                return true;
            }
            catch (Exception exception)
            {
                Debug.LogWarning("Failed to start simulator updater (" + fileName + "): " + exception.Message);
                return false;
            }
        }

        private static bool CommandExists(string fileName, string arguments)
        {
            try
            {
                ProcessStartInfo startInfo = new ProcessStartInfo();
                startInfo.FileName = fileName;
                startInfo.Arguments = arguments;
                startInfo.UseShellExecute = false;
                startInfo.CreateNoWindow = true;
                Process process = Process.Start(startInfo);
                if (process == null || !process.WaitForExit(3000))
                {
                    if (process != null)
                    {
                        process.Kill();
                    }

                    return false;
                }

                return process.ExitCode == 0;
            }
            catch (Exception exception)
            {
                Debug.LogWarning("Simulator updater probe failed (" + fileName + " " + arguments + "): " + exception.Message);
                return false;
            }
        }

        private sealed class SimulatorUpdateButton : MonoBehaviour
        {
            private GUIStyle buttonStyle;

            private void OnGUI()
            {
                Scene scene = SceneManager.GetActiveScene();
                if (scene.buildIndex != 0 && scene.name != "menu")
                {
                    return;
                }

                if (buttonStyle == null)
                {
                    buttonStyle = new GUIStyle(GUI.skin.button);
                    buttonStyle.fontSize = 22;
                    buttonStyle.fontStyle = FontStyle.Bold;
                    buttonStyle.alignment = TextAnchor.MiddleCenter;
                }

                float width = Mathf.Max(320.0f, Screen.width * 0.22f);
                float height = Mathf.Max(46.0f, Screen.height * 0.042f);
                Rect rect = new Rect((Screen.width - width) * 0.5f, Screen.height - height - 24.0f, width, height);
                if (GUI.Button(rect, ButtonLabel, buttonStyle))
                {
                    UpdateAndRestart();
                }
            }
        }
    }
}
