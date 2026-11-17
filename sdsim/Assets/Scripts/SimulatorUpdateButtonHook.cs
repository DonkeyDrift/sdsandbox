using System;
using System.Diagnostics;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
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

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void InstallDefault()
        {
            Install(null);
        }

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
            string fileName;
            string argumentPrefix;
            if (!TryFindUpdater(out fileName, out argumentPrefix))
            {
                return false;
            }

            return ShellStart(fileName, argumentPrefix + arguments);
        }

        private static bool TryFindUpdater(out string fileName, out string argumentPrefix)
        {
            fileName = null;
            argumentPrefix = "";

            string customUpdater = Environment.GetEnvironmentVariable("DONKEY_SIM_UPDATER");
            if (!string.IsNullOrEmpty(customUpdater) && System.IO.File.Exists(customUpdater))
            {
                fileName = customUpdater;
                argumentPrefix = "";
                return true;
            }

            if (Application.platform == RuntimePlatform.WindowsPlayer)
            {
                if (CommandExists("cmd", "/c where donkey-sim-update"))
                {
                    fileName = "cmd";
                    argumentPrefix = "/c donkey-sim-update ";
                    return true;
                }

                if (CommandExists("cmd", "/c where python"))
                {
                    fileName = "cmd";
                    argumentPrefix = "/c python -m " + UpdaterModule + " ";
                    return true;
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
                    fileName = candidate;
                    argumentPrefix = "";
                    return true;
                }
            }

            if (CommandExists("donkey-sim-update", "--help"))
            {
                fileName = "donkey-sim-update";
                argumentPrefix = "";
                return true;
            }

            string[] pythonCandidates = { "/opt/homebrew/bin/python3", "/usr/local/bin/python3", "/usr/bin/python3", "python3" };
            foreach (string python in pythonCandidates)
            {
                if (CommandExists(python, "-m " + UpdaterModule + " --help"))
                {
                    fileName = python;
                    argumentPrefix = "-m " + UpdaterModule + " ";
                    return true;
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
            private const int CheckTimeoutMilliseconds = 30000;

            private GUIStyle buttonStyle;
            private bool updateCheckStarted;
            private volatile bool updateAvailable;
            private bool testInvoked;

            private void Update()
            {
                if (IsMenuScene(SceneManager.GetActiveScene()))
                {
                    StartUpdateCheck();
                }

                if (!testInvoked &&
                    Environment.GetEnvironmentVariable("DONKEY_SIM_UPDATE_TEST") == "1" &&
                    Time.frameCount >= 10)
                {
                    testInvoked = true;
                    UpdateAndRestart();
                }
            }

            private static bool IsMenuScene(Scene scene)
            {
                return scene.buildIndex == 0 || scene.name == "menu";
            }

            private void StartUpdateCheck()
            {
                if (updateCheckStarted)
                {
                    return;
                }

                updateCheckStarted = true;
                ThreadPool.QueueUserWorkItem(state => RunUpdateCheck());
            }

            private void RunUpdateCheck()
            {
                string fileName;
                string argumentPrefix;
                if (!TryFindUpdater(out fileName, out argumentPrefix))
                {
                    return;
                }

                try
                {
                    using (Process process = CreateUpdateCheckProcess(fileName, argumentPrefix))
                    {
                        process.Start();
                        Task<string> outputTask = process.StandardOutput.ReadToEndAsync();
                        if (!process.WaitForExit(CheckTimeoutMilliseconds))
                        {
                            process.Kill();
                            return;
                        }

                        string output = outputTask.Result;
                        if (process.ExitCode != 0)
                        {
                            return;
                        }

                        Match match = Regex.Match(
                            output,
                            "\"update_available\"\\s*:\\s*(true|false)");
                        if (!match.Success)
                        {
                            return;
                        }

                        updateAvailable = match.Groups[1].Value == "true";
                    }
                }
                catch (Exception)
                {
                    updateAvailable = false;
                }
            }

            private static Process CreateUpdateCheckProcess(string fileName, string argumentPrefix)
            {
                ProcessStartInfo startInfo = new ProcessStartInfo();
                startInfo.FileName = fileName;
                startInfo.Arguments = argumentPrefix + "check --json";
                startInfo.UseShellExecute = false;
                startInfo.CreateNoWindow = true;
                startInfo.RedirectStandardOutput = true;
                return new Process { StartInfo = startInfo };
            }

            private void OnGUI()
            {
                if (!updateAvailable || !IsMenuScene(SceneManager.GetActiveScene()))
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
