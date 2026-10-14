using System;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;

/// <summary>
/// 命令行构建入口(game-ci 风格):
///   Unity -batchmode -projectPath sdsim -executeMethod BuildCommand.Build \
///         -buildTargetStandaloneOSX path/to/output.app -quit
/// 项目版本见 ProjectSettings/ProjectVersion.txt(当前 2020.3.49f1)。
/// </summary>
public static class BuildCommand
{
    public static void Build()
    {
        string outputPath = GetArg("-buildTargetStandaloneOSX");
        if (string.IsNullOrEmpty(outputPath))
            outputPath = "build/donkey_sim.app";

        BuildPlayerOptions opts = new BuildPlayerOptions
        {
            // 用 EditorBuildSettings 里启用的全部场景(11 个),只编 menu 会导致
            // 客户端连上后无法加载 generated_track 等场景、永远收不到遥测。
            scenes = System.Linq.Enumerable.ToArray(
                System.Linq.Enumerable.Select(
                    System.Linq.Enumerable.Where(EditorBuildSettings.scenes, s => s.enabled),
                    s => s.path)),
            locationPathName = outputPath,
            target = BuildTarget.StandaloneOSX,
            options = BuildOptions.None,
        };

        BuildReport report = BuildPipeline.BuildPlayer(opts);
        BuildSummary summary = report.summary;
        Console.WriteLine($"[BuildCommand] result={summary.result} size={summary.totalSize} errors={summary.totalErrors}");
        if (summary.result != BuildResult.Succeeded)
            EditorApplication.Exit(1);
        EditorApplication.Exit(0);
    }

    private static string GetArg(string name)
    {
        var args = Environment.GetCommandLineArgs();
        for (int i = 0; i < args.Length; i++)
            if (args[i] == name && i + 1 < args.Length)
                return args[i + 1];
        return null;
    }
}
