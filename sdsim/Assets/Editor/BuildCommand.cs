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
            scenes = new[] { "Assets/Scenes/menu.unity" },
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
