using UnityEngine;
using UnityEditor;
using UnityEditor.Build.Reporting;

/// <summary>
/// Batch-friendly macOS player build used by the app-icon tooling
/// (tools/simulator-icon). Unlike PlayerBuilder.MacBuild this does NOT copy
/// AssetBundles (the source checkout does not carry them) — the caller
/// backfills StreamingAssets from a published package afterwards.
/// </summary>
public static class IconBuild
{
    public static void Mac()
    {
        EditorBuildSettingsScene[] scenes = EditorBuildSettings.scenes;
        BuildReport report = BuildPipeline.BuildPlayer(
            scenes,
            "Builds/DonkeySimMac/donkey_sim.app",
            BuildTarget.StandaloneOSX,
            BuildOptions.None);
        BuildSummary summary = report.summary;
        Debug.Log("IconBuild result: " + summary.result + " size " + summary.totalSize);
        if (summary.result != BuildResult.Succeeded)
        {
            EditorApplication.Exit(1);
        }
    }
}
