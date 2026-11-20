using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SceneLoaded : MonoBehaviour
{
    public bool inFrontEnd = false;

    // Start is called before the first frame update
    void Start()
    {
        // The scene may hold several SandboxServers (public + private API).
        // FindObjectOfType order is unspecified and changed across Unity
        // versions, so configure every server instead of an arbitrary one.
        SandboxServer[] servers = FindObjectsByType<SandboxServer>(FindObjectsSortMode.None);

        foreach (SandboxServer server in servers)
        {
            server.OnSceneLoaded(inFrontEnd);
        }
    }
}
