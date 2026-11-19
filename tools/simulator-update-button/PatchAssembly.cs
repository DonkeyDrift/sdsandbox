using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Mono.Cecil;
using Mono.Cecil.Cil;

/// <summary>
/// Inject <c>&lt;HookType&gt;.Install(this)</c> calls at the head of an anchor method
/// inside Assembly-CSharp so the hooks activate in the published player
/// (loose DLLs' [RuntimeInitializeOnLoadMethod] does NOT fire in a built
/// player — the Cecil call is the activation path; verified 2026-11-19).
///
/// Anchor history: the original flow anchored at tk.TcpMenuHandler.Start.
/// The 2022.3.49 base shipped in the current releases never executes that
/// method (a WriteLine probe at its head never fires), so the pipeline now
/// anchors at CarRecovery.Update — verified to run every frame from first
/// boot. Both hooks' Install methods are idempotent, so a per-frame anchor
/// is safe.
///
/// usage: PatchAssembly &lt;Assembly-CSharp.dll&gt; &lt;output Assembly-CSharp.dll&gt;
///            &lt;hook1.dll=DonkeySimUpdate.HookType1[::AnchorType.AnchorMethod]&gt;
///            [hook2.dll=Type2[::Anchor.Method] ...]
///
/// The optional ::Type.Method suffix overrides the anchor for that hook
/// (default tk.TcpMenuHandler.Start).
/// </summary>
internal static class PatchAssembly
{
    private static int Main(string[] arguments)
    {
        if (arguments.Length < 3)
        {
            Console.Error.WriteLine(
                "usage: PatchAssembly <Assembly-CSharp.dll> <output> <hook.dll=Type[::Anchor.Method]> [more...]");
            return 2;
        }

        string assemblyPath = Path.GetFullPath(arguments[0]);
        string outputPath = Path.GetFullPath(arguments[1]);
        string managedDirectory = Path.GetDirectoryName(assemblyPath);

        var hooks = new List<Tuple<string, string, string, string>>();
        foreach (string argument in arguments.Skip(2))
        {
            string[] parts = argument.Split(new[] { '=' }, 2);
            if (parts.Length != 2)
            {
                Console.Error.WriteLine($"bad hook spec (want dll=Type[::Anchor.Method]): {argument}");
                return 2;
            }

            string anchorType = "tk.TcpMenuHandler";
            string anchorMethod = "Start";
            string[] hookParts = parts[1].Split(new[] { "::" }, StringSplitOptions.None);
            string typeName = hookParts[0];
            if (hookParts.Length == 2)
            {
                string[] anchor = hookParts[1].Split(new[] { '.' }, 2);
                if (anchor.Length != 2)
                {
                    Console.Error.WriteLine($"bad anchor (want Type.Method): {hookParts[1]}");
                    return 2;
                }

                anchorType = anchor[0];
                anchorMethod = anchor[1];
            }

            hooks.Add(Tuple.Create(Path.GetFullPath(parts[0]), typeName, anchorType, anchorMethod));
        }

        DefaultAssemblyResolver resolver = new DefaultAssemblyResolver();
        resolver.AddSearchDirectory(managedDirectory);
        foreach (Tuple<string, string, string, string> hook in hooks)
        {
            resolver.AddSearchDirectory(Path.GetDirectoryName(hook.Item1));
        }

        ReaderParameters readerParameters = new ReaderParameters();
        readerParameters.AssemblyResolver = resolver;

        AssemblyDefinition assembly = AssemblyDefinition.ReadAssembly(assemblyPath, readerParameters);
        bool anyInjected = false;

        foreach (Tuple<string, string, string, string> hook in hooks)
        {
            string dllPath = hook.Item1;
            string typeName = hook.Item2;
            if (!File.Exists(dllPath))
            {
                Console.Error.WriteLine($"hook dll not found: {dllPath}");
                return 3;
            }

            AssemblyDefinition hookAssembly = AssemblyDefinition.ReadAssembly(dllPath, readerParameters);
            TypeDefinition hookType = hookAssembly.MainModule.GetType(typeName);
            if (hookType == null)
            {
                Console.Error.WriteLine($"hook type not found: {typeName} in {dllPath}");
                return 3;
            }

            MethodDefinition installMethod = hookType.Methods.Single(method => method.Name == "Install");

            TypeDefinition targetType = assembly.MainModule.GetType(hook.Item3);
            if (targetType == null)
            {
                Console.Error.WriteLine($"anchor type not found: {hook.Item3}");
                return 3;
            }

            MethodDefinition anchorMethod = targetType.Methods.FirstOrDefault(
                method => method.Name == hook.Item4 && method.HasBody);
            if (anchorMethod == null)
            {
                Console.Error.WriteLine($"anchor method not found: {hook.Item3}.{hook.Item4}");
                return 3;
            }

            if (AlreadyPatched(anchorMethod, typeName))
            {
                Console.WriteLine($"{typeName} already patched into {hook.Item3}.{hook.Item4}");
                continue;
            }

            MethodReference importedMethod = assembly.MainModule.ImportReference(installMethod);
            ILProcessor il = anchorMethod.Body.GetILProcessor();
            Instruction first = anchorMethod.Body.Instructions[0];
            il.InsertBefore(first, il.Create(OpCodes.Ldarg_0));
            il.InsertBefore(first, il.Create(OpCodes.Call, importedMethod));
            anchorMethod.Body.MaxStackSize = Math.Max(anchorMethod.Body.MaxStackSize, (ushort)8);
            anyInjected = true;
            Console.WriteLine($"injected {typeName}.Install into {hook.Item3}.{hook.Item4}");
        }

        if (!anyInjected)
        {
            Console.WriteLine("nothing to inject");
            return 0;
        }

        string outputDirectory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrEmpty(outputDirectory))
        {
            Directory.CreateDirectory(outputDirectory);
        }

        assembly.Write(outputPath);
        Console.WriteLine("patched " + outputPath);
        return 0;
    }

    private static bool AlreadyPatched(MethodDefinition method, string typeName)
    {
        foreach (Instruction instruction in method.Body.Instructions)
        {
            MethodReference reference = instruction.Operand as MethodReference;
            if (reference != null &&
                reference.DeclaringType.FullName == typeName &&
                reference.Name == "Install")
            {
                return true;
            }
        }

        return false;
    }
}
