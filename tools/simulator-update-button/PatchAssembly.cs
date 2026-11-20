using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Mono.Cecil;
using Mono.Cecil.Cil;

/// <summary>
/// Cecil patcher for published Donkey simulator packages.
///
/// Two invocation shapes:
///
///   legacy: PatchAssembly &lt;Assembly-CSharp.dll&gt; &lt;SimulatorUpdateButtonHook.dll&gt; &lt;output&gt;
///           Injects SimulatorUpdateButtonHook.Install(this) at the head of
///           tk.TcpMenuHandler.Start, then applies the settings patches
///           (RemoveSettingsVersionButton + InjectChannelToggle).
///
///   spec:   PatchAssembly &lt;Assembly-CSharp.dll&gt; &lt;output&gt; &lt;hook.dll=Type[::Anchor.Method]&gt; [...]
///           Injects each hook's Install at its anchor (default
///           tk.TcpMenuHandler.Start), then applies the settings patches.
///
/// Anchor note (verified 2026-11-19): the 2022.3.49 base shipped in the
/// upstream releases never executes tk.TcpMenuHandler.Start (a WriteLine
/// probe at its head never fires) — package pipelines for that base anchor
/// at CarRecovery.Update, which runs every frame from first boot; every
/// Install is idempotent, so a per-frame anchor is safe. Source-built
/// (Unity 6) packages keep the Start anchor.
/// </summary>
internal static class PatchAssembly
{
    private static int Main(string[] arguments)
    {
        if (arguments.Length < 3)
        {
            Console.Error.WriteLine(
                "usage: PatchAssembly <Assembly-CSharp.dll> <hook.dll> <output>   (legacy single hook)\n" +
                "       PatchAssembly <Assembly-CSharp.dll> <output> <hook.dll=Type[::Anchor.Method]> [...]   (spec)");
            return 2;
        }

        string assemblyPath = Path.GetFullPath(arguments[0]);
        string managedDirectory = Path.GetDirectoryName(assemblyPath);

        var hooks = new List<Tuple<string, string, string, string>>();
        string outputPath;
        bool runSettingsPatches = true;

        if (arguments[2].IndexOf('=') >= 0)
        {
            outputPath = arguments[1];
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
        }
        else
        {
            outputPath = arguments[2];
            hooks.Add(Tuple.Create(
                Path.GetFullPath(arguments[1]),
                "DonkeySimUpdate.SimulatorUpdateButtonHook",
                "tk.TcpMenuHandler",
                "Start"));
        }

        outputPath = Path.GetFullPath(outputPath);

        DefaultAssemblyResolver resolver = new DefaultAssemblyResolver();
        resolver.AddSearchDirectory(managedDirectory);
        foreach (Tuple<string, string, string, string> hook in hooks)
        {
            resolver.AddSearchDirectory(Path.GetDirectoryName(hook.Item1));
        }

        ReaderParameters readerParameters = new ReaderParameters();
        readerParameters.AssemblyResolver = resolver;

        AssemblyDefinition assembly = AssemblyDefinition.ReadAssembly(assemblyPath, readerParameters);

        foreach (Tuple<string, string, string, string> hook in hooks)
        {
            if (!File.Exists(hook.Item1))
            {
                Console.Error.WriteLine($"hook dll not found: {hook.Item1}");
                return 3;
            }

            AssemblyDefinition hookAssembly = AssemblyDefinition.ReadAssembly(hook.Item1, readerParameters);
            TypeDefinition hookType = hookAssembly.MainModule.GetType(hook.Item2);
            if (hookType == null)
            {
                Console.Error.WriteLine($"hook type not found: {hook.Item2} in {hook.Item1}");
                return 3;
            }

            InjectInstall(assembly, hookType, hook.Item2, hook.Item3, hook.Item4);
        }

        if (runSettingsPatches)
        {
            RemoveSettingsVersionButton(assembly);
            AssemblyDefinition firstHook = AssemblyDefinition.ReadAssembly(hooks[0].Item1, readerParameters);
            TypeDefinition firstHookType = firstHook.MainModule.GetType(hooks[0].Item2);
            InjectChannelToggle(assembly, firstHookType);
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

    private static void InjectInstall(
        AssemblyDefinition assembly,
        TypeDefinition hookType,
        string typeName,
        string anchorTypeName,
        string anchorMethodName)
    {
        MethodDefinition installMethod = hookType.Methods.Single(method => method.Name == "Install");
        TypeDefinition targetType = assembly.MainModule.GetType(anchorTypeName);
        if (targetType == null)
        {
            Console.Error.WriteLine($"anchor type not found: {anchorTypeName}");
            Environment.Exit(3);
        }

        MethodDefinition anchorMethod = targetType.Methods.FirstOrDefault(
            method => method.Name == anchorMethodName && method.HasBody);
        if (anchorMethod == null)
        {
            Console.Error.WriteLine($"anchor method not found: {anchorTypeName}.{anchorMethodName}");
            Environment.Exit(3);
        }

        if (AlreadyPatched(anchorMethod, typeName))
        {
            Console.WriteLine($"{typeName}.Install already present in {anchorTypeName}.{anchorMethodName}");
            return;
        }

        MethodReference importedMethod = assembly.MainModule.ImportReference(installMethod);
        ILProcessor il = anchorMethod.Body.GetILProcessor();
        Instruction first = anchorMethod.Body.Instructions[0];
        il.InsertBefore(first, il.Create(OpCodes.Ldarg_0));
        il.InsertBefore(first, il.Create(OpCodes.Call, importedMethod));
        anchorMethod.Body.MaxStackSize = Math.Max(anchorMethod.Body.MaxStackSize, (ushort)8);
        Console.WriteLine($"injected {typeName}.Install into {anchorTypeName}.{anchorMethodName}");
    }

    private static void RemoveSettingsVersionButton(AssemblyDefinition assembly)
    {
        TypeDefinition globalStateEditor = assembly.MainModule.GetType("GlobalStateEditor");
        if (globalStateEditor != null)
        {
            ShortCircuitVersionButton(globalStateEditor);
        }

        TypeDefinition versionCheck = assembly.MainModule.GetType("VersionCheck");
        if (versionCheck != null)
        {
            EmptyMethod(versionCheck, "Awake");
            EmptyMethod(versionCheck, "GetLatestVersion");
        }
    }

    private static void ShortCircuitVersionButton(TypeDefinition globalStateEditor)
    {
        MethodDefinition onGui = globalStateEditor.Methods.SingleOrDefault(method => method.Name == "OnGUI");
        if (onGui == null || !onGui.HasBody)
        {
            Console.WriteLine("GlobalStateEditor.OnGUI not found; skipping settings version button removal");
            return;
        }

        ILProcessor il = onGui.Body.GetILProcessor();
        var instructions = onGui.Body.Instructions;
        for (int i = 0; i < instructions.Count; i++)
        {
            if (!IsFieldLoad(instructions[i], OpCodes.Ldfld, "VersionCheck", "latest"))
            {
                continue;
            }

            int versionIndex = NextMeaningful(onGui.Body, i);
            if (versionIndex < 0 || !IsFieldLoad(instructions[versionIndex], OpCodes.Ldsfld, "GlobalState", "version"))
            {
                continue;
            }

            int callIndex = NextMeaningful(onGui.Body, versionIndex);
            if (callIndex < 0 || instructions[callIndex].OpCode != OpCodes.Call)
            {
                continue;
            }

            MethodReference called = instructions[callIndex].Operand as MethodReference;
            if (called == null || called.Name != "op_Inequality")
            {
                continue;
            }

            Instruction call = instructions[callIndex];
            Instruction pop1 = il.Create(OpCodes.Pop);
            Instruction pop2 = il.Create(OpCodes.Pop);
            Instruction zero = il.Create(OpCodes.Ldc_I4_0);
            il.Replace(call, pop1);
            il.InsertAfter(pop1, pop2);
            il.InsertAfter(pop2, zero);
            Console.WriteLine("short-circuited the settings version button in GlobalStateEditor.OnGUI");
            return;
        }

        Console.WriteLine("settings version button comparison not found in GlobalStateEditor.OnGUI (already removed)");
    }

    private static bool IsFieldLoad(Instruction instruction, OpCode opCode, string typeName, string fieldName)
    {
        if (instruction.OpCode != opCode)
        {
            return false;
        }

        FieldReference field = instruction.Operand as FieldReference;
        return field != null &&
            field.Name == fieldName &&
            field.DeclaringType != null &&
            field.DeclaringType.Name == typeName;
    }

    private static int NextMeaningful(MethodBody body, int index)
    {
        var instructions = body.Instructions;
        for (int i = index + 1; i < instructions.Count; i++)
        {
            if (instructions[i].OpCode != OpCodes.Nop)
            {
                return i;
            }
        }

        return -1;
    }

    private static void EmptyMethod(TypeDefinition type, string name)
    {
        MethodDefinition method = type.Methods.SingleOrDefault(candidate => candidate.Name == name);
        if (method == null || !method.HasBody)
        {
            Console.WriteLine(type.Name + "." + name + " not found; skipping");
            return;
        }

        MethodBody body = method.Body;
        if (body.Instructions.Count == 1 && body.Instructions[0].OpCode == OpCodes.Ret)
        {
            Console.WriteLine(type.Name + "." + name + " already empty");
            return;
        }

        body.Variables.Clear();
        body.ExceptionHandlers.Clear();
        body.Instructions.Clear();
        body.Instructions.Add(Instruction.Create(OpCodes.Ret));
        body.MaxStackSize = Math.Max(body.MaxStackSize, (ushort)1);
        Console.WriteLine("emptied " + type.Name + "." + name);
    }

    private static void InjectChannelToggle(AssemblyDefinition assembly, TypeDefinition hookType)
    {
        MethodDefinition drawMethod = hookType.Methods.SingleOrDefault(method => method.Name == "DrawChannelToggle");
        if (drawMethod == null)
        {
            Console.WriteLine("DrawChannelToggle not found on the hook; skipping channel toggle injection");
            return;
        }

        TypeDefinition globalStateEditor = assembly.MainModule.GetType("GlobalStateEditor");
        if (globalStateEditor == null)
        {
            Console.WriteLine("GlobalStateEditor not found; skipping channel toggle injection");
            return;
        }

        MethodDefinition onGui = globalStateEditor.Methods.SingleOrDefault(method => method.Name == "OnGUI");
        if (onGui == null || !onGui.HasBody)
        {
            Console.WriteLine("GlobalStateEditor.OnGUI not found; skipping channel toggle injection");
            return;
        }

        foreach (Instruction instruction in onGui.Body.Instructions)
        {
            MethodReference reference = instruction.Operand as MethodReference;
            if (reference != null &&
                reference.DeclaringType.FullName == "DonkeySimUpdate.SimulatorUpdateButtonHook" &&
                reference.Name == "DrawChannelToggle")
            {
                Console.WriteLine("channel toggle already injected in GlobalStateEditor.OnGUI");
                return;
            }
        }

        Instruction endGroup = null;
        foreach (Instruction instruction in onGui.Body.Instructions)
        {
            MethodReference reference = instruction.Operand as MethodReference;
            if (instruction.OpCode == OpCodes.Call &&
                reference != null &&
                reference.DeclaringType.Name == "GUI" &&
                reference.Name == "EndGroup")
            {
                endGroup = instruction;
            }
        }

        if (endGroup == null)
        {
            Console.WriteLine("GUI.EndGroup not found in GlobalStateEditor.OnGUI; skipping channel toggle injection");
            return;
        }

        MethodReference importedDraw = assembly.MainModule.ImportReference(drawMethod);
        FieldReference rectField = onGui.Module.ImportReference(globalStateEditor.Fields.Single(field => field.Name == "rect"));
        ILProcessor il = onGui.Body.GetILProcessor();
        Instruction callDraw = il.Create(OpCodes.Call, importedDraw);
        Instruction loadRect = il.Create(OpCodes.Ldfld, rectField);
        Instruction loadThis = il.Create(OpCodes.Ldarg_0);
        il.InsertAfter(endGroup, callDraw);
        il.InsertAfter(endGroup, loadRect);
        il.InsertAfter(endGroup, loadThis);
        onGui.Body.MaxStackSize = Math.Max(onGui.Body.MaxStackSize, (ushort)8);
        Console.WriteLine("injected update channel toggle into GlobalStateEditor.OnGUI");
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
