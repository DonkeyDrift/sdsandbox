using System;
using System.IO;
using System.Linq;
using Mono.Cecil;
using Mono.Cecil.Cil;

internal static class PatchAssembly
{
    private static int Main(string[] arguments)
    {
        if (arguments.Length != 3)
        {
            Console.Error.WriteLine("usage: PatchAssembly <Assembly-CSharp.dll> <SimulatorUpdateButtonHook.dll> <output Assembly-CSharp.dll>");
            return 2;
        }

        string assemblyPath = Path.GetFullPath(arguments[0]);
        string hookPath = Path.GetFullPath(arguments[1]);
        string outputPath = Path.GetFullPath(arguments[2]);
        string managedDirectory = Path.GetDirectoryName(assemblyPath);

        DefaultAssemblyResolver resolver = new DefaultAssemblyResolver();
        resolver.AddSearchDirectory(managedDirectory);
        resolver.AddSearchDirectory(Path.GetDirectoryName(hookPath));

        ReaderParameters readerParameters = new ReaderParameters();
        readerParameters.AssemblyResolver = resolver;

        AssemblyDefinition assembly = AssemblyDefinition.ReadAssembly(assemblyPath, readerParameters);
        AssemblyDefinition hookAssembly = AssemblyDefinition.ReadAssembly(hookPath, readerParameters);
        TypeDefinition hookType = hookAssembly.MainModule.GetType("DonkeySimUpdate.SimulatorUpdateButtonHook");
        if (hookType == null)
        {
            Console.Error.WriteLine("hook type not found");
            return 3;
        }

        MethodDefinition installMethod = hookType.Methods.Single(method => method.Name == "Install");
        TypeDefinition targetType = assembly.MainModule.GetType("tk.TcpMenuHandler");
        if (targetType == null)
        {
            Console.Error.WriteLine("tk.TcpMenuHandler not found");
            return 3;
        }

        MethodDefinition startMethod = targetType.Methods.Single(method => method.Name == "Start");
        if (AlreadyPatched(startMethod))
        {
            Console.WriteLine("Install hook already present in tk.TcpMenuHandler.Start");
        }
        else
        {
            MethodReference importedMethod = assembly.MainModule.ImportReference(installMethod);
            ILProcessor il = startMethod.Body.GetILProcessor();
            Instruction first = startMethod.Body.Instructions[0];
            il.InsertBefore(first, il.Create(OpCodes.Ldarg_0));
            il.InsertBefore(first, il.Create(OpCodes.Call, importedMethod));
            startMethod.Body.MaxStackSize = Math.Max(startMethod.Body.MaxStackSize, (ushort)8);
        }

        RemoveSettingsVersionButton(assembly);

        string outputDirectory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrEmpty(outputDirectory))
        {
            Directory.CreateDirectory(outputDirectory);
        }

        assembly.Write(outputPath);
        Console.WriteLine("patched " + outputPath);
        return 0;
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

    private static bool AlreadyPatched(MethodDefinition method)
    {
        foreach (Instruction instruction in method.Body.Instructions)
        {
            MethodReference reference = instruction.Operand as MethodReference;
            if (reference != null &&
                reference.DeclaringType.FullName == "DonkeySimUpdate.SimulatorUpdateButtonHook" &&
                reference.Name == "Install")
            {
                return true;
            }
        }

        return false;
    }
}
