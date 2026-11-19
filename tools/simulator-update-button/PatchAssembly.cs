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
        TypeDefinition hiderType = hookAssembly.MainModule.GetType("DonkeySimUpdate.MacTitleBarHider");
        if (hookType == null || hiderType == null)
        {
            Console.Error.WriteLine("hook types not found");
            return 3;
        }

        MethodDefinition installMethod = hookType.Methods.Single(method => method.Name == "Install");
        MethodDefinition applyMethod = hiderType.Methods.Single(method => method.Name == "Apply");
        TypeDefinition targetType = assembly.MainModule.GetType("tk.TcpMenuHandler");
        if (targetType == null)
        {
            Console.Error.WriteLine("tk.TcpMenuHandler not found");
            return 3;
        }

        MethodDefinition startMethod = targetType.Methods.Single(method => method.Name == "Start");
        bool hasInstall = HasInjectedCall(startMethod, "Install");
        bool hasApply = HasInjectedCall(startMethod, "Apply");
        if (hasInstall && hasApply)
        {
            Console.WriteLine("Assembly already patched");
            return 0;
        }

        MethodReference importedInstall = assembly.MainModule.ImportReference(installMethod);
        MethodReference importedApply = assembly.MainModule.ImportReference(applyMethod);
        ILProcessor il = startMethod.Body.GetILProcessor();
        Instruction first = startMethod.Body.Instructions[0];
        if (!hasInstall)
        {
            il.InsertBefore(first, il.Create(OpCodes.Ldarg_0));
            il.InsertBefore(first, il.Create(OpCodes.Call, importedInstall));
        }

        if (!hasApply)
        {
            il.InsertBefore(first, il.Create(OpCodes.Call, importedApply));
        }

        startMethod.Body.MaxStackSize = Math.Max(startMethod.Body.MaxStackSize, (ushort)8);

        string outputDirectory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrEmpty(outputDirectory))
        {
            Directory.CreateDirectory(outputDirectory);
        }

        assembly.Write(outputPath);
        Console.WriteLine("patched " + outputPath);
        return 0;
    }

    private static bool HasInjectedCall(MethodDefinition method, string name)
    {
        foreach (Instruction instruction in method.Body.Instructions)
        {
            MethodReference reference = instruction.Operand as MethodReference;
            if (reference != null &&
                reference.DeclaringType.Namespace == "DonkeySimUpdate" &&
                reference.Name == name)
            {
                return true;
            }
        }

        return false;
    }
}
