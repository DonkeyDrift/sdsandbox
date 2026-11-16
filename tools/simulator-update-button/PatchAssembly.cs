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
            Console.WriteLine("Assembly already patched");
            return 0;
        }

        MethodReference importedMethod = assembly.MainModule.ImportReference(installMethod);
        ILProcessor il = startMethod.Body.GetILProcessor();
        Instruction first = startMethod.Body.Instructions[0];
        il.InsertBefore(first, il.Create(OpCodes.Ldarg_0));
        il.InsertBefore(first, il.Create(OpCodes.Call, importedMethod));
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
