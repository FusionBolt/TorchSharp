// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Data;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;

namespace TorchSharp
{
    public static partial class torch
    {
#if LIBTORCH_1_10_0_1
        const string libtorchPackageVersion = "1.10.0.1";
#else
#error "Please update libtorchPackageVersion to match LibTorchPackageVersion"
#endif
#if CUDA_11_3
        const string cudaVersion = "11.3";
#else
#error "Please update cudaVersion to match CudaVersionDot"
#endif

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        static extern bool SetDllDirectory(string lpPathName);

        static string nativeRid =>
            RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "win-x64" :
            RuntimeInformation.IsOSPlatform(OSPlatform.Linux) ? "linux-x64" :
            RuntimeInformation.IsOSPlatform(OSPlatform.OSX) ? "osx-x64" :
            "any";

        static string nativeGlob =>
            RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? @".*\.dll" :
            RuntimeInformation.IsOSPlatform(OSPlatform.OSX) ? @".*\.dylib\.*" :
            // must match
            //   lib.so
            //   lib.so.1
            //   lib.so.11.0
            //   lib.so.11.3
            @".*\.so(\.\d*)*";
        static bool nativeBackendLoaded = false;
        // static bool nativeBackendCudaLoaded = false;

        internal static bool TryLoadNativeLibraryFromFile(string path, StringBuilder trace) {
            bool ok;
            try {
                trace.AppendLine($"    Trying to load native component {path}");
                ok = NativeLibrary.TryLoad(path, out var res);
                if (!ok)
                    trace.AppendLine($"    Failed to load native component {path}");
            } catch {
                ok = false;
            }
            return ok;
        }

        internal static bool TryLoadNativeLibraryByName(string name, Assembly assembly, StringBuilder trace)
        {
            bool ok;
            try {
                trace.AppendLine($"    Trying to load native component {name} relative to {assembly.Location}");
                ok = NativeLibrary.TryLoad(name, assembly, null, out var res);
                if (!ok)
                    trace.AppendLine($"    Failed to load native component {name} relative to {assembly.Location}");
            } catch (Exception exn) {
                trace.AppendLine($"    Failed to load native component {name} relative to {assembly.Location}: {exn.Message}");
                ok = false;
            }
            return ok;
        }

        private static void LoadNativeBackend(bool useCudaBackend, out StringBuilder trace)
        {
            trace = new StringBuilder();
            if (!nativeBackendLoaded)
            {
                var ok = TryLoadNativeLibraryByName("LibTorchSharp", typeof(torch).Assembly, trace);
                if (!ok)
                {
                    var message = $"load libtorchsharp failed";
                    Console.WriteLine(message);
                    throw new NotSupportedException(message);
                }
                nativeBackendLoaded = true;
            }
        }

        /// Copy all native runtime DLLs into single directory if it hasn't been done already
        private static bool CopyNativeComponentsIntoSingleDirectory(string packagesDir,
            string packagePattern,
            string packageVersion,
            string target,
            StringBuilder trace)
        {
            // Some loads will fail due to missing dependencies but then
            // these will be resolved in subsequent iterations.
            trace.AppendLine($"    Consolidating native binaries, packagesDir={packagesDir}, packagePattern={packagePattern}, packageVersion={packageVersion} to target={target}...");
            if (Directory.Exists(packagesDir)) {
                var packages =
                    Directory.GetDirectories(packagesDir, packagePattern)
                       .Where(d => Directory.Exists(Path.Combine(d, packageVersion)))
                       .ToArray();

                if (packages.Length > 0) {
                    if (!Directory.Exists(target))
                        Directory.CreateDirectory(target);
                    foreach (var package in packages) {
                        var natives = Path.Combine(package, packageVersion, "runtimes", nativeRid, "native");
                        trace.AppendLine($"    CopyNativeComponentsIntoSingleDirectory: natives={natives}");
                        if (Directory.Exists(natives)) {
                            var nativeRegExp = new Regex("^" + nativeGlob + "$");
                            foreach (var file in Directory.GetFiles(natives).Where(path => nativeRegExp.IsMatch(path))) {
                                var targetFile = Path.Combine(target, Path.GetFileName(file));
                                if (!File.Exists(targetFile)) {
                                    trace.AppendLine($"Copy {file} --> {targetFile}");
                                    File.Copy(file, targetFile);
                                }
                            }
                        }
                    }
                    return true;
                }
            }
            return false;
        }

        public static bool TryInitializeDeviceType(DeviceType deviceType)
        {
            LoadNativeBackend(deviceType == DeviceType.CUDA, out _);
            if (deviceType == DeviceType.CUDA) {
                return cuda.CallTorchCudaIsAvailable();
            } else {
                return true;
            }
        }

        public static void InitializeDeviceType(DeviceType deviceType)
        {
            LoadNativeBackend(deviceType == DeviceType.CUDA, out var trace);
            if (deviceType == DeviceType.CUDA) {

                // For CUDA, we double-check that CudaIsAvailable actually returns true.
                // If it doesn't we report the entire load trace.
                var result = cuda.CallTorchCudaIsAvailable();
                if (!result)
                    throw new InvalidOperationException($"Torch device type {deviceType} did not initialise on the current machine. Trace from LoadNativeBackend:\n{trace}");
            }
        }

        public static Device InitializeDevice(torch.Device device)
        {
            if (device == null)
                device = torch.CPU;
            InitializeDeviceType(device.type);
            return device;
        }

        public static partial class random
        {
            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSGenerator_manual_seed(long seed);

            /// <summary>
            /// Sets the seed for generating random numbers. Returns a torch.Generator object.
            /// </summary>
            /// <param name="seed">The desired seed.</param>
            /// <returns></returns>
            public static Generator manual_seed(long seed)
            {
                TryInitializeDeviceType(DeviceType.CUDA);
                var res = THSGenerator_manual_seed(seed);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Generator(res);
            }
        }

        public static partial class nn
        {
            public static partial class utils
            {
                [DllImport("LibTorchSharp")]
                extern static double THSTensor_clip_grad_norm_(IntPtr tensor, int len, double max_norm, double norm_type);

                /// <summary>
                /// Clips gradient norm of an iterable of parameters.
                /// The norm is computed over all gradients together, as if they were concatenated into a single vector.
                /// Gradients are modified in-place.
                /// </summary>
                /// <param name="tensors"></param>
                /// <param name="max_norm"></param>
                /// <param name="norm_type"></param>
                /// <returns></returns>
                public static double clip_grad_norm_(IList<Tensor> tensors, double max_norm, double norm_type = 2.0)
                {
                    using (var parray = new PinnedArray<IntPtr>()) {
                        IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

                        return THSTensor_clip_grad_norm_(tensorsRef, parray.Array.Length, max_norm, norm_type);
                    }
                }

            }
        }

        public static partial class cuda
        {
            [DllImport("LibTorchSharp")]
            private static extern bool THSTorchCuda_is_available();

            /// This must be a separate method to the failure to bind DllImport THSTorchCuda_is_available
            /// is not raised as early as a DllImportException
            [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
            internal static bool CallTorchCudaIsAvailable()
            {
                return THSTorchCuda_is_available();
            }

            /// <summary>
            /// Returns a bool indicating if CUDA is currently available.
            /// </summary>
            /// <returns></returns>
            public static bool is_available()
            {
                TryInitializeDeviceType(DeviceType.CUDA);
                return CallTorchCudaIsAvailable();
            }

            [DllImport("LibTorchSharp")]
            private static extern bool THSTorchCuda_cudnn_is_available();

            /// <summary>
            /// Returns a bool indicating if CUDNN is currently available.
            /// </summary>
            public static bool is_cudnn_available()
            {
                TryInitializeDeviceType(DeviceType.CUDA);
                return THSTorchCuda_cudnn_is_available();
            }

            [DllImport("LibTorchSharp")]
            private static extern int THSTorchCuda_device_count();

            /// <summary>
            /// Returns the number of GPUs available.
            /// </summary>
            /// <returns></returns>
            public static int device_count()
            {
                TryInitializeDeviceType(DeviceType.CUDA);
                return THSTorchCuda_device_count();
            }

            [DllImport("LibTorchSharp")]
            private static extern void THSCuda_manual_seed(long seed);

            /// <summary>
            /// Sets the seed for generating random numbers for the current GPU.
            /// It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
            /// </summary>
            /// <param name="seed">The desired seed.</param>
            public static void manual_seed(long seed)
            {
                TryInitializeDeviceType(DeviceType.CUDA);
                THSCuda_manual_seed(seed);
            }

            [DllImport("LibTorchSharp")]
            private static extern void THSCuda_manual_seed_all(long seed);

            /// <summary>
            /// Sets the seed for generating random numbers on all GPUs.
            /// It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
            /// </summary>
            /// <param name="seed"></param>
            public static void manual_seed_all(long seed)
            {
                TryInitializeDeviceType(DeviceType.CUDA);
                THSCuda_manual_seed_all(seed);
            }

            [DllImport("LibTorchSharp")]
            private static extern void THSCuda_synchronize(long device_index);

            /// <summary>
            /// Waits for all kernels in all streams on a CUDA device to complete.
            /// </summary>
            /// <param name="seed">Device for which to synchronize.
            /// It uses the current device, given by current_device(), if a device is not provided.</param>
            public static void synchronize(long seed = -1L)
            {
                TryInitializeDeviceType(DeviceType.CUDA);
                THSCuda_synchronize(seed);
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTorch_get_and_reset_last_err();

        //[Conditional("DEBUG")]
        internal static void CheckForErrors()
        {
            var error = THSTorch_get_and_reset_last_err();

            if (error != IntPtr.Zero)
            {
                throw new ExternalException(Marshal.PtrToStringAnsi(error));
            }
        }
    }

    /// <summary>
    /// The LibTorch device types.
    /// </summary>
    /// <remarks>TorchSharp currently only supports CPU and CUDA.</remarks>
    public enum DeviceType
    {
        CPU = 0,
        CUDA = 1, // CUDA.
        MKLDNN = 2, // Reserved for explicit MKLDNN
        OPENGL = 3, // OpenGL
        OPENCL = 4, // OpenCL
        IDEEP = 5, // IDEEP.
        HIP = 6, // AMD HIP
        FPGA = 7, // FPGA
        MSNPU = 8, // MSNPU
        XLA = 9 // XLA / TPU
    }
}
