// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable
using System;
using System.Runtime.InteropServices;


namespace TorchSharp.PInvoke
{
    internal static partial class NativeMethods
    {
        //TODO: IMPLEMENT THIS

        [DllImport("LibTorchSharp")]
        internal static extern void THSHistogram_histogram(
            IntPtr input, long bins, IntPtr ranges, long length,
            IntPtr weight, bool density, out IntPtr hist, out IntPtr hist_bins);
    }
}
