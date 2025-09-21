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
        internal static extern IntPtr THSQuantize_quantize_per_channel(IntPtr x, IntPtr scales, IntPtr zero_points,
            long axis, sbyte dtype);
        /*internal static extern IntPtr THSQuantize_quantize_per_channel_out_scale_axis(IntPtr x, IntPtr self, IntPtr zero_points,
            long axis, sbyte dtype);*/
    }
}