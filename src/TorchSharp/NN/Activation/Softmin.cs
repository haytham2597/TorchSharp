// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using TorchSharp.Amp;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a Softmin module.
        /// </summary>
        public sealed class Softmin : torch.nn.Module<Tensor, Tensor>
        {
            internal Softmin(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_Softmin_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public override string GetName()
            {
                return typeof(Softmin).Name;
            }

           // Rather than spending cycles only to discover that this module has neither
            // parameters nor buffers, just shortcut the move completely.
            protected internal override nn.Module _to(Device device, ScalarType dtype, bool non_blocking) => this;
            protected internal override nn.Module _to(DeviceType deviceType, int deviceIndex, bool non_blocking) => this;
            protected internal override nn.Module _to(ScalarType dtype, bool non_blocking) => this;
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Softmin
            /// </summary>
            /// <param name="dim">A dimension along which Softmin will be computed (so every slice along dim will sum to 1)</param>
            /// <returns></returns>
            public static Softmin Softmin(long dim)
            {
                var handle = THSNN_Softmin_ctor(dim, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                handle = AutocastMode.AutoCast(handle, ScalarType.Float32); //Should put this here???
                return new Softmin(handle, boxedHandle);
            }

            public static partial class functional
            {
                /// <summary>
                /// Softmin
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="dim">A dimension along which Softmin will be computed (so every slice along dim will sum to 1)</param>
                /// <returns></returns>
                public static Tensor softmin(Tensor x, long dim)
                {
                    using (var m = nn.Softmin(dim)) {
                        return m.call(x);
                    }
                }
            }
        }
    }
}
