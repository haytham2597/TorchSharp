// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a GELU module.
        /// </summary>
        public sealed class GELU : ParameterLessModule<Tensor, Tensor>
        {
            internal GELU(bool inplace) : base(nameof(GELU))
            {
                this.inplace = inplace;
            }

            public override Tensor forward(Tensor tensor)
            {
                return torch.nn.functional.gelu(tensor, inplace);
            }

            public bool inplace {get; set; }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            public enum Approx
            {
                none,
                tanh
            }
            /// <summary>
            /// Gaussian Error Linear Units
            /// </summary>
            /// <returns></returns>
            public static GELU GELU(torch.nn.Approx approximate = Approx.none)
            {
                var handle = THSNN_GELU_ctor(out var boxedHandle, approximate.ToString());
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new GELU(handle, boxedHandle);
            }
            public static partial class functional
            {
                /// <summary>
                /// Gaussian Error Linear Units
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="inplace">Do the operation in-place. Default: False</param>
                public static Tensor gelu(Tensor x, bool inplace)
                {
                    return inplace ? x.gelu_().alias() : x.gelu();
                }

                /// <summary>
                /// Gaussian Error Linear Units
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <remarks>The defaulting of 'inplace' to 'false' is implemented as an overload to avoid a breaking change.</remarks>
                public static Tensor gelu(Tensor x)
                {
                    return gelu(x,false);
                }
            }
        }
    }
}
