// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using TorchSharp.Amp;
using static TorchSharp.PInvoke.NativeMethods;

#nullable enable
namespace TorchSharp
{
    public static partial class torch
    {
        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static Tensor tensor(float scalar, Device? device = null, bool requires_grad = false)
        {
            device = InitializeDevice(device);
            var handle = THSTensor_newFloat32Scalar(scalar, (int)device.type, device.index, requires_grad);
            if (handle == IntPtr.Zero) { CheckForErrors(); }


            //var t = new Tensor(handle).AutoCast();
            var t = new Tensor(handle);
            /*if (is_autocast_cache_enabled()) {
                if (is_autocast_gpu_enabled())
                    return t.to(get_autocast_gpu_dtype()); //this work, but should put that on all tensor factorie... 
            }*/
            return t;
        }

        /// <summary>
        /// Create a scalar complex number tensor from independent real and imaginary components
        /// </summary>
        public static Tensor tensor(float real, float imaginary, ScalarType? dtype = null, Device? device = null, bool requires_grad = false)
        {
            device = InitializeDevice(device);
            var handle = THSTensor_newComplexFloat32Scalar(real, imaginary, (int)device.type, device.index, requires_grad);
            if (handle == IntPtr.Zero) { CheckForErrors(); }
            return InstantiateTensorWithLeakSafeTypeChange(handle, dtype);
        }

        /// <summary>
        /// Create a scalar complex number tensor from a tuple of (real, imaginary)
        /// </summary>
        public static Tensor tensor((float Real, float Imaginary) scalar, ScalarType? dtype = null, Device? device = null, bool requires_grad = false)
        {
            return tensor(scalar.Real, scalar.Imaginary, dtype: dtype, device: device);
        }

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        [Pure]
        public static Tensor tensor(IList<float> rawArray, ReadOnlySpan<long> dimensions, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray.ToArray(), dimensions, (sbyte)ScalarType.Float32, dtype, device, requires_grad, false, names);
        }

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        [Pure]
        public static Tensor tensor(float[] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, stackalloc long[] { rawArray.LongLength }, (sbyte)ScalarType.Float32, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        [Pure]
        public static Tensor tensor(float[] rawArray, ReadOnlySpan<long> dimensions, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, dimensions, (sbyte)ScalarType.Float32, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a 1-D tensor from an array of values, shaping it based on the input array.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        [Pure]
        public static Tensor tensor(IList<float> rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return tensor(rawArray, stackalloc long[] { (long)rawArray.Count }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a two-dimensional tensor.
        /// </summary>
        /// <remarks>
        /// The Torch runtime does not take ownership of the data, so there is no device argument.
        /// The input array must have rows * columns elements.
        /// </remarks>
        [Pure]
        public static Tensor tensor(IList<float> rawArray, long rows, long columns, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return tensor(rawArray, stackalloc long[] { rows, columns }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a three-dimensional tensor.
        /// </summary>
        /// <remarks>
        /// The Torch runtime does not take ownership of the data, so there is no device argument.
        /// The input array must have dim0*dim1*dim2 elements.
        /// </remarks>
        [Pure]
        public static Tensor tensor(IList<float> rawArray, long dim0, long dim1, long dim2, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return tensor(rawArray, stackalloc long[] { dim0, dim1, dim2 }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a four-dimensional tensor.
        /// </summary>
        /// <remarks>
        /// The Torch runtime does not take ownership of the data, so there is no device argument.
        /// The input array must have dim0*dim1*dim2*dim3 elements.
        /// </remarks>
        [Pure]
        public static Tensor tensor(IList<float> rawArray, long dim0, long dim1, long dim2, long dim3, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return tensor(rawArray, stackalloc long[] { dim0, dim1, dim2, dim3 }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a two-dimensional tensor from a two-dimensional array of values.
        /// </summary>
        [Pure]
        public static Tensor tensor(float[,] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, stackalloc long[] { rawArray.GetLongLength(0), rawArray.GetLongLength(1) }, (sbyte)ScalarType.Float32, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a three-dimensional tensor from a three-dimensional array of values.
        /// </summary>
        [Pure]
        public static Tensor tensor(float[,,] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, stackalloc long[] { rawArray.GetLongLength(0), rawArray.GetLongLength(1), rawArray.GetLongLength(2) }, (sbyte)ScalarType.Float32, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a four-dimensional tensor from a four-dimensional array of values.
        /// </summary>
        [Pure]
        public static Tensor tensor(float[,,,] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, stackalloc long[] { rawArray.GetLongLength(0), rawArray.GetLongLength(1), rawArray.GetLongLength(2), rawArray.GetLongLength(3) }, (sbyte)ScalarType.Float32, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        [Pure]
        public static Tensor tensor(Memory<float> rawArray, ReadOnlySpan<long> dimensions, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, dimensions, (sbyte)ScalarType.Float32, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Cast a tensor to a 32-bit floating point tensor.
        /// </summary>
        /// <param name="t">The input tensor</param>
        /// <returns>'this' if the tensor is already a 32-bit floating point; otherwise, a new tensor.</returns>
        public static Tensor FloatTensor(Tensor t) => t.to(ScalarType.Float32);
    }
}
