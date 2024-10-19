using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using TorchSharp.PInvoke;
using TorchSharp.Utils;

namespace TorchSharp.Amp
{
    /*public static class Autocast
    {
        public static torch.Tensor AutoCast(this torch.Tensor input)
        {
            return AutocastMode.GetInstance().CastTensor(input);
        }
    }*/
    //TODO: Should make Singleton and IDisposable on ENTER
    public sealed class AutocastMode : IDisposable
    {
        public bool _enabled=false;
        public bool IsEnter { private set; get; }=false;
        public bool IsDisposed = false;
        private bool prev_cache_enabled, prev;
        private torch.ScalarType prev_fastdtype;
        //internal bool Prev;
        private bool _cache_enabled=false;
        internal torch.ScalarType fast_dtype = torch.ScalarType.Float32;
        internal torch.ScalarType? dtype = torch.ScalarType.Float32;
        public DeviceType device = DeviceType.CUDA;
        private static AutocastMode instance;
        public static AutocastMode GetInstance(bool enabled=false)
        {
            return instance ??= new AutocastMode(torch.cuda_is_available() ? torch.CUDA : torch.CPU, enabled:enabled,cache_enabled:true);
        }

        private AutocastMode(torch.Device dev, torch.ScalarType? dtype = null, bool enabled=true, bool? cache_enabled = null)
        {
            //https://pytorch.org/docs/stable/amp.html#cuda-ops-that-can-autocast-to-float16
            if (dtype == null)
                dtype = torch.get_autocast_dtype(dev.type);
            this.device = dev.type;
            if (!torch.is_autocast_available(device))
                throw new Exception($"User specified an unsupported autocast device_type {device}");
            fast_dtype = torch.get_autocast_dtype(device);
            _cache_enabled = torch.is_autocast_cache_enabled();
            if (enabled && !torch.cuda_is_available() && dev.type == DeviceType.CUDA) //Is not available for doing multicast
                enabled = false;
            if (this.dtype.HasValue)
                fast_dtype = dtype.Value;
            if (cache_enabled.HasValue)
                _cache_enabled = cache_enabled.Value;

            if (dev.type == DeviceType.CPU) {
                if (fast_dtype != torch.ScalarType.Float16 || fast_dtype != torch.ScalarType.BFloat16) {
                    Debug.WriteLine($"In CPU autocast, but the target d type is not suported. Disabling autocast. CPU autocast only supports dtype of {torch.ScalarType.Float16} or {torch.ScalarType.BFloat16}");
                    enabled = false;
                }
            } else if (dev.type == DeviceType.CUDA) {

                if (enabled && fast_dtype == torch.ScalarType.BFloat16 && !torch.cuda.is_bf16_supported())
                    throw new Exception("Current CUDA Device does not support bfloat16. Please switch dtype to float16.");
            }
            this._enabled = enabled;
        }

        public torch.ScalarType GetFastType()
        {
            return torch.get_autocast_dtype(device);
        }
        private static torch.ScalarType GetDtype(IntPtr handle)
        {
            return (torch.ScalarType)NativeMethods.THSTensor_type(handle);
        }

        public static IntPtr AutoCast(IntPtr handle)
        {
            return ToIf(handle, GetInstance().GetFastType());
        }
        public static (IntPtr h1, IntPtr h2) AutoCast(IntPtr handle1, IntPtr handle2)
        {
            var ft = GetInstance().GetFastType();
            return (ToIf(handle1, ft), ToIf(handle2, ft));
        }
        public static (IntPtr h1, IntPtr h2, IntPtr h3) AutoCast(IntPtr handle1, IntPtr handle2, IntPtr handle3)
        {
            var ft = GetInstance().GetFastType();
            return (ToIf(handle1, ft), ToIf(handle2, ft), ToIf(handle3, ft));
        }
        public static (IntPtr h1, IntPtr h2) AutoCast(IntPtr handle1, IntPtr handle2, torch.ScalarType dtype)
        {
            return (ToIf(handle1, dtype), ToIf(handle2, dtype));
        }

        public static (IntPtr h1, IntPtr h2, IntPtr h3) AutoCast(IntPtr handle1, IntPtr handle2, IntPtr handle3, torch.ScalarType dtype)
        {
            return (ToIf(handle1, dtype), ToIf(handle2, dtype), ToIf(handle3, dtype));
        }

        public static IntPtr AutoCast(IntPtr handle, torch.ScalarType dtype)
        {
            return ToIf(handle, dtype);
        }

        public static torch.Tensor AutoCast(torch.Tensor tensor)
        {
            return new torch.Tensor(AutoCast(tensor.Handle));
            //return tensor.to(AutocastMode.GetInstance().GetFastType());
        }
        public static IntPtr To(IntPtr ptr, torch.ScalarType type)
        {
            Debug.WriteLine($"{nameof(AutocastMode)} Tensor converting from: {GetDtype(ptr)} to: {type}");
            var res = NativeMethods.THSTensor_to_type(ptr, (sbyte)type);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return res;
        }

        private static DeviceType GetDeviceType(IntPtr ptr)
        {
            return (DeviceType)NativeMethods.THSTensor_device_type(ptr);
        }
        public static IntPtr ToIf(IntPtr ptr, torch.ScalarType type)
        {
            if (!GetInstance()._enabled || !GetInstance().IsEnter)
                return ptr;
            if (GetDtype(ptr) == type) //if already have same dtype is not necesary convert to dtype, right???
                return ptr;
            /*if (!NativeMethods.THSAmp_is_autocast_enabled(NativeMethods.THSTensor_device_type(ptr)))
                return ptr;*/
            var res = NativeMethods.THSTensor_to_type(ptr, (sbyte)type);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return res;
        }
        public static IntPtr ToIf(IntPtr ptr, torch.ScalarType type, DeviceType device_type)
        {
            bool is_elegible = GetDtype(ptr) != torch.ScalarType.Float64 && GetDeviceType(ptr) == device_type;

            if (!NativeMethods.THSAmp_is_autocast_enabled(NativeMethods.THSTensor_device_type(ptr)))
                return ptr;
            var res = NativeMethods.THSTensor_to_type(ptr, (sbyte)type);
            if (res == IntPtr.Zero)
                torch.CheckForErrors();
            return res;
        }

        public static bool IsAutocastEnabled(DeviceType device = DeviceType.CUDA)
        {
            return torch.is_autocast_enabled(!torch.cuda_is_available() ? DeviceType.CPU : device);
        }

        public IDisposable Enter()
        {
            prev_cache_enabled = torch.is_autocast_cache_enabled();
            prev = torch.is_autocast_enabled(device);
            prev_fastdtype = torch.get_autocast_dtype(device);
            torch.set_autocast_enabled(device, _enabled);
            torch.set_autocast_dtype(device, fast_dtype);
            torch.autocast_increment_nesting();
            torch.set_autocast_cache_enabled(_cache_enabled);
            IsEnter = true;
            return this;
        }

        private void Dispose(bool disposing)
        {
            IsEnter = false;
            this._enabled = false;
            if (torch.autocast_decrement_nesting() == 0)
                torch.clear_autocast_cache();
            torch.set_autocast_enabled(device, prev);
            torch.set_autocast_dtype(device, prev_fastdtype);
            torch.set_autocast_cache_enabled(prev_cache_enabled);
        }

        /*~AutocastMode()
        {

        }*/
        
        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
}