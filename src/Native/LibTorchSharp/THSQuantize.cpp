#include "THSQuantize.h"

Tensor THSQuantize_quantize_per_channel(const Tensor x, const Tensor scales, const Tensor zero_points, int64_t axis, int8_t scalar_type)
{
    CATCH_TENSOR(torch::quantize_per_channel(*x, *scales, *zero_points, axis, at::ScalarType(scalar_type)))
}
Tensor quantize_per_channel_out_scale_axis(const Tensor out, const Tensor self, const Tensor scales, const Tensor zero_points, int64_t axis, int8_t dtype){
    CATCH_TENSOR(torch::quantize_per_channel_out(*out, *self, *scales, *zero_points, axis, at::ScalarType(dtype)))
}
Tensor quantize_per_channel_outf_scale_axis(const Tensor self, const Tensor scales, const Tensor zero_points, int64_t axis, int8_t dtype, const Tensor out) {
    CATCH_TENSOR(torch::quantize_per_channel_outf(*self, *scales, *zero_points, axis, at::ScalarType(dtype), *out))
}

Tensor quantize_per_tensor(const Tensor& self, double scale, int64_t zero_point, int8_t dtype) {
    CATCH_TENSOR(torch::quantize_per_tensor(*self, scale, zero_point, at::ScalarType(dtype)))
}
Tensor quantize_per_tensor_scale(const Tensor& self, const Tensor& scale, const Tensor& zero_point, int8_t dtype){
    CATCH_TENSOR(torch::quantize_per_tensor(*self, *scale, *zero_point, at::ScalarType(dtype)))
}

void quantize_per_tensors(Tensor* tensors, const int length, const Tensor& scales, const Tensor& zero_points, int8_t dtype, Tensor* (*allocator)(size_t length)){
    CATCH(
        auto res = torch::quantize_per_tensor(toTensors<at::Tensor>((torch::Tensor**)tensors, length), *scales, *zero_points, at::ScalarType(dtype));
        const size_t sz = res.size();
        Tensor* result = allocator(sz);
        for (size_t i = 0; i < sz; i++)
            result[i] = new torch::Tensor(res[i]);
    )
}

Tensor quantize_per_tensor_out(Tensor& out, const Tensor& self, double scale, int64_t zero_point, int8_t dtype){
    CATCH_TENSOR(torch::quantize_per_tensor_out(*out, *self, scale, zero_point, at::ScalarType(dtype)));
}
Tensor quantize_per_tensor_outf(const Tensor& self, double scale, int64_t zero_point, int8_t dtype, const Tensor& out){
    CATCH_TENSOR(torch::quantize_per_tensor_outf(*self, scale, zero_point, at::ScalarType(dtype), *out))
}
Tensor quantize_per_tensor_out_scale(Tensor& out, const Tensor& self, const Tensor& scale, const Tensor& zero_point, int8_t dtype){
    CATCH_TENSOR(torch::quantize_per_tensor_out(*out, *self, *scale, *zero_point, at::ScalarType(dtype)))
}
Tensor quantize_per_tensor_outf_scale(const Tensor& self, const Tensor& scale, const Tensor& zero_point, int8_t dtype, Tensor& out){
    CATCH_TENSOR(torch::quantize_per_tensor_outf(*self, *scale, *zero_point, at::ScalarType(dtype), *out))
}
void quantize_per_tensors_out(Tensor* out, const int length_out, Tensor* tensors, const int length_tensors, const Tensor& scales, const Tensor& zero_points, int8_t dtype){
    //Have out reference??? INVESTIGATE
    CATCH(
        torch::quantize_per_tensor_out(
            toTensors<at::Tensor>((torch::Tensor**)out, length_out),
            toTensors<at::Tensor>((torch::Tensor**)tensors, length_tensors),
            *scales,
            *zero_points,
            at::ScalarType(dtype)
        );
    )
}
/*void quantize_per_tensors_outf(Tensor* tensors, const int length, const Tensor& scales, const Tensor& zero_points, int8_t dtype, Tensor* (*allocator)(size_t length)){
    //INVESTIGATE ABOUT OUT PARAMETER
    at::TensorList outs;
    torch::quantize_per_tensor_outf(toTensors<at::Tensor>((torch::Tensor**)tensors, length), *scales, *zero_points, at::ScalarType(dtype), outs);
}*/

