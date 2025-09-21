// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#pragma once

#include "../Stdafx.h"

#include "torch/torch.h"

#include "Utils.h"

EXPORT_API(Tensor) THSQuantize_quantize_per_channel(const Tensor x, const Tensor scales, const Tensor zero_points, int64_t axis, int8_t dtype);

EXPORT_API(Tensor) quantize_per_channel_out_scale_axis(const Tensor out, const Tensor self, const Tensor scales, const Tensor zero_points, int64_t axis, int8_t dtype);
EXPORT_API(Tensor) quantize_per_channel_outf_scale_axis(const Tensor self, const Tensor scales, const Tensor zero_points, int64_t axis, int8_t dtype, const Tensor out);

EXPORT_API(Tensor) quantize_per_tensor(const Tensor& self, double scale, int64_t zero_point, int8_t dtype);
EXPORT_API(Tensor) quantize_per_tensor_scale(const Tensor& self, const Tensor& scale, const Tensor& zero_point, int8_t dtype);

EXPORT_API(Tensor) quantize_per_tensor_out(Tensor& out, const Tensor& self, double scale, int64_t zero_point, int8_t dtype);
EXPORT_API(Tensor) quantize_per_tensor_out_scale(Tensor& out, const Tensor& self, const Tensor& scale, const Tensor& zero_point, int8_t dtype);

EXPORT_API(Tensor) quantize_per_tensor_outf(const Tensor& self, double scale, int64_t zero_point, int8_t dtype, const Tensor& out);
EXPORT_API(Tensor) quantize_per_tensor_outf_scale(const Tensor& self, const Tensor& scale, const Tensor& zero_point, int8_t dtype, Tensor& out);

EXPORT_API(void) quantize_per_tensors(Tensor* tensors, const int length_tensors, const Tensor& scales, const Tensor& zero_points, int8_t dtype, Tensor* (*allocator)(size_t length));
EXPORT_API(void) quantize_per_tensors_out(Tensor* out, const int length_out, Tensor* tensors, const int length_tensors, const Tensor& scales, const Tensor& zero_points, int8_t dtype);
//EXPORT_API(void) quantize_per_tensors_outf(Tensor* tensors, const int length_tensors, const Tensor& scales, const Tensor& zero_points, int8_t dtype, Tensor* (*allocator)(size_t length));