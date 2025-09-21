#include "THSHistogram.h"

void THSHistogram_histogram(const Tensor input, const int64_t bins, double* ranges, int64_t length, Tensor weight, bool density, Tensor hist, Tensor hist_bins)
{
    CATCH(
        const auto tup = torch::histogram(*input, bins, ranges == nullptr ? std::nullopt : at::ArrayRef<double>(ranges, length), *weight, density);
        //This reference is okey??? TEST THIS
        *hist = std::get<0>(tup);
        *hist_bins = std::get<1>(tup);
    )
}

void THSHistogram_histogram_tensor(const Tensor input, const Tensor bins, const double* ranges, int64_t length, Tensor weight, bool density, Tensor hist, Tensor hist_bins)
{
    CATCH(
        auto tup = torch::histogram(*input, *bins, ranges == nullptr ? std::nullopt : at::ArrayRef<double>(ranges, length), *weight, density);
        //This reference is okey??? TEST THIS
        *hist = std::get<0>(tup);
        *hist_bins = std::get<1>(tup);
    )
}

void THSHistogram_histogramdd(const Tensor input, const int64_t* bins, int64_t length, const double* ranges, int64_t length_ranges, Tensor weight, bool density, Tensor hist, Tensor*(*allocator)(size_t length_loc))
{
    CATCH(
        auto tup = torch::histogramdd(*input, at::IntArrayRef(bins, length), ranges == nullptr ? std::nullopt : at::ArrayRef<double>(ranges, length_ranges), *weight, density);
        *hist = std::get<0>(tup);
        auto res = std::get<1>(tup);
        const size_t sz = res.size();
        Tensor* result = allocator(sz);
        for (size_t i = 0; i < sz; i++)
            result[i] = new torch::Tensor(res[i]);
    )
}



void THSHistogram_histogramdd_intbins(const Tensor input, int64_t bins, const double* ranges, int64_t length_ranges, Tensor weight, bool density, Tensor hist, Tensor* (*allocator)(size_t length_loc))
{
    CATCH(
        auto tup = torch::histogramdd(*input, bins, ranges == nullptr ? std::nullopt : at::ArrayRef<double>(ranges, length_ranges), *weight, density);
        *hist = std::get<0>(tup);
        auto res = std::get<1>(tup);
        const size_t sz = res.size();
        Tensor * result = allocator(sz);
        for (size_t i = 0; i < sz; i++)
            result[i] = new torch::Tensor(res[i]);
    )
}


void THSHistogram_histogramdd_tensors(const Tensor input, Tensor* tensors, int64_t length, const double* ranges, int64_t length_ranges, Tensor weight, bool density, Tensor hist, Tensor* (*allocator)(size_t length_loc))
{
    CATCH(
        auto tup = torch::histogramdd(*input, toTensors<at::Tensor>((torch::Tensor**)tensors, length), ranges == nullptr ? std::nullopt : at::ArrayRef<double>(ranges, length_ranges), *weight, density);
        *hist = std::get<0>(tup);
        auto res = std::get<1>(tup);
        const size_t sz = res.size();
        Tensor * result = allocator(sz);
        for (size_t i = 0; i < sz; i++)
            result[i] = new torch::Tensor(res[i]);
    )
}