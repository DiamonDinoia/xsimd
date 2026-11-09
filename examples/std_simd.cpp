/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
 * Martin Renou                                                             *
 * Copyright (c) QuantStack                                                 *
 * Copyright (c) Serge Guelton                                              *
 * Copyright (c) Marco Barbone                                              *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

// Soft-threshold denoising then L2 normalization: a std::simd kernel, an xsimd driver loop.

#include <xsimd/xsimd.hpp>

#include <cmath>
#include <cstddef>
#include <cstdio>

namespace
{
    constexpr std::size_t N = xsimd::batch<float>::size;
    constexpr std::size_t len = 2 * N + 3;    constexpr float threshold = 0.3f;
    constexpr double tol = 1e-5;
}

int main()
{
#if defined(XSIMD_WITH_STD_SIMD) || defined(XSIMD_WITH_STD_EXPERIMENTAL_SIMD)
    using std_vec = xsimd::detail::std_simd<float, N>;
    namespace stdx = xsimd::detail::stdx;
    auto soft_threshold = [](std_vec x, float t)
    {
        return stdx::max(x - t, std_vec(0.f)) + stdx::min(x + t, std_vec(0.f));
    };
    float x[len], y[len], ref[len];
    for (std::size_t i = 0; i < len; ++i)
        x[i] = (static_cast<float>((i * 7) % 11) - 5.f) * 0.15f;
    float sq = 0.f;
    std::size_t i = 0;
    for (; i + N <= len; i += N)
    {
        xsimd::batch<float> b = xsimd::load_unaligned(x + i);
        xsimd::batch<float> r = soft_threshold(b, threshold);        sq += xsimd::reduce_add(r * r);
        xsimd::store_unaligned(y + i, r);
    }
    for (; i < len; ++i)
    {
        const float a = std::fabs(x[i]) - threshold;
        y[i] = a > 0.f ? std::copysign(a, x[i]) : 0.f;
        sq += y[i] * y[i];
    }
    const float scale = 1.f / std::sqrt(sq);
    for (std::size_t j = 0; j < len; ++j)
        y[j] *= scale;
    float rsq = 0.f;
    for (std::size_t j = 0; j < len; ++j)
    {
        const float a = std::fabs(x[j]) - threshold;
        ref[j] = a > 0.f ? std::copysign(a, x[j]) : 0.f;
        rsq += ref[j] * ref[j];
    }
    const float rscale = 1.f / std::sqrt(rsq);
    double max_err = 0.;
    for (std::size_t j = 0; j < len; ++j)
    {
        const double r = ref[j] * rscale;
        max_err = std::fmax(max_err, std::fabs(y[j] - r) / (std::fabs(r) > 0. ? std::fabs(r) : 1.));
    }
    std::printf("max relative error %g (tolerance %g)\n", max_err, tol);
    return max_err <= tol ? 0 : 1;
#else
    std::printf("std::simd interop is off\n");
    return 0;
#endif
}
