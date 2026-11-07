/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
 * Martin Renou                                                             *
 * Copyright (c) QuantStack                                                 *
 * Copyright (c) Serge Guelton                                              *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

// Soft-threshold denoising then L2 normalization: a std::simd kernel, an xsimd driver loop.
// Compile with -DXSIMD_ENABLE_STD_SIMD. Without the macro, interop is off.

#include <xsimd/xsimd.hpp>

#include <cmath>
#include <cstddef>
#include <cstdio>

namespace
{
    constexpr std::size_t N = xsimd::batch<float>::size;
    constexpr std::size_t len = 2 * N + 3; // not a multiple of N, so the tail runs
    constexpr float threshold = 0.3f;
    constexpr double tol = 1e-5;

#if defined(XSIMD_WITH_STD_SIMD)
    using std_vec = std::simd::vec<float, N>;
#elif defined(XSIMD_WITH_STD_EXPERIMENTAL_SIMD)
    using std_vec = std::experimental::simd<float, std::experimental::simd_abi::deduce_t<float, N>>;
#endif

#if defined(XSIMD_WITH_STD_SIMD) || defined(XSIMD_WITH_STD_EXPERIMENTAL_SIMD)
    // third-party kernel: the standard API only, it knows nothing of xsimd
    std_vec soft_threshold(std_vec x, float t)
    {
#if defined(XSIMD_WITH_STD_SIMD)
        std_vec m = std::simd::max(std::simd::select(x < 0.f, -x, x) - t, std_vec(0.f));
        return std::simd::select(x < 0.f, -m, m);
#elif defined(XSIMD_WITH_STD_EXPERIMENTAL_SIMD)
        std_vec m = x;
        std::experimental::where(x < 0.f, m) = -m;
        m = std::experimental::max(m - t, std_vec(0.f));
        std::experimental::where(x < 0.f, m) = -m;
        return m;
#endif
    }
#endif
}

int main()
{
#if defined(XSIMD_WITH_STD_SIMD) || defined(XSIMD_WITH_STD_EXPERIMENTAL_SIMD)
    float x[len], y[len], ref[len];
    for (std::size_t i = 0; i < len; ++i)
        x[i] = (static_cast<float>((i * 7) % 11) - 5.f) * 0.15f;
    // xsimd driver: batched soft-threshold and sum of squares, scalar tail
    float sq = 0.f;
    std::size_t i = 0;
    for (; i + N <= len; i += N)
    {
        xsimd::batch<float> b = xsimd::load_unaligned(x + i);
        xsimd::batch<float> r = soft_threshold(b, threshold); // no casts: batch to std type and back
        sq += xsimd::reduce_add(r * r);
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
    // scalar reference: the same algorithm in plain arithmetic
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
