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

// Conversion between xsimd::batch and std::simd / std::experimental::simd.
// Compile with -DXSIMD_ENABLE_STD_SIMD. Without the macro, interop is off.

#include <xsimd/xsimd.hpp>

#include <cstddef>
#include <cstdio>
#include <numeric>

namespace
{
    constexpr std::size_t N = xsimd::batch<float>::size;
}

#if defined(XSIMD_WITH_STD_SIMD)
using std_vec = std::simd::vec<float, N>;
static std_vec roundtrip(std_vec v) { return v; }
#elif defined(XSIMD_WITH_STD_EXPERIMENTAL_SIMD)
using std_vec = std::experimental::simd<float, std::experimental::simd_abi::deduce_t<float, N>>;
static std_vec roundtrip(std_vec v) { return v; }
#endif

int main()
{
#if defined(XSIMD_WITH_STD_SIMD) || defined(XSIMD_WITH_STD_EXPERIMENTAL_SIMD)
    float data[N];
    std::iota(data, data + N, 0.f);

    xsimd::batch<float> b = xsimd::batch<float>::load_unaligned(data);
    std_vec r = roundtrip(b); // batch to std type, no cast
    xsimd::batch<float> back = r; // std type to batch, no cast

    bool ok = true;
    for (std::size_t i = 0; i < N; ++i)
    {
        std::printf("%g ", static_cast<double>(back.get(i)));
        if (back.get(i) != static_cast<float>(i))
        {
            ok = false;
        }
    }
    std::printf("\n");
    return ok ? 0 : 1;
#else
    std::printf("std::simd interop is off\n");
    return 0;
#endif
}
