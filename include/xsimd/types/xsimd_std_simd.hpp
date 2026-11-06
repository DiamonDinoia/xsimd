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

#ifndef XSIMD_STD_SIMD_HPP
#define XSIMD_STD_SIMD_HPP

#include <cstddef>

#ifdef XSIMD_ENABLE_STD_SIMD

#if __cplusplus > 202302L && __has_include(<simd>)
#include <simd>
// shortcut: GCC 16.2 defines only __glibcxx_simd, drop that check once GCC defines __cpp_lib_simd
#if defined(__cpp_lib_simd) || defined(__glibcxx_simd)
#define XSIMD_WITH_STD_SIMD 1
#endif
#endif

#ifndef XSIMD_WITH_STD_SIMD
#if __has_include(<experimental/simd>)
#include <experimental/simd>
#if defined(__cpp_lib_experimental_parallel_simd)
#define XSIMD_WITH_STD_EXPERIMENTAL_SIMD 1
#endif
#endif
#endif

#if !defined(XSIMD_WITH_STD_SIMD) && !defined(XSIMD_WITH_STD_EXPERIMENTAL_SIMD)
#error "XSIMD_ENABLE_STD_SIMD: no std::simd or std::experimental::simd in this standard library"
#endif

namespace xsimd
{
    namespace detail
    {
#if defined(XSIMD_WITH_STD_SIMD)
        template <class T, std::size_t N>
        using std_simd = std::simd::vec<T, N>;
        template <class T, std::size_t N>
        using std_simd_mask = std::simd::mask<T, N>;
#else
        template <class T, std::size_t N>
        using std_simd = std::experimental::simd<T, std::experimental::simd_abi::deduce_t<T, N>>;
        template <class T, std::size_t N>
        using std_simd_mask = std::experimental::simd_mask<T, std::experimental::simd_abi::deduce_t<T, N>>;
#endif
    }
}

#endif // XSIMD_ENABLE_STD_SIMD

#endif
