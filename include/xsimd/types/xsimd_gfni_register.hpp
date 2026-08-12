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

#ifndef XSIMD_GFNI_REGISTER_HPP
#define XSIMD_GFNI_REGISTER_HPP

#include "./xsimd_avx512bw_register.hpp"
#include "./xsimd_avx512vnni_avx512vbmi2_register.hpp"
#include "./xsimd_fma3_avx2_register.hpp"
#include "./xsimd_fma3_sse_register.hpp"

#if XSIMD_WITH_GFNI_SSE4_2 || XSIMD_WITH_GFNI_AVX2 || XSIMD_WITH_GFNI_AVX512BW || XSIMD_WITH_GFNI_AVX512VNNI_AVX512VBMI2
// The GFNI intrinsics live in gfniintrin.h, which only immintrin.h pulls in.
// The SSE chain stops at nmmintrin.h, so the legacy encoded forms would
// otherwise be missing from an SSE-only build.
#include <immintrin.h>
#endif

namespace xsimd
{
    template <typename arch>
    struct gfni;

    // The bases are the FMA3 flavours rather than the plain ones so the FMA3
    // kernels stay in the dispatch chain, following avxvnni. fma3<T> always
    // derives from T and only registers kernels when FMA3 is enabled, so the
    // base is transparent when it is not.

    /**
     * @ingroup architectures
     *
     * SSE4.2 + GFNI instructions, legacy encoded
     */
    template <>
    struct gfni<sse4_2> : fma3<sse4_2>
    {
        static constexpr bool supported() noexcept { return XSIMD_WITH_GFNI_SSE4_2; }
        static constexpr bool available() noexcept { return true; }
        static constexpr char const* name() noexcept { return "gfni+sse4.2"; }
    };

    /**
     * @ingroup architectures
     *
     * AVX2 + GFNI instructions, VEX encoded
     */
    template <>
    struct gfni<avx2> : fma3<avx2>
    {
        static constexpr bool supported() noexcept { return XSIMD_WITH_GFNI_AVX2; }
        static constexpr bool available() noexcept { return true; }
        static constexpr char const* name() noexcept { return "gfni+avx2"; }
    };

    /**
     * @ingroup architectures
     *
     * AVX512BW + GFNI instructions, EVEX encoded
     */
    template <>
    struct gfni<avx512bw> : avx512bw
    {
        static constexpr bool supported() noexcept { return XSIMD_WITH_GFNI_AVX512BW; }
        static constexpr bool available() noexcept { return true; }
        static constexpr char const* name() noexcept { return "gfni+avx512bw"; }
    };

    /**
     * @ingroup architectures
     *
     * AVX512VNNI + AVX512VBMI2 + GFNI instructions
     *
     * Every shipping AVX512 part that carries GFNI also carries VBMI2 and VNNI,
     * so without this tier the dispatcher would always prefer the richer AVX512
     * chain and GFNI would never be reached on the hardware that has it. This
     * mirrors avx512vnni, which is instantiated twice for the same reason.
     */
    template <>
    struct gfni<avx512vnni<avx512vbmi2>> : avx512vnni<avx512vbmi2>
    {
        static constexpr bool supported() noexcept { return XSIMD_WITH_GFNI_AVX512VNNI_AVX512VBMI2; }
        static constexpr bool available() noexcept { return true; }
        static constexpr char const* name() noexcept { return "gfni+avx512vnni+avx512vbmi2"; }
    };

#if XSIMD_WITH_GFNI_SSE4_2

    namespace types
    {
        XSIMD_DECLARE_SIMD_REGISTER_ALIAS(gfni<sse4_2>, sse4_2);
    }
#endif

#if XSIMD_WITH_GFNI_AVX2

    namespace types
    {
        XSIMD_DECLARE_SIMD_REGISTER_ALIAS(gfni<avx2>, avx2);
    }
#endif

#if XSIMD_WITH_GFNI_AVX512BW

#if !XSIMD_WITH_AVX512BW
#error "architecture inconsistency: gfni+avx512bw requires avx512bw"
#endif

    namespace types
    {
        template <class T>
        struct get_bool_simd_register<T, gfni<avx512bw>>
        {
            using type = simd_avx512_bool_register<T>;
        };

        XSIMD_DECLARE_SIMD_REGISTER_ALIAS(gfni<avx512bw>, avx512bw);
    }
#endif

#if XSIMD_WITH_GFNI_AVX512VNNI_AVX512VBMI2

    namespace types
    {
        template <class T>
        struct get_bool_simd_register<T, gfni<avx512vnni<avx512vbmi2>>>
        {
            using type = simd_avx512_bool_register<T>;
        };

        XSIMD_DECLARE_SIMD_REGISTER_ALIAS(gfni<avx512vnni<avx512vbmi2>>, avx512bw);
    }
#endif
}

#endif
