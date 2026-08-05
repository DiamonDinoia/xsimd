// AVX-512 registrations. avx512bw refines avx512f with the byte/word forms.
#ifndef PROTO_AVX512_HPP
#define PROTO_AVX512_HPP

#include "proto_ops.hpp"

#if XSIMD_WITH_AVX512F
namespace proto
{
    template <class T>
    constexpr auto table(add_t, tag<T>, avx512f) noexcept
    {
        return by_type<T>(
            [](auto l, auto r) { return _mm512_add_ps(l, r); },
            [](auto l, auto r) { return _mm512_add_pd(l, r); },
            nullptr, nullptr,
            [](auto l, auto r) { return _mm512_add_epi32(l, r); },
            [](auto l, auto r) { return _mm512_add_epi64(l, r); });
    }
}
#endif

#if XSIMD_WITH_AVX512BW
namespace proto
{
    template <class T>
    constexpr auto table(add_t, tag<T>, avx512bw) noexcept
    {
        return by_type<T>(
            [](auto l, auto r) { return _mm512_add_ps(l, r); },
            [](auto l, auto r) { return _mm512_add_pd(l, r); },
            [](auto l, auto r) { return _mm512_add_epi8(l, r); },
            [](auto l, auto r) { return _mm512_add_epi16(l, r); },
            [](auto l, auto r) { return _mm512_add_epi32(l, r); },
            [](auto l, auto r) { return _mm512_add_epi64(l, r); });
    }
}
#endif

#endif
