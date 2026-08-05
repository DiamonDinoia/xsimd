// AVX registrations. avx2 refines avx: the exact-match overload wins, so an
// avx2 batch gets the integer intrinsics and a plain avx batch does not.
#ifndef PROTO_AVX_HPP
#define PROTO_AVX_HPP

#include "proto_ops.hpp"

#if XSIMD_WITH_AVX
namespace proto
{
    template <class T>
    constexpr auto table(add_t, tag<T>, avx) noexcept
    {
        return by_type<T>(
            [](auto l, auto r) { return _mm256_add_ps(l, r); },
            [](auto l, auto r) { return _mm256_add_pd(l, r); },
            nullptr, nullptr, nullptr, nullptr);
    }
}
#endif

#if XSIMD_WITH_AVX2
namespace proto
{
    template <class T>
    constexpr auto table(add_t, tag<T>, avx2) noexcept
    {
        return by_type<T>(
            [](auto l, auto r) { return _mm256_add_ps(l, r); },
            [](auto l, auto r) { return _mm256_add_pd(l, r); },
            [](auto l, auto r) { return _mm256_add_epi8(l, r); },
            [](auto l, auto r) { return _mm256_add_epi16(l, r); },
            [](auto l, auto r) { return _mm256_add_epi32(l, r); },
            [](auto l, auto r) { return _mm256_add_epi64(l, r); });
    }
}
#endif

#endif
