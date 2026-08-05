// SSE2 registrations. Also serves avx_128 / avx2_128 / avx512vl_128, which
// derive from sse2 and so bind to these overloads by derived-to-base conversion.
#ifndef PROTO_SSE2_HPP
#define PROTO_SSE2_HPP

#include "proto_ops.hpp"

namespace proto
{
    template <class T>
    constexpr auto table(add_t, tag<T>, sse2) noexcept
    {
        return by_type<T>(
            [](auto l, auto r) { return _mm_add_ps(l, r); },
            [](auto l, auto r) { return _mm_add_pd(l, r); },
            [](auto l, auto r) { return _mm_add_epi8(l, r); },
            [](auto l, auto r) { return _mm_add_epi16(l, r); },
            [](auto l, auto r) { return _mm_add_epi32(l, r); },
            [](auto l, auto r) { return _mm_add_epi64(l, r); });
    }
}

#endif
