#ifndef PROTO_ADD_HPP
#define PROTO_ADD_HPP

#include "proto_dispatch.hpp"

namespace proto
{
    /// Recipe for `add`: the native intrinsic table, and the entry point used
    /// when a missing entry has to be serviced by recursing on halves.
    struct add_op
    {
        template <class T, class A>
        static constexpr auto w128() noexcept
        {
            return by_type<T>(
                [](auto l, auto r) { return _mm_add_ps(l, r); },
                [](auto l, auto r) { return _mm_add_pd(l, r); },
                [](auto l, auto r) { return _mm_add_epi8(l, r); },
                [](auto l, auto r) { return _mm_add_epi16(l, r); },
                [](auto l, auto r) { return _mm_add_epi32(l, r); },
                [](auto l, auto r) { return _mm_add_epi64(l, r); });
        }

        template <class T, class A>
        static constexpr auto w256() noexcept
        {
#if XSIMD_WITH_AVX
            constexpr bool i = std::is_base_of_v<avx2, A>;
            return by_type<T>(
                [](auto l, auto r) { return _mm256_add_ps(l, r); },
                [](auto l, auto r) { return _mm256_add_pd(l, r); },
                only_if<i>([](auto l, auto r) { return _mm256_add_epi8(l, r); }),
                only_if<i>([](auto l, auto r) { return _mm256_add_epi16(l, r); }),
                only_if<i>([](auto l, auto r) { return _mm256_add_epi32(l, r); }),
                only_if<i>([](auto l, auto r) { return _mm256_add_epi64(l, r); }));
#else
            return nullptr;
#endif
        }

        template <class T, class A>
        static constexpr auto w512() noexcept
        {
#if XSIMD_WITH_AVX512F
            constexpr bool bw = std::is_base_of_v<avx512bw, A>;
            return by_type<T>(
                [](auto l, auto r) { return _mm512_add_ps(l, r); },
                [](auto l, auto r) { return _mm512_add_pd(l, r); },
                only_if<bw>([](auto l, auto r) { return _mm512_add_epi8(l, r); }),
                only_if<bw>([](auto l, auto r) { return _mm512_add_epi16(l, r); }),
                [](auto l, auto r) { return _mm512_add_epi32(l, r); },
                [](auto l, auto r) { return _mm512_add_epi64(l, r); });
#else
            return nullptr;
#endif
        }

        template <class T, class A>
        static XSIMD_INLINE constexpr auto native() noexcept
        {
            return by_width<A>(w512<T, A>(), w256<T, A>(), w128<T, A>());
        }

        template <class T, class A>
        static XSIMD_INLINE batch<T, A> apply(batch<T, A> a, batch<T, A> b) noexcept
        {
            return dispatch<add_op, T, A>(a, b);
        }

    };

    template <class T, class A>
    XSIMD_INLINE batch<T, A> add(batch<T, A> a, batch<T, A> b) noexcept
    {
        return add_op::apply(a, b);
    }
}

#endif
