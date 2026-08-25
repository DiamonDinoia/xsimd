// Type-driven fallback to an emulated register for value types with no SIMD
// register on the default architecture (long double today: no ISA has 80-bit
// lanes). Everything stays architecture-generic behind the same batch API:
// agents writing vector kernels target real archs, and future native support
// switches back on by itself.
//
// Enabled by XSIMD_ENABLE_EMULATED_TYPES. When defined, the default template
// argument of batch/batch_bool/batch_constant resolves through the dispatch
// below instead of XSIMD_BATCH_DEFAULT_ARCH_IMPL.
#ifndef XSIMD_EMULATED_AUTO_HPP
#define XSIMD_EMULATED_AUTO_HPP

#include <cstddef>
#include <type_traits>

#include "../config/xsimd_arch.hpp"
#include "./xsimd_emulated_register.hpp"
#include "./xsimd_register.hpp"

namespace xsimd
{
    namespace detail
    {
        // Native vector width in bits for architecture A, measured through its
        // double batch (every mainstream arch has one). Fallback to 128 for
        // archs without one (e.g. the scalar-ish generic only).
        template <class A, class = void>
        struct arch_bits : std::integral_constant<std::size_t, 128>
        {
        };
        template <class A>
        struct arch_bits<A, std::void_t<decltype(types::simd_register<double, A>())>>
            : std::integral_constant<std::size_t,
                                     sizeof(types::simd_register<double, A>) * 8>
        {
        };

        // When T has no real register under A, the batch behaves as an
        // emulated register of A's native width: lane_count(= bits/8/sizeof(T)).
        template <class T, class A, class = void>
        struct arch_or_emulated
        {
            using type = A;
        };
        template <class T, class A>
        struct arch_or_emulated<T, A,
                                std::enable_if_t<!types::has_simd_register<T, A>::value>>
        {
            using type = emulated<arch_bits<A>::value>;
        };

        template <class T, class A>
        using arch_or_emulated_t = typename arch_or_emulated<T, A>::type;
    }
}

#endif // XSIMD_EMULATED_AUTO_HPP
