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

#ifndef XSIMD_GFNI_HPP
#define XSIMD_GFNI_HPP

#include "../types/xsimd_gfni_register.hpp"

#include <cstdint>
#include <type_traits>

namespace xsimd
{

    namespace kernel
    {
        using namespace types;

        namespace detail
        {
            // The encoding follows the register width, and the branches not
            // taken are discarded before instantiation, so the wider intrinsics
            // are never referenced on a target that lacks them.
            template <uint8_t Xor, class A, class T>
            XSIMD_INLINE batch<T, A> gfni_affine(batch<T, A> const& self, batch<uint64_t, A> const& matrix) noexcept
            {
                constexpr std::size_t width = batch<uint8_t, A>::size;
                if constexpr (width == 16)
                    return bitwise_cast<T>(batch<uint8_t, A>(_mm_gf2p8affine_epi64_epi8(self, matrix, Xor)));
                else if constexpr (width == 32)
                    return bitwise_cast<T>(batch<uint8_t, A>(_mm256_gf2p8affine_epi64_epi8(self, matrix, Xor)));
                else
                    return bitwise_cast<T>(batch<uint8_t, A>(_mm512_gf2p8affine_epi64_epi8(self, matrix, Xor)));
            }

            // byte i of an element takes the byte mirrored across that element
            template <std::size_t S>
            struct gfni_byte_reverse
            {
                static constexpr uint8_t get(std::size_t i, std::size_t) noexcept
                {
                    return uint8_t((i / S) * S + (S - 1 - i % S));
                }
            };

            // requires_arch<A> is std::add_const_t<A>&, a non-deduced context,
            // so the tier cannot be pattern-matched from the tag. A is deduced
            // from the batch instead and the tag only has to rank better than
            // requires_arch<common>, which it does: exact match beats the
            // derived-to-base conversion.
            template <class A>
            struct is_gfni : std::false_type
            {
            };

            template <class Base>
            struct is_gfni<gfni<Base>> : std::true_type
            {
            };

            template <class A, class R>
            using enable_gfni_t = std::enable_if_t<is_gfni<A>::value, R>;
        }

        template <uint8_t Xor, class A, class T>
        XSIMD_INLINE detail::enable_gfni_t<A, batch<T, A>> bit_matmul(batch<T, A> const& self, batch<uint64_t, A> const& matrix, requires_arch<A>) noexcept
        {
            static_assert(std::is_integral_v<T>, "bit_matmul requires an integral batch");
            return detail::gfni_affine<Xor>(self, matrix);
        }

        // a permutation is a GF(2) matrix with one bit set per row, so the
        // whole rearrangement is a single VGF2P8AFFINEQB on a constant
        template <class A, class T, uint8_t... Vs>
        XSIMD_INLINE detail::enable_gfni_t<A, batch<T, A>> bit_permute(batch<T, A> const& self, bit_permute_constant<Vs...>, requires_arch<A>) noexcept
        {
            return detail::gfni_affine<0>(self, batch<uint64_t, A>(bit_permute_constant<Vs...>::as_gf2_matrix()));
        }

        // mirror the bits of every byte with the anti-diagonal matrix, then
        // mirror the byte order inside each element
        template <class A, class T>
        XSIMD_INLINE detail::enable_gfni_t<A, batch<T, A>> bit_reverse(batch<T, A> const& self, requires_arch<A>) noexcept
        {
            static_assert(std::is_integral_v<T>, "bit_reverse requires an integral batch");
            auto bits = detail::gfni_affine<0>(self, batch<uint64_t, A>(UINT64_C(0x8040201008040201)));
            if constexpr (sizeof(T) == 1)
                return bits;
            else
                return bitwise_cast<T>(swizzle(bitwise_cast<uint8_t>(bits),
                                               make_batch_constant<uint8_t, detail::gfni_byte_reverse<sizeof(T)>, A>(), A {}));
        }
    }
}

#endif
