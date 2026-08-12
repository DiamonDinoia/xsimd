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

#ifndef XSIMD_COMMON_BITWISE_HPP
#define XSIMD_COMMON_BITWISE_HPP

#include "../../types/xsimd_batch_constant.hpp"
#include "./xsimd_common_details.hpp"

#include <climits>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace xsimd
{

    namespace kernel
    {

        using namespace types;

        namespace detail
        {
            // bit i is set when i / s is even: 0x55.. for s == 1, 0x33.. for
            // s == 2, 0x0f0f.. for s == 4, and so on up to s == bits / 2
            template <class U>
            constexpr U alternating_mask(unsigned s) noexcept
            {
                U m = 0;
                for (unsigned i = 0; i < sizeof(U) * CHAR_BIT; ++i)
                    if (((i / s) & 1u) == 0u)
                        m = U(m | U(U(1) << i));
                return m;
            }
        }

        // popcount
        template <class A, class T, class /*=std::enable_if_t<std::is_integral_v<T>>*/>
        XSIMD_INLINE batch<T, A> popcount(batch<T, A> const& self, requires_arch<common>) noexcept
        {
            using U = as_unsigned_integer_t<T>;
            using b_type = batch<U, A>;
            constexpr std::size_t bits = sizeof(T) * CHAR_BIT;

            b_type x = bitwise_cast<U>(self);
            x = x - ((x >> 1) & b_type(detail::alternating_mask<U>(1)));
            b_type const m2(detail::alternating_mask<U>(2));
            x = (x & m2) + ((x >> 2) & m2);
            x = (x + (x >> 4)) & b_type(detail::alternating_mask<U>(4));
            if constexpr (bits >= 16)
                x = (x + (x >> 8)) & b_type(detail::alternating_mask<U>(8));
            if constexpr (bits >= 32)
                x = (x + (x >> 16)) & b_type(detail::alternating_mask<U>(16));
            if constexpr (bits >= 64)
                x = (x + (x >> 32)) & b_type(detail::alternating_mask<U>(32));
            return bitwise_cast<T>(x);
        }
    }
}

#endif
