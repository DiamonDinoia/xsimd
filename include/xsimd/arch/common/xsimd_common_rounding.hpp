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

#ifndef XSIMD_COMMON_ROUNDING_HPP
#define XSIMD_COMMON_ROUNDING_HPP

#include "./xsimd_common_details.hpp"

namespace xsimd
{

    namespace kernel
    {

        using namespace types;

        // ceil
        template <class A, class T>
        XSIMD_INLINE batch<T, A> ceil(batch<T, A> const& self, requires_arch<common>) noexcept
        {
            batch<T, A> truncated_self = trunc(self);
            auto need = truncated_self < self;
            batch<T, A> step = bitwise_and(bitwise_cast<T>(need), batch<T, A>(1));
            // the fence keeps the compiler from blending +1 onto every lane, where 2^digits + 1 rounds and raises
            detail::fenv_fence(step);
            return select(need, truncated_self + step, truncated_self);
        }

        // floor
        template <class A, class T>
        XSIMD_INLINE batch<T, A> floor(batch<T, A> const& self, requires_arch<common>) noexcept
        {
            batch<T, A> truncated_self = trunc(self);
            auto need = truncated_self > self;
            batch<T, A> step = bitwise_and(bitwise_cast<T>(need), batch<T, A>(1));
            // the fence keeps the compiler from blending -1 onto every lane, where -2^digits - 1 rounds and raises
            detail::fenv_fence(step);
            return select(need, truncated_self - step, truncated_self);
        }

        // round
        template <class A, class T>
        XSIMD_INLINE batch<T, A> round(batch<T, A> const& self, requires_arch<common>) noexcept
        {
            // Ties away from zero. c - 0.5 and c - 1 round on huge lanes, so those
            // lanes go through as zero and are put back by the last select.
            auto v = abs(self);
            auto big = !(v < constants::twotonmb<batch<T, A>>()); // also catches nan
            auto vs = select(big, batch<T, A>(0), v);
            auto c = ceil(vs);
            auto cp = select(c - 0.5 > vs, c - 1, c);
            return select(big, self, copysign(cp, self));
        }

        // trunc
        template <class A, class T, class = std::enable_if_t<std::is_integral_v<T>>>
        XSIMD_INLINE batch<T, A> trunc(batch<T, A> const& self, requires_arch<common>) noexcept
        {
            return self;
        }
        template <class A>
        XSIMD_INLINE batch<float, A> trunc(batch<float, A> const& self, requires_arch<common>) noexcept
        {
            return detail::truncf_impl(self);
        }
        template <class A>
        XSIMD_INLINE batch<double, A> trunc(batch<double, A> const& self, requires_arch<common>) noexcept
        {
            return detail::truncf_impl(self);
        }

    }

}

#endif
