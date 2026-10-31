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
            return select(truncated_self < self, truncated_self + 1, truncated_self);
        }

        // floor
        template <class A, class T>
        XSIMD_INLINE batch<T, A> floor(batch<T, A> const& self, requires_arch<common>) noexcept
        {
            batch<T, A> truncated_self = trunc(self);
            return select(truncated_self > self, truncated_self - 1, truncated_self);
        }

        // round
        template <class A, class T>
        XSIMD_INLINE batch<T, A> round(batch<T, A> const& self, requires_arch<common>) noexcept
        {
            auto v = abs(self);
            auto c = ceil(v);
            auto cp = select(c - 0.5 > v, c - 1, c);
            return select(v > constants::maxflint<batch<T, A>>(), self, copysign(cp, self));
        }

        // trunc
        template <class A, class T, class = std::enable_if_t<std::is_integral_v<T>>>
        XSIMD_INLINE batch<T, A> trunc(batch<T, A> const& self, requires_arch<common>) noexcept
        {
            return self;
        }
        namespace detail
        {
            // trunc must not raise FE_INEXACT (C17 7.12.9.8), so the fractional bits are
            // removed with exact-only operations: scaling by powers of two and a subtraction
            // of two adjacent values, never a rounding operation.
            template <class A, class T, class U, int SigBits, int ExpBias>
            XSIMD_INLINE batch<T, A> trunc_impl(batch<T, A> const& self) noexcept
            {
                using batch_type = batch<T, A>;
                using int_batch = batch<U, A>;
                const U exp_mask = (U(1) << (sizeof(U) * 8 - 1 - SigBits)) - 1;
                const U exp_max = exp_mask - 1; // largest exponent scaled below
                const U exp_min = U(ExpBias); // |x| < 1 truncates to zero
                const U sig_max = U(ExpBias + SigBits - 1); // |x| >= 2^SigBits is integral
                batch_type sign = bitofsign(self);
                int_batch bits = ::xsimd::bitwise_cast<U>(self ^ sign);
                int_batch e = (bits >> SigBits) & int_batch(exp_mask);
                // lanes outside [1, 2^SigBits) are forced to zero before the scaling so
                // that inf/nan/huge inputs cannot overflow or flag there
                auto big = e > int_batch(sig_max);
                auto small = e < int_batch(exp_min);
                int_batch e_safe = select(big, int_batch(exp_max), e);
                int_batch p = (int_batch(exp_max) - e_safe) << SigBits; // bits of 2^-e
                int_batch q = e_safe << SigBits; // bits of 2^e
                int_batch xb = select(big, int_batch(0), bits);
                batch_type x = ::xsimd::bitwise_cast<T>(xb);
                batch_type y = x * ::xsimd::bitwise_cast<T>(p); // in [1, 2), exact
                batch_type frac = (y - batch_type(1)) * ::xsimd::bitwise_cast<T>(q); // exact
                int_batch tb = ::xsimd::bitwise_cast<U>(x - frac); // exact
                tb = select(small, int_batch(0), tb);
                tb = select(big, bits, tb);
                return ::xsimd::bitwise_cast<T>(tb) ^ sign;
            }
        }
        template <class A>
        XSIMD_INLINE batch<float, A> trunc(batch<float, A> const& self, requires_arch<common>) noexcept
        {
            return detail::trunc_impl<A, float, uint32_t, 23, 127>(self);
        }
        template <class A>
        XSIMD_INLINE batch<double, A> trunc(batch<double, A> const& self, requires_arch<common>) noexcept
        {
            return detail::trunc_impl<A, double, uint64_t, 52, 1023>(self);
        }

    }

}

#endif
