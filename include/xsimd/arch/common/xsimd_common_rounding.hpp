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
            // ceil must not raise (C17 7.12.9.1): the compiler may speculate the +1
            // onto lanes where it rounds (2^digits + 1), so clear any inexactness the
            // formula does not semantically raise, like truncf_impl does for its adds.
#if defined(FE_INEXACT)
            const int raised = std::fetestexcept(FE_INEXACT);
#endif
            batch<T, A> truncated_self = trunc(self);
            auto need = truncated_self < self;
            batch<T, A> adjusted = select(need, truncated_self, batch<T, A>(0));
            batch<T, A> r = select(need, adjusted + 1, truncated_self);
#if defined(FE_INEXACT)
            detail::fenv_fence(r);
            if (!(raised & FE_INEXACT) && std::fetestexcept(FE_INEXACT))
            {
                std::feclearexcept(FE_INEXACT);
            }
#endif
            return r;
        }

        // floor
        template <class A, class T>
        XSIMD_INLINE batch<T, A> floor(batch<T, A> const& self, requires_arch<common>) noexcept
        {
            // floor must not raise; same dance and reason as ceil
#if defined(FE_INEXACT)
            const int raised = std::fetestexcept(FE_INEXACT);
#endif
            batch<T, A> truncated_self = trunc(self);
            auto need = truncated_self > self;
            batch<T, A> adjusted = select(need, truncated_self, batch<T, A>(0));
            batch<T, A> r = select(need, adjusted - 1, truncated_self);
#if defined(FE_INEXACT)
            detail::fenv_fence(r);
            if (!(raised & FE_INEXACT) && std::fetestexcept(FE_INEXACT))
            {
                std::feclearexcept(FE_INEXACT);
            }
#endif
            return r;
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
