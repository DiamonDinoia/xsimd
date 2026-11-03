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

#ifndef XSIMD_COMMON_CAST_HPP
#define XSIMD_COMMON_CAST_HPP

#include "../../config/xsimd_macros.hpp"
#include "../../utils/xsimd_type_traits.hpp"

#include <array>

namespace xsimd
{
    namespace kernel
    {
        template <class A, class T>
        XSIMD_INLINE std::array<batch<widen_t<T>, A>, 2> widen(batch<T, A> const& x, requires_arch<common>) noexcept
        {
            alignas(A::alignment()) T buffer[batch<T, A>::size];
            x.store_aligned(&buffer[0]);

            using T_out = widen_t<T>;
            alignas(A::alignment()) T_out out_buffer[batch<T, A>::size];
            for (size_t i = 0; i < batch<T, A>::size; ++i)
                out_buffer[i] = static_cast<T_out>(buffer[i]);

            return { batch<T_out, A>::load_aligned(&out_buffer[0]),
                     batch<T_out, A>::load_aligned(&out_buffer[batch<T_out, A>::size]) };
        }

        // narrow_cast: C++ cast semantics (truncation toward zero to an integer
        // type), same element count in a narrower register for the one-batch
        // form, twice as many half-width elements for the two-batch form.
        template <class A, class B_out, class T_in>
        XSIMD_INLINE B_out narrow_cast(batch<T_in, A> const& self, B_out const&, requires_arch<common>) noexcept
        {
            using T_out = typename B_out::value_type;
            alignas(A::alignment()) T_in buffer_in[batch<T_in, A>::size];
            alignas(B_out::alignment()) T_out buffer_out[B_out::size];
            self.store_aligned(&buffer_in[0]);
            for (size_t i = 0; i < B_out::size; ++i)
                buffer_out[i] = static_cast<T_out>(buffer_in[i]);
            return B_out::load_aligned(&buffer_out[0]);
        }

        template <class A, class T_out, class T_in>
        XSIMD_INLINE batch<T_out, A> narrow_cast(batch<T_in, A> const& lo, batch<T_in, A> const& hi, batch<T_out, A> const&, requires_arch<common>) noexcept
        {
            alignas(A::alignment()) T_in buffer_in[2 * batch<T_in, A>::size];
            alignas(batch<T_out, A>::alignment()) T_out buffer_out[batch<T_out, A>::size];
            lo.store_aligned(&buffer_in[0]);
            hi.store_aligned(&buffer_in[batch<T_in, A>::size]);
            for (size_t i = 0; i < batch<T_out, A>::size; ++i)
                buffer_out[i] = static_cast<T_out>(buffer_in[i]);
            return batch<T_out, A>::load_aligned(&buffer_out[0]);
        }
    }

}

#endif
