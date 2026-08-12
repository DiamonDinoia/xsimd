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

#ifndef XSIMD_AVX512CD_HPP
#define XSIMD_AVX512CD_HPP

#include "../types/xsimd_avx512cd_register.hpp"

#include <type_traits>

namespace xsimd
{

    namespace kernel
    {
        using namespace types;

        // countl_zero
        template <class A, class T, class = std::enable_if_t<std::is_integral_v<T>>>
        XSIMD_INLINE batch<T, A> countl_zero(batch<T, A> const& self, requires_arch<avx512cd>) noexcept
        {
            if constexpr (sizeof(T) == 4)
                return _mm512_lzcnt_epi32(self);
            else if constexpr (sizeof(T) == 8)
                return _mm512_lzcnt_epi64(self);
            else
                return countl_zero(self, avx512f {});
        }
    }

}

#endif
