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

#ifndef XSIMD_COMMON_FWD_HPP
#define XSIMD_COMMON_FWD_HPP

#include "../types/xsimd_batch_constant.hpp"

#include <cstddef>
#include <type_traits>

namespace xsimd
{
    namespace kernel
    {
        // forward declaration
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        XSIMD_INLINE batch<T, A> abs(batch<T, A> const& self, requires_arch<common>) noexcept;
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        XSIMD_INLINE batch<T, A> bitwise_lshift(batch<T, A> const& self, batch<T, A> const& other, requires_arch<common>) noexcept;
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        XSIMD_INLINE batch<T, A> bitwise_rshift(batch<T, A> const& self, batch<T, A> const& other, requires_arch<common>) noexcept;
        template <class A, class T>
        XSIMD_INLINE batch_bool<T, A> gt(batch<T, A> const& self, batch<T, A> const& other, requires_arch<common>) noexcept;
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        XSIMD_INLINE batch<T, A> mul(batch<T, A> const& self, batch<T, A> const& other, requires_arch<common>) noexcept;
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        XSIMD_INLINE batch<T, A> sadd(batch<T, A> const& self, batch<T, A> const& other, requires_arch<common>) noexcept;
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        XSIMD_INLINE batch<T, A> ssub(batch<T, A> const& self, batch<T, A> const& other, requires_arch<common>) noexcept;
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        XSIMD_INLINE T hadd(batch<T, A> const& self, requires_arch<common>) noexcept;

        template <class A, class T_in, class T_out>
        XSIMD_INLINE batch<T_out, A> masked_load(T_in const* mem, typename batch<T_out, A>::batch_bool_type const& mask, convert<T_out>, requires_arch<common>) noexcept;
        template <class A, class T_in, class T_out>
        XSIMD_INLINE void masked_store(T_out* mem, batch<T_in, A> const& src, typename batch<T_in, A>::batch_bool_type const& mask, requires_arch<common>) noexcept;
        // compile-time masked_load / masked_store
        template <class A, class T_in, class T_out, bool... Values>
        XSIMD_INLINE batch<T_out, A> masked_load(T_in const* mem, batch_bool_constant<T_out, A, Values...> mask, convert<T_out>, requires_arch<common>) noexcept;
        template <class A, class T_in, class T_out, bool... Values>
        XSIMD_INLINE void masked_store(T_out* mem, batch<T_in, A> const& src, batch_bool_constant<T_in, A, Values...> mask, requires_arch<common>) noexcept;

        // Forward declarations for pack-level helpers
        namespace detail
        {
            template <typename T, T... Vs>
            XSIMD_INLINE constexpr bool is_identity() noexcept;
            template <typename T, class A, T... Vs>
            XSIMD_INLINE constexpr bool is_identity(batch_constant<T, A, Vs...>) noexcept;
            template <typename T, class A, T... Vs>
            XSIMD_INLINE constexpr bool is_all_different(batch_constant<T, A, Vs...>) noexcept;
            template <typename T, class A, T... Vs>
            XSIMD_INLINE constexpr bool is_dup_lo(batch_constant<T, A, Vs...>) noexcept;
            template <typename T, class A, T... Vs>
            XSIMD_INLINE constexpr bool is_dup_hi(batch_constant<T, A, Vs...>) noexcept;
            template <typename T, class A, T... Vs>
            XSIMD_INLINE constexpr bool is_cross_lane(batch_constant<T, A, Vs...>) noexcept;
            template <typename T, class A, T... Vs>
            XSIMD_INLINE constexpr bool no_duplicates(batch_constant<T, A, Vs...>) noexcept;

        }
    }
}

#endif
