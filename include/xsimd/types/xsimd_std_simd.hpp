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

#ifndef XSIMD_STD_SIMD_HPP
#define XSIMD_STD_SIMD_HPP

#include <cstddef>
#include <type_traits>
#include <utility>

#ifdef XSIMD_ENABLE_STD_SIMD

#include <bitset>
#include <cstdint>

#include "xsimd/config/xsimd_macros.hpp"

#if __cplusplus > 202302L && __has_include(<simd>)
#include <simd>
#endif

#if !defined(__cpp_lib_simd) && !defined(__glibcxx_simd) && __has_include(<experimental/simd>)
#include <experimental/simd>
#endif

namespace xsimd
{
    namespace detail
    {
        // array-backed registers (emulated widths): the std array constructor is
        // not native register support, those widths keep the generic path
        template <class Reg, class = void>
        struct reg_is_array : std::false_type
        {
        };

        template <class Reg>
        struct reg_is_array<Reg, std::void_t<typename Reg::register_type::value_type, decltype(std::declval<typename Reg::register_type&>().size())>> : std::true_type
        {
        };

        // one path selection: the register path needs the std type constructible
        // from and convertible back to the register so both directions compile
        template <class S, class Reg>
        struct std_reg_path
            : std::bool_constant<!reg_is_array<Reg>::value && std::is_constructible_v<S, typename Reg::register_type> && std::is_constructible_v<typename Reg::register_type, S>>
        {
        };

        // matching: the std type with the batch's element type and width; the
        // specializations below tie the element type so a float mask never
        // matches a double batch_bool of the same width
        template <class S, class T, std::size_t N, class = void>
        struct is_std_simd : std::false_type
        {
        };

        template <class S, class T, std::size_t N, class = void>
        struct is_std_simd_mask : std::false_type
        {
        };

#if defined(__cpp_lib_simd) || defined(__glibcxx_simd)
#define XSIMD_WITH_STD_SIMD 1
        namespace stdx = std::simd;

        template <class T, std::size_t N>
        using std_simd = stdx::vec<T, N>;
        template <class T, std::size_t N>
        using std_simd_mask = stdx::mask<T, N>;

        template <class B>
        using std_mask_native = std_reg_path<std_simd<typename B::operand_type, B::size>, typename B::batch_type>;

        template <class S, class B, std::size_t... I>
        XSIMD_INLINE B std_mask_to_lanes_batch(S const& m, std::index_sequence<I...>) noexcept
        {
            return B(m[I]...);
        }

        template <class S, std::size_t N>
        XSIMD_INLINE S std_mask_from_bits(std::uint64_t bits) noexcept
        {
            return S(std::bitset<N>(bits));
        }

        template <class S, std::size_t N>
        XSIMD_INLINE std::uint64_t std_mask_to_bits(S const& m) noexcept
        {
            return m.to_bitset().to_ullong();
        }

        // shortcut: the bit view carries 64 lanes, a wider native register needs the generic path
        template <class S, class B>
        XSIMD_INLINE B std_mask_to_batch(S const& m) noexcept
        {
            if constexpr (std_mask_native<B>::value)
            {
                return B::from_mask(std_mask_to_bits<S, B::size>(m));
            }
            return std_mask_to_lanes_batch<S, B>(m, std::make_index_sequence<B::size> {});
        }

        // shortcut: the bit view carries 64 lanes, a wider batch needs the vector compare
        template <class S, class B>
        XSIMD_INLINE S std_mask_from_batch(B const& b) noexcept
        {
            if constexpr (B::size <= 64)
            {
                return std_mask_from_bits<S, B::size>(b.mask());
            }
            using V = typename B::batch_type;
            using T = typename B::operand_type;
            std_simd<T, B::size> const sel = select(b, V(T(1)), V(T(0)));
            return S(sel != std_simd<T, B::size>(T(0)));
        }

        template <class T, class Abi, std::size_t N>
        struct is_std_simd<stdx::basic_vec<T, Abi>, T, N>
            : std::bool_constant<N == stdx::basic_vec<T, Abi>::size()>
        {
        };

        template <std::size_t Bytes, class Abi, class T, std::size_t N>
        struct is_std_simd_mask<stdx::basic_mask<Bytes, Abi>, T, N>
            : std::bool_constant<Bytes == sizeof(T) && N == stdx::basic_mask<Bytes, Abi>::size()>
        {
        };
#elif defined(__cpp_lib_experimental_parallel_simd)
#define XSIMD_WITH_STD_EXPERIMENTAL_SIMD 1
        namespace stdx = std::experimental;

        template <class T, std::size_t N>
        using std_simd = stdx::simd<T, stdx::simd_abi::deduce_t<T, N>>;
        template <class T, std::size_t N>
        using std_simd_mask = stdx::simd_mask<T, stdx::simd_abi::deduce_t<T, N>>;

        template <class B>
        using std_mask_native = std_reg_path<std_simd<typename B::operand_type, B::size>, typename B::batch_type>;

        // shortcut: the TS mask has no public bitset interface; a mask holding its lanes in a register-sized word sets lanes in place, other masks fold lanes through a bool array
        template <class S, std::size_t N>
        XSIMD_INLINE S std_mask_from_bits(std::uint64_t bits) noexcept
        {
            if constexpr (sizeof(S) <= sizeof(std::uint64_t))
            {
                S mask(false);
                for (std::size_t i = 0; i < N; ++i)
                {
                    mask[i] = ((bits >> i) & 1) != 0;
                }
                return mask;
            }
            else
            {
                bool tmp[N];
                for (std::size_t i = 0; i < N; ++i)
                {
                    tmp[i] = ((bits >> i) & 1) != 0;
                }
                return S(tmp, stdx::element_aligned);
            }
        }

        // the mask's where-expression writes the 0/1 vector in place, so the
        // register path converts and compares vectors; other widths use the
        // bool lane load/store route
        template <class S, class B>
        XSIMD_INLINE B std_mask_to_batch(S const& m) noexcept
        {
            using V = typename B::batch_type;
            using T = typename B::operand_type;
            using W = stdx::simd<T, typename S::abi_type>;
            if constexpr (std_mask_native<B>::value)
            {
                W values(T(0));
                stdx::where(m, values) = T(1);
                V const batch = values;
                return batch != V(T(0));
            }
            else
            {
                bool tmp[B::size];
                for (std::size_t i = 0; i < B::size; ++i)
                {
                    tmp[i] = m[i];
                }
                return B::load_unaligned(tmp);
            }
        }

        // the TS lane compare trips a libstdc++ static_assert on clang for char and long lanes, so non-fixed_size masks take the bit pivot
        template <class S, class B>
        XSIMD_INLINE S std_mask_from_batch(B const& b) noexcept
        {
            if constexpr (B::size <= 64 && !std::is_same_v<typename S::abi_type, stdx::simd_abi::fixed_size<B::size>>)
            {
                return std_mask_from_bits<S, B::size>(b.mask());
            }
            else if constexpr (std_mask_native<B>::value)
            {
                using V = typename B::batch_type;
                using T = typename B::operand_type;
                using W = stdx::simd<T, typename S::abi_type>;
                W const sel = select(b, V(T(1)), V(T(0)));
                return S(sel != W(T(0)));
            }
            else
            {
                bool tmp[B::size];
                b.store_unaligned(tmp);
                return S(tmp, stdx::element_aligned);
            }
        }

        template <class T, class Abi, std::size_t N>
        struct is_std_simd<stdx::simd<T, Abi>, T, N>
            : std::bool_constant<N == stdx::simd<T, Abi>::size()>
        {
        };

        template <class T, class Abi, std::size_t N>
        struct is_std_simd_mask<stdx::simd_mask<T, Abi>, T, N>
            : std::bool_constant<N == stdx::simd_mask<T, Abi>::size()>
        {
        };
#else
#error "XSIMD_ENABLE_STD_SIMD: no std::simd or std::experimental::simd in this standard library"
#endif

    }
}

#endif // XSIMD_ENABLE_STD_SIMD

#endif
