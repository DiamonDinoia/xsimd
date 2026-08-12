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
            template <class F, std::size_t... Is>
            XSIMD_INLINE void static_for(F&& f, std::index_sequence<Is...>) noexcept
            {
                (f(std::integral_constant<std::size_t, Is> {}), ...);
            }

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

            constexpr std::size_t ilog2(std::size_t n) noexcept
            {
                std::size_t r = 0;
                while (n > 1)
                {
                    n >>= 1;
                    ++r;
                }
                return r;
            }

            // y[i] = parity of y[0..i], computed in log2(bits) steps
            template <class B>
            XSIMD_INLINE B parallel_suffix(B y) noexcept
            {
                constexpr std::size_t bits = sizeof(typename B::value_type) * CHAR_BIT;
                y = y ^ (y << 1);
                y = y ^ (y << 2);
                y = y ^ (y << 4);
                if constexpr (bits >= 16)
                    y = y ^ (y << 8);
                if constexpr (bits >= 32)
                    y = y ^ (y << 16);
                if constexpr (bits >= 64)
                    y = y ^ (y << 32);
                return y;
            }

            // accumulate the destination bits that all move by D positions:
            // one mask and one shift covers the whole group
            template <int D, class Cst, class A>
            XSIMD_INLINE batch<uint8_t, A> bit_permute_group(batch<uint8_t, A> const& acc,
                                                             batch<uint8_t, A> const& x, Cst) noexcept
            {
                using b_type = batch<uint8_t, A>;
                constexpr uint8_t m = Cst::source_mask(D);
                if constexpr (m == 0)
                    return acc;
                else if constexpr (D == 0)
                    return acc | (x & b_type(m));
                else if constexpr (D > 0)
                    return acc | ((x & b_type(m)) << D);
                else
                    return acc | ((x & b_type(m)) >> (-D));
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

        // countl_zero
        template <class A, class T, class /*=std::enable_if_t<std::is_integral_v<T>>*/>
        XSIMD_INLINE batch<T, A> countl_zero(batch<T, A> const& self, requires_arch<common>) noexcept
        {
            using U = as_unsigned_integer_t<T>;
            using b_type = batch<U, A>;
            constexpr std::size_t bits = sizeof(T) * CHAR_BIT;

            // smear the highest set bit down, then count what is left
            b_type x = bitwise_cast<U>(self);
            x = x | (x >> 1);
            x = x | (x >> 2);
            x = x | (x >> 4);
            if constexpr (bits >= 16)
                x = x | (x >> 8);
            if constexpr (bits >= 32)
                x = x | (x >> 16);
            if constexpr (bits >= 64)
                x = x | (x >> 32);
            return batch<T, A>(T(bits)) - popcount(bitwise_cast<T>(x), A {});
        }

        // countr_zero
        template <class A, class T, class /*=std::enable_if_t<std::is_integral_v<T>>*/>
        XSIMD_INLINE batch<T, A> countr_zero(batch<T, A> const& self, requires_arch<common>) noexcept
        {
            using U = as_unsigned_integer_t<T>;
            using b_type = batch<U, A>;

            // x & -x isolates the lowest set bit; for x == 0 the decrement
            // wraps to all ones, which popcounts to the element width
            b_type x = bitwise_cast<U>(self);
            b_type lowest = x & (b_type(U(0)) - x);
            return popcount(bitwise_cast<T>(b_type(lowest - b_type(U(1)))), A {});
        }

        // bit_permute
        template <class A, class T, uint8_t... Vs>
        XSIMD_INLINE batch<T, A> bit_permute(batch<T, A> const& self, bit_permute_constant<Vs...>,
                                             requires_arch<common>) noexcept
        {
            using cst_type = bit_permute_constant<Vs...>;
            auto x = bitwise_cast<uint8_t>(self);
            batch<uint8_t, A> res(uint8_t(0));
            // destination bit k reads source bit Vs[k], so a group shifts by
            // k - Vs[k], which spans [-7, 7]
            detail::static_for([&](auto i)
                               { res = detail::bit_permute_group<int(decltype(i)::value) - 7>(res, x, cst_type {}); },
                               std::make_index_sequence<15> {});
            return bitwise_cast<T>(res);
        }

        // bit_reverse
        template <class A, class T, class /*=std::enable_if_t<std::is_integral_v<T>>*/>
        XSIMD_INLINE batch<T, A> bit_reverse(batch<T, A> const& self, requires_arch<common>) noexcept
        {
            using U = as_unsigned_integer_t<T>;
            using b_type = batch<U, A>;
            constexpr std::size_t bits = sizeof(T) * CHAR_BIT;

            // swap adjacent groups of s bits, doubling s until the element is
            // mirrored. Expressing this as bit_permute<7..0> instead would put
            // every bit in a displacement group of its own, which costs far
            // more without a single-instruction permute to lower onto.
            b_type x = bitwise_cast<U>(self);
            b_type const m1(detail::alternating_mask<U>(1));
            x = ((x >> 1) & m1) | ((x & m1) << 1);
            b_type const m2(detail::alternating_mask<U>(2));
            x = ((x >> 2) & m2) | ((x & m2) << 2);
            b_type const m4(detail::alternating_mask<U>(4));
            x = ((x >> 4) & m4) | ((x & m4) << 4);
            if constexpr (bits >= 16)
            {
                b_type const m(detail::alternating_mask<U>(8));
                x = ((x >> 8) & m) | ((x & m) << 8);
            }
            if constexpr (bits >= 32)
            {
                b_type const m(detail::alternating_mask<U>(16));
                x = ((x >> 16) & m) | ((x & m) << 16);
            }
            if constexpr (bits >= 64)
            {
                b_type const m(detail::alternating_mask<U>(32));
                x = ((x >> 32) & m) | ((x & m) << 32);
            }
            return bitwise_cast<T>(x);
        }

        // bit_extract
        template <class A, class T, class /*=std::enable_if_t<std::is_integral_v<T>>*/>
        XSIMD_INLINE batch<T, A> bit_extract(batch<T, A> const& self, batch<T, A> const& mask,
                                             requires_arch<common>) noexcept
        {
            using U = as_unsigned_integer_t<T>;
            using b_type = batch<U, A>;
            constexpr std::size_t steps = detail::ilog2(sizeof(T) * CHAR_BIT);

            // Hacker's Delight, fig. 7-4: gather the selected bits right in
            // log2(bits) compress-by-2^i rounds
            b_type x = bitwise_cast<U>(self) & bitwise_cast<U>(mask);
            b_type m = bitwise_cast<U>(mask);
            b_type mk = (~m) << 1;
            detail::static_for([&](auto i)
                               {
                                   constexpr int sh = 1 << decltype(i)::value;
                                   b_type mp = detail::parallel_suffix(mk);
                                   b_type mv = mp & m;
                                   m = (m ^ mv) | (mv >> sh);
                                   b_type t = x & mv;
                                   x = (x ^ t) | (t >> sh);
                                   mk = mk & ~mp; },
                               std::make_index_sequence<steps> {});
            return bitwise_cast<T>(x);
        }

        // bit_deposit
        template <class A, class T, class /*=std::enable_if_t<std::is_integral_v<T>>*/>
        XSIMD_INLINE batch<T, A> bit_deposit(batch<T, A> const& self, batch<T, A> const& mask,
                                             requires_arch<common>) noexcept
        {
            using U = as_unsigned_integer_t<T>;
            using b_type = batch<U, A>;
            constexpr std::size_t steps = detail::ilog2(sizeof(T) * CHAR_BIT);

            // Hacker's Delight, fig. 7-10: the same rounds as bit_extract
            // record how far each group travels, then replay them backwards
            b_type x = bitwise_cast<U>(self);
            b_type m0 = bitwise_cast<U>(mask);
            b_type m = m0;
            b_type mk = (~m) << 1;
            b_type moved[steps];
            detail::static_for([&](auto i)
                               {
                                   constexpr std::size_t I = decltype(i)::value;
                                   b_type mp = detail::parallel_suffix(mk);
                                   b_type mv = mp & m;
                                   moved[I] = mv;
                                   m = (m ^ mv) | (mv >> (1 << I));
                                   mk = mk & ~mp; },
                               std::make_index_sequence<steps> {});
            detail::static_for([&](auto i)
                               {
                                   constexpr std::size_t I = steps - 1 - decltype(i)::value;
                                   b_type mv = moved[I];
                                   b_type t = x << (1 << I);
                                   x = (x & ~mv) | (t & mv); },
                               std::make_index_sequence<steps> {});
            return bitwise_cast<T>(x & m0);
        }

        // multishift
        template <class A>
        XSIMD_INLINE batch<uint8_t, A> multishift(batch<uint8_t, A> const& ctrl,
                                                  batch<uint64_t, A> const& data,
                                                  requires_arch<common>) noexcept
        {
            using b_type = batch<uint64_t, A>;
            b_type c = bitwise_cast<uint64_t>(ctrl);
            b_type res(uint64_t(0));
            // byte k of every qword takes 8 bits of that qword starting at
            // its own offset, wrapping around the 64-bit boundary
            detail::static_for([&](auto k)
                               {
                                   constexpr int K = int(decltype(k)::value);
                                   b_type off = (c >> (8 * K)) & b_type(uint64_t(63));
                                   b_type rotated = (data >> off) | (data << ((b_type(uint64_t(64)) - off) & b_type(uint64_t(63))));
                                   res = res | ((rotated & b_type(uint64_t(0xff))) << (8 * K)); },
                               std::make_index_sequence<8> {});
            return bitwise_cast<uint8_t>(res);
        }
    }
}

#endif
