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

#ifndef XSIMD_COMMON_DETAILS_HPP
#define XSIMD_COMMON_DETAILS_HPP

#include "../../math/xsimd_rem_pio2.hpp"
#include "../../types/xsimd_common_arch.hpp"
#include "../../types/xsimd_utils.hpp"
#include "../xsimd_constants.hpp"

#include <cfenv>
#include <complex>

namespace xsimd
{
    // Forward declaration. Should we put them in a separate file?
    template <class T, class A>
    XSIMD_INLINE batch<T, A> abs(batch<T, A> const& self) noexcept;
    template <class T, class A>
    XSIMD_INLINE batch<T, A> abs(batch<std::complex<T>, A> const& self) noexcept;
    template <class T, class A>
    XSIMD_INLINE bool any(batch_bool<T, A> const& self) noexcept;
    template <class T, class A>
    XSIMD_INLINE batch<T, A> atan2(batch<T, A> const& self, batch<T, A> const& other) noexcept;
    template <class A, class T_out, class T_in>
    XSIMD_INLINE batch<T_out, A> batch_cast(batch<T_in, A> const&, batch<T_out, A> const& out) noexcept;
    template <class T, class A>
    XSIMD_INLINE batch<T, A> bitofsign(batch<T, A> const& self) noexcept;
    template <class T_out, class T_in, class A>
    XSIMD_INLINE batch<T_out, A> bitwise_cast(batch<T_in, A> const& self) noexcept;
    template <class T, class A>
    XSIMD_INLINE batch<T, A> cos(batch<T, A> const& self) noexcept;
    template <class T, class A>
    XSIMD_INLINE batch<T, A> cosh(batch<T, A> const& self) noexcept;
    template <class T, class A>
    XSIMD_INLINE batch<T, A> exp(batch<T, A> const& self) noexcept;
    template <class T, class A>
    XSIMD_INLINE batch<T, A> fma(batch<T, A> const& x, batch<T, A> const& y, batch<T, A> const& z) noexcept;
    template <class T, class A>
    XSIMD_INLINE batch<T, A> fms(batch<T, A> const& x, batch<T, A> const& y, batch<T, A> const& z) noexcept;
    template <class T, class A>
    XSIMD_INLINE batch<T, A> fmas(batch<T, A> const& x, batch<T, A> const& y, batch<T, A> const& z) noexcept;
    template <class T, class A>
    XSIMD_INLINE batch<T, A> frexp(const batch<T, A>& x, const batch<as_integer_t<T>, A>& e) noexcept;
    template <class T, class A, uint64_t... Coefs>
    XSIMD_INLINE batch<T, A> horner(const batch<T, A>& self) noexcept;
    template <class T, class A>
    XSIMD_INLINE batch<T, A> hypot(const batch<T, A>& self) noexcept;
    template <class T, class A>
    XSIMD_INLINE batch_bool<T, A> is_even(batch<T, A> const& self) noexcept;
    template <class T, class A>
    XSIMD_INLINE batch_bool<T, A> is_flint(batch<T, A> const& self) noexcept;
    template <class T, class A>
    XSIMD_INLINE batch_bool<T, A> is_odd(batch<T, A> const& self) noexcept;
    template <class T, class A>
    XSIMD_INLINE typename batch<T, A>::batch_bool_type isinf(batch<T, A> const& self) noexcept;
    template <class T, class A>
    XSIMD_INLINE typename batch<T, A>::batch_bool_type isfinite(batch<T, A> const& self) noexcept;
    template <class T, class A>
    XSIMD_INLINE typename batch<T, A>::batch_bool_type isnan(batch<T, A> const& self) noexcept;
    template <class T, class A>
    XSIMD_INLINE batch<T, A> ldexp(const batch<T, A>& x, const batch<as_integer_t<T>, A>& e) noexcept;
    template <class T, class A>
    XSIMD_INLINE batch<T, A> log(batch<T, A> const& self) noexcept;
    template <class T, class A>
    XSIMD_INLINE batch<T, A> nearbyint(batch<T, A> const& self) noexcept;
    template <class T, class A>
    XSIMD_INLINE batch<as_integer_t<T>, A> nearbyint_as_int(const batch<T, A>& x) noexcept;
    template <class T, class A>
    XSIMD_INLINE batch<T, A> rint(batch<T, A> const& self) noexcept;
    template <class T, class A>
    XSIMD_INLINE T reduce_add(batch<T, A> const&) noexcept;
    template <class T, class A>
    XSIMD_INLINE T reduce_mul(batch<T, A> const&) noexcept;
    template <class T, class A>
    XSIMD_INLINE batch<T, A> select(batch_bool<T, A> const&, batch<T, A> const&, batch<T, A> const&) noexcept;
    template <class T, class A>
    XSIMD_INLINE batch<std::complex<T>, A> select(batch_bool<T, A> const&, batch<std::complex<T>, A> const&, batch<std::complex<T>, A> const&) noexcept;
    template <class T, class A>
    XSIMD_INLINE batch<T, A> sign(batch<T, A> const& self) noexcept;
    template <class T, class A>
    XSIMD_INLINE batch<T, A> signnz(batch<T, A> const& self) noexcept;
    template <class T, class A>
    XSIMD_INLINE batch<T, A> sin(batch<T, A> const& self) noexcept;
    template <class T, class A>
    XSIMD_INLINE batch<T, A> sinh(batch<T, A> const& self) noexcept;
    template <class T, class A>
    XSIMD_INLINE std::pair<batch<T, A>, batch<T, A>> sincos(batch<T, A> const& self) noexcept;
    template <class T, class A>
    XSIMD_INLINE batch<T, A> sqrt(batch<T, A> const& self) noexcept;
    template <class T, class A, class Vt, Vt... Values>
    XSIMD_INLINE std::enable_if_t<std::is_arithmetic_v<T>, batch<T, A>>
    swizzle(batch<T, A> const& x, batch_constant<Vt, A, Values...> mask) noexcept;
    template <class T, class A>
    XSIMD_INLINE batch<T, A> tan(batch<T, A> const& self) noexcept;
    template <class T, class A>
    XSIMD_INLINE batch<as_float_t<T>, A> to_float(batch<T, A> const& self) noexcept;
    template <class T, class A>
    XSIMD_INLINE batch<as_integer_t<T>, A> to_int(batch<T, A> const& self) noexcept;
    template <class T, class A>
    XSIMD_INLINE batch<T, A> trunc(batch<T, A> const& self) noexcept;

    namespace kernel
    {

        namespace detail
        {
            // Prevent -ffast-math from reassociating floating-point
            // arithmetic across this point.  The reason string
            // documents *why* at each call site; unused at runtime.
            //
            // Zero-cost register constraints per target:
            //   x86  "+x"  (XMM/YMM/ZMM, also scalar float/double)
            //   ARM  "+w"  (V-reg / SVE Z-reg, also scalar float/double)
            //   PPC  "+wa" (VS register, also scalar float/double)
            //   RISC-V "+f" (F/D register, scalar float/double)
            //   RISC-V RVV "+vr" (V register; GCC 15+ / Clang 20+)
            //
            // On unknown targets the "+m" fallback spills; it is
            // only emitted when the compiler can actually reassociate.
            template <class T>
            XSIMD_INLINE void reassociation_barrier(T& x, const char*) noexcept
            {
#if XSIMD_REASSOCIATIVE_MATH && XSIMD_WITH_INLINE_ASM && !defined(__EMSCRIPTEN__)
#if XSIMD_WITH_SSE2
                __asm__ volatile("" : "+x"(x));
#elif XSIMD_WITH_NEON || XSIMD_WITH_SVE
                __asm__ volatile("" : "+w"(x));
#elif XSIMD_WITH_VSX
                __asm__ volatile("" : "+wa"(x));
#else
                __asm__ volatile("" : "+m"(x));
#endif
#else
                (void)x;
#endif
            }

            // RISC-V scalar float/double: use F/D registers instead of
            // spilling through "+m".  These overloads also serve
            // emulated batches on RISC-V via the std::array overload.
#if XSIMD_REASSOCIATIVE_MATH && XSIMD_WITH_INLINE_ASM && defined(__riscv)
            XSIMD_INLINE void reassociation_barrier(float& x, const char*) noexcept
            {
                __asm__ volatile("" : "+f"(x));
            }
            XSIMD_INLINE void reassociation_barrier(double& x, const char*) noexcept
            {
                __asm__ volatile("" : "+f"(x));
            }
#endif

            template <class T, size_t N>
            XSIMD_INLINE void reassociation_barrier(std::array<T, N>& arr, const char* reason) noexcept
            {
                for (auto& v : arr)
                    reassociation_barrier(v, reason);
            }

            template <class T, class A>
            XSIMD_INLINE void reassociation_barrier(batch<T, A>& b, const char* reason) noexcept
            {
#if XSIMD_REASSOCIATIVE_MATH && XSIMD_WITH_RVV && XSIMD_WITH_INLINE_ASM && ((__GNUC__ >= 15) || (__clang_major__ >= 20))
                __asm__ volatile("" : "+vr"(b.data.value.value));
                (void)reason;
#else
                reassociation_barrier(b.data, reason);
#endif
            }

#if defined(FE_INEXACT) && XSIMD_WITH_INLINE_ASM && !defined(__EMSCRIPTEN__)
            // Pin the rounding between the two fenv reads; without this the compiler may
            // move the arithmetic past std::fetestexcept and misattribute its flag.
            template <class T>
            XSIMD_INLINE void fenv_fence(T& v) noexcept
            {
                __asm__ volatile("" : "+m"(v) : : "memory");
            }
#else
            template <class T>
            XSIMD_INLINE void fenv_fence(T&) noexcept
            {
            }
#endif

            // trunc raises no exception (C17 7.12.9.8). The magic add rounds |self|
            // to an integer whatever the rounding mode is and one decrement turns
            // that into floor; the add can raise FE_INEXACT, so drop only that flag
            // and only when it was clear before.
            template <class A, class T>
            XSIMD_INLINE batch<T, A> truncf_impl(batch<T, A> const& self) noexcept
            {
                using batch_type = batch<T, A>;
                batch_type sign = bitofsign(self);
                batch_type v = self ^ sign; // |self|
                // twotonmb = 2^SigBits: |self| >= it is integral, and the add lands
                // on the unit grid in [2^SigBits, 2^(SigBits+1))
                batch_type t2n = constants::twotonmb<batch_type>();
                auto big = !(v < t2n); // >= 2^SigBits or nan, already integral
                batch_type vs = select(big, batch_type(0), v);
#if defined(FE_INEXACT)
                const int raised = std::fetestexcept(FE_INEXACT);
#endif
                batch_type d0 = vs + t2n;
                reassociation_barrier(d0, "prevent collapsing (v + 2^n) - 2^n back to v");
                batch_type d = d0 - t2n;
                d = d - select(d > vs, batch_type(1), batch_type(0)); // floor(|self|), exact
                batch_type r = select(big, self, d ^ sign);
#if defined(FE_INEXACT)
                fenv_fence(r);
                if (!(raised & FE_INEXACT) && std::fetestexcept(FE_INEXACT))
                {
                    std::feclearexcept(FE_INEXACT);
                }
#endif
                return r;
            }

            template <class F, class A, class T, class... Batches>
            XSIMD_INLINE batch<T, A> apply(F&& func, batch<T, A> const& self, batch<T, A> const& other) noexcept
            {
                constexpr std::size_t size = batch<T, A>::size;
                alignas(A::alignment()) T self_buffer[size];
                alignas(A::alignment()) T other_buffer[size];
                self.store_aligned(&self_buffer[0]);
                other.store_aligned(&other_buffer[0]);
                for (std::size_t i = 0; i < size; ++i)
                {
                    self_buffer[i] = func(self_buffer[i], other_buffer[i]);
                }
                return batch<T, A>::load_aligned(self_buffer);
            }

            template <class U, class F, class A, class T>
            XSIMD_INLINE batch<U, A> apply_transform(F&& func, batch<T, A> const& self) noexcept
            {
                static_assert(batch<T, A>::size == batch<U, A>::size,
                              "Source and destination sizes must match");
                constexpr std::size_t src_size = batch<T, A>::size;
                constexpr std::size_t dest_size = batch<U, A>::size;
                alignas(A::alignment()) T self_buffer[src_size];
                alignas(A::alignment()) U other_buffer[dest_size];
                self.store_aligned(&self_buffer[0]);
                for (std::size_t i = 0; i < src_size; ++i)
                {
                    other_buffer[i] = func(self_buffer[i]);
                }
                return batch<U, A>::load_aligned(other_buffer);
            }
        }

        // some common fast_cast conversion
        namespace detail
        {
            template <class A>
            XSIMD_INLINE batch<uint8_t, A> fast_cast(batch<int8_t, A> const& self, batch<uint8_t, A> const&, requires_arch<common>) noexcept
            {
                return bitwise_cast<uint8_t>(self);
            }
            template <class A>
            XSIMD_INLINE batch<uint16_t, A> fast_cast(batch<int16_t, A> const& self, batch<uint16_t, A> const&, requires_arch<common>) noexcept
            {
                return bitwise_cast<uint16_t>(self);
            }
            template <class A>
            XSIMD_INLINE batch<uint32_t, A> fast_cast(batch<int32_t, A> const& self, batch<uint32_t, A> const&, requires_arch<common>) noexcept
            {
                return bitwise_cast<uint32_t>(self);
            }
            template <class A>
            XSIMD_INLINE batch<uint64_t, A> fast_cast(batch<int64_t, A> const& self, batch<uint64_t, A> const&, requires_arch<common>) noexcept
            {
                return bitwise_cast<uint64_t>(self);
            }
            template <class A>
            XSIMD_INLINE batch<int8_t, A> fast_cast(batch<uint8_t, A> const& self, batch<int8_t, A> const&, requires_arch<common>) noexcept
            {
                return bitwise_cast<int8_t>(self);
            }
            template <class A>
            XSIMD_INLINE batch<int16_t, A> fast_cast(batch<uint16_t, A> const& self, batch<int16_t, A> const&, requires_arch<common>) noexcept
            {
                return bitwise_cast<int16_t>(self);
            }
            template <class A>
            XSIMD_INLINE batch<int32_t, A> fast_cast(batch<uint32_t, A> const& self, batch<int32_t, A> const&, requires_arch<common>) noexcept
            {
                return bitwise_cast<int32_t>(self);
            }
            template <class A>
            XSIMD_INLINE batch<int64_t, A> fast_cast(batch<uint64_t, A> const& self, batch<int64_t, A> const&, requires_arch<common>) noexcept
            {
                return bitwise_cast<int64_t>(self);
            }

            // double -> int64 without hardware cvttpd_epi64. Must not call trunc(): the common trunc goes
            // through to_int and recurses back here on any arch without its own trunc. |x| >= 2^52 is already
            // integral; below that (|x| + 2^52) - 2^52 rounds |x| to the nearest integer exactly and a -1 where
            // the rounding went up gives floor(|x|) = |trunc(x)|. Each half stays |half| < 2^32, so the mantissa
            // of 1.5 * 2^52 + half holds it exactly and one integer subtraction reads it out. Exact for every x
            // whose truncation fits int64_t.
            template <class A>
            XSIMD_INLINE batch<double, A> trunc_magnitude_below_2_52(batch<double, A> const& x) noexcept
            {
                const batch<double, A> two52(4503599627370496.0);
                const batch<double, A> ax = abs(x);
                const batch<double, A> r = (ax + two52) - two52;
                return copysign(select(r > ax, r - batch<double, A>(1.0), r), x);
            }
            template <class A>
            XSIMD_INLINE batch<int64_t, A> fast_cast(batch<double, A> const& x, batch<int64_t, A> const&, requires_arch<common>) noexcept
            {
                using batch_type = batch<double, A>;
                batch_type magic(0x1.8p52);
                batch<int64_t, A> magic_i = bitwise_cast<int64_t>(magic);
                batch_type two52(0x1.0p52);
                batch_type t = select(abs(x) < two52, trunc_magnitude_below_2_52(x), x);
                batch_type hi = trunc_magnitude_below_2_52(t * batch_type(0x1p-32)); // |hi| <= 2^31
                batch_type lo = t - hi * batch_type(0x1p32); // exact, |lo| < 2^32
                detail::reassociation_barrier(lo, "keep lo below 2^32 before the magic add");
                batch<int64_t, A> hi_i = bitwise_cast<int64_t>(hi + magic) - magic_i;
                batch<int64_t, A> lo_i = bitwise_cast<int64_t>(lo + magic) - magic_i;
                return (hi_i << 32) + lo_i;
            }

            // Provide a common uint32_t -> float cast only if we have a
            // non-common int32_t -> float fast_cast
            template <class A, class = decltype(fast_cast(std::declval<batch<int32_t, A> const&>(), std::declval<batch<float, A> const&>(), A {}))>
            XSIMD_INLINE batch<float, A> fast_cast(batch<uint32_t, A> const& v, batch<float, A> const&, requires_arch<common>) noexcept
            {
                // see https://stackoverflow.com/questions/34066228/how-to-perform-uint32-float-conversion-with-sse
                batch<uint32_t, A> msk_lo(0xFFFF);
                batch<float, A> cnst65536f(65536.0f);

                auto v_lo = batch_cast<int32_t>(v & msk_lo); /* extract the 16 lowest significant bits of self                             */
                auto v_hi = batch_cast<int32_t>(v >> 16); /* 16 most significant bits of v                                                 */
                auto v_lo_flt = batch_cast<float>(v_lo); /* No rounding                                                                */
                auto v_hi_flt = batch_cast<float>(v_hi); /* No rounding                                                                */
                v_hi_flt = cnst65536f * v_hi_flt; /* No rounding                                                            */
                return v_hi_flt + v_lo_flt; /* Rounding may occur here, mul and add may fuse to fma for haswell and newer   */
            }

            // Provide a common float -> uint32_t cast only if we have a
            // non-common float -> int32_t fast_cast
            template <class A, class = decltype(fast_cast(std::declval<batch<float, A> const&>(), std::declval<batch<int32_t, A> const&>(), A {}))>
            XSIMD_INLINE batch<uint32_t, A> fast_cast(batch<float, A> const& v, batch<uint32_t, A> const&, requires_arch<common>) noexcept
            {
                auto is_large = v >= batch<float, A>(1u << 31);
                auto small_v = bitwise_cast<float>(batch_cast<int32_t>(v));
                auto large_v = bitwise_cast<float>(
                    batch_cast<int32_t>(v - batch<float, A>(1u << 31))
                    ^ batch<int32_t, A>(1u << 31));
                return bitwise_cast<uint32_t>(select(is_large, large_v, small_v));
            }
        }

        namespace detail
        {
            // Generic conversion handling machinery. Each architecture must define
            // conversion function when such conversions exits in the form of
            // intrinsic. Then we use that information to automatically decide whether
            // to use scalar or vector conversion when doing load / store / batch_cast
            struct with_fast_conversion
            {
            };
            struct with_slow_conversion
            {
            };

            template <class A, class From, class To, class = void>
            struct conversion_type_impl
            {
                using type = with_slow_conversion;
            };

            template <class A, class From, class To>
            struct conversion_type_impl<A, From, To,
                                        std::void_t<decltype(fast_cast(std::declval<const batch<From, A>&>(),
                                                                       std::declval<const batch<To, A>&>(),
                                                                       std::declval<const A&>()))>>
            {
                using type = with_fast_conversion;
            };

            template <class A, class From, class To>
            using conversion_type = typename conversion_type_impl<A, From, To>::type;
        }

        namespace detail
        {
            /* origin: boost/simdfunction/horn.hpp*/
            /*
             * ====================================================
             * copyright 2016 NumScale SAS
             *
             * Distributed under the Boost Software License, Version 1.0.
             * (See copy at http://boost.org/LICENSE_1_0.txt)
             * ====================================================
             */
            template <class B, uint64_t c>
            XSIMD_INLINE B coef() noexcept
            {
                using value_type = typename B::value_type;
                return B(bit_cast<value_type>(as_unsigned_integer_t<value_type>(c)));
            }
            template <class B>
            XSIMD_INLINE B horner(const B&) noexcept
            {
                return B(typename B::value_type(0.));
            }

            template <class B, uint64_t c0>
            XSIMD_INLINE B horner(const B&) noexcept
            {
                return coef<B, c0>();
            }

            template <class B, uint64_t c0, uint64_t c1, uint64_t... args>
            XSIMD_INLINE B horner(const B& self) noexcept
            {
                return fma(self, horner<B, c1, args...>(self), coef<B, c0>());
            }

            /* origin: boost/simdfunction/horn1.hpp*/
            /*
             * ====================================================
             * copyright 2016 NumScale SAS
             *
             * Distributed under the Boost Software License, Version 1.0.
             * (See copy at http://boost.org/LICENSE_1_0.txt)
             * ====================================================
             */
            template <class B>
            XSIMD_INLINE B horner1(const B&) noexcept
            {
                return B(1.);
            }

            template <class B, uint64_t c0>
            XSIMD_INLINE B horner1(const B& x) noexcept
            {
                return x + detail::coef<B, c0>();
            }

            template <class B, uint64_t c0, uint64_t c1, uint64_t... args>
            XSIMD_INLINE B horner1(const B& x) noexcept
            {
                return fma(x, horner1<B, c1, args...>(x), detail::coef<B, c0>());
            }
        }

    }

}

#endif
