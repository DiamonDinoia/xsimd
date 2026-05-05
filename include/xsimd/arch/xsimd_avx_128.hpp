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

#ifndef XSIMD_AVX_128_HPP
#define XSIMD_AVX_128_HPP

#include <type_traits>

#include "../types/xsimd_avx_register.hpp"
#include "../types/xsimd_batch_constant.hpp"

namespace xsimd
{
    namespace kernel
    {
        using namespace types;

        // broadcast
        template <class A, class T, class = std::enable_if_t<std::is_same<T, float>::value>>
        XSIMD_INLINE batch<T, A> broadcast(T val, requires_arch<avx_128>) noexcept
        {
            return _mm_broadcast_ss(&val);
        }

        // eq
        template <class A>
        XSIMD_INLINE batch_bool<float, A> eq(batch<float, A> const& self, batch<float, A> const& other, requires_arch<avx_128>) noexcept
        {
            return _mm_cmp_ps(self, other, _CMP_EQ_OQ);
        }
        template <class A>
        XSIMD_INLINE batch_bool<double, A> eq(batch<double, A> const& self, batch<double, A> const& other, requires_arch<avx_128>) noexcept
        {
            return _mm_cmp_pd(self, other, _CMP_EQ_OQ);
        }

        // gt
        template <class A>
        XSIMD_INLINE batch_bool<float, A> gt(batch<float, A> const& self, batch<float, A> const& other, requires_arch<avx_128>) noexcept
        {
            return _mm_cmp_ps(self, other, _CMP_GT_OQ);
        }
        template <class A>
        XSIMD_INLINE batch_bool<double, A> gt(batch<double, A> const& self, batch<double, A> const& other, requires_arch<avx_128>) noexcept
        {
            return _mm_cmp_pd(self, other, _CMP_GT_OQ);
        }

        // ge
        template <class A>
        XSIMD_INLINE batch_bool<float, A> ge(batch<float, A> const& self, batch<float, A> const& other, requires_arch<avx_128>) noexcept
        {
            return _mm_cmp_ps(self, other, _CMP_GE_OQ);
        }
        template <class A>
        XSIMD_INLINE batch_bool<double, A> ge(batch<double, A> const& self, batch<double, A> const& other, requires_arch<avx_128>) noexcept
        {
            return _mm_cmp_pd(self, other, _CMP_GE_OQ);
        }

        // lt
        template <class A>
        XSIMD_INLINE batch_bool<float, A> lt(batch<float, A> const& self, batch<float, A> const& other, requires_arch<avx_128>) noexcept
        {
            return _mm_cmp_ps(self, other, _CMP_LT_OQ);
        }
        template <class A>
        XSIMD_INLINE batch_bool<double, A> lt(batch<double, A> const& self, batch<double, A> const& other, requires_arch<avx_128>) noexcept
        {
            return _mm_cmp_pd(self, other, _CMP_LT_OQ);
        }

        // le
        template <class A>
        XSIMD_INLINE batch_bool<float, A> le(batch<float, A> const& self, batch<float, A> const& other, requires_arch<avx_128>) noexcept
        {
            return _mm_cmp_ps(self, other, _CMP_LE_OQ);
        }
        template <class A>
        XSIMD_INLINE batch_bool<double, A> le(batch<double, A> const& self, batch<double, A> const& other, requires_arch<avx_128>) noexcept
        {
            return _mm_cmp_pd(self, other, _CMP_LE_OQ);
        }

        // neq
        template <class A>
        XSIMD_INLINE batch_bool<float, A> neq(batch<float, A> const& self, batch<float, A> const& other, requires_arch<avx_128>) noexcept
        {
            return _mm_cmp_ps(self, other, _CMP_NEQ_UQ);
        }
        template <class A>
        XSIMD_INLINE batch_bool<double, A> neq(batch<double, A> const& self, batch<double, A> const& other, requires_arch<avx_128>) noexcept
        {
            return _mm_cmp_pd(self, other, _CMP_NEQ_UQ);
        }

        // load_masked
        template <class A, bool... Values, class Mode>
        XSIMD_INLINE batch<float, A> load_masked(float const* mem, batch_bool_constant<float, A, Values...> mask, convert<float>, Mode, requires_arch<avx_128>) noexcept
        {
            return _mm_maskload_ps(mem, mask.as_batch());
        }
        template <class A, bool... Values, class Mode>
        XSIMD_INLINE batch<double, A> load_masked(double const* mem, batch_bool_constant<double, A, Values...> mask, convert<double>, Mode, requires_arch<avx_128>) noexcept
        {
            return _mm_maskload_pd(mem, mask.as_batch());
        }

        // Runtime-mask load for float/double on AVX-128. Both aligned_mode and
        // unaligned_mode map to _mm_maskload_* — the intrinsic does not fault
        // on masked-off lanes, so partial loads across page boundaries are safe.
        template <class A, class Mode>
        XSIMD_INLINE batch<float, A>
        load_masked(float const* mem, batch_bool<float, A> mask, convert<float>, Mode, requires_arch<avx_128>) noexcept
        {
            return _mm_maskload_ps(mem, _mm_castps_si128(mask));
        }
        template <class A, class Mode>
        XSIMD_INLINE batch<double, A>
        load_masked(double const* mem, batch_bool<double, A> mask, convert<double>, Mode, requires_arch<avx_128>) noexcept
        {
            return _mm_maskload_pd(mem, _mm_castpd_si128(mask));
        }

        // store_masked
        template <class A, bool... Values, class Mode>
        XSIMD_INLINE void store_masked(float* mem, batch<float, A> const& src, batch_bool_constant<float, A, Values...> mask, Mode, requires_arch<avx_128>) noexcept
        {
            return _mm_maskstore_ps(mem, mask.as_batch(), src);
        }

        template <class A, bool... Values, class Mode>
        XSIMD_INLINE void store_masked(double* mem, batch<double, A> const& src, batch_bool_constant<double, A, Values...> mask, Mode, requires_arch<avx_128>) noexcept
        {
            return _mm_maskstore_pd(mem, mask.as_batch(), src);
        }

        // Runtime-mask store for float/double on AVX-128. Same fault-suppression
        // semantics as the masked loads above; alignment mode is irrelevant.
        template <class A, class Mode>
        XSIMD_INLINE void
        store_masked(float* mem, batch<float, A> const& src, batch_bool<float, A> mask, Mode, requires_arch<avx_128>) noexcept
        {
            _mm_maskstore_ps(mem, _mm_castps_si128(mask), src);
        }
        template <class A, class Mode>
        XSIMD_INLINE void
        store_masked(double* mem, batch<double, A> const& src, batch_bool<double, A> mask, Mode, requires_arch<avx_128>) noexcept
        {
            _mm_maskstore_pd(mem, _mm_castpd_si128(mask), src);
        }

        // 128-bit head/tail. SSE2/4 have no general maskmov; only AVX added
        // _mm_maskload_* for FP. Integer T is routed through the FP maskload
        // via a float-bitcast pointer (same rationale as 256-bit avx.hpp).
        namespace detail
        {
            XSIMD_INLINE __m128 headtail_lanemask128_head_ps(std::size_t n) noexcept
            {
                alignas(16) static constexpr float iota[4] = { 0.f, 1.f, 2.f, 3.f };
                return _mm_cmplt_ps(_mm_load_ps(iota), _mm_set1_ps(static_cast<float>(n)));
            }
            XSIMD_INLINE __m128 headtail_lanemask128_tail_ps(std::size_t n_skip) noexcept
            {
                alignas(16) static constexpr float iota[4] = { 0.f, 1.f, 2.f, 3.f };
                return _mm_cmpge_ps(_mm_load_ps(iota), _mm_set1_ps(static_cast<float>(n_skip)));
            }
            XSIMD_INLINE __m128d headtail_lanemask128_head_pd(std::size_t n) noexcept
            {
                alignas(16) static constexpr double iota[2] = { 0., 1. };
                return _mm_cmplt_pd(_mm_load_pd(iota), _mm_set1_pd(static_cast<double>(n)));
            }
            XSIMD_INLINE __m128d headtail_lanemask128_tail_pd(std::size_t n_skip) noexcept
            {
                alignas(16) static constexpr double iota[2] = { 0., 1. };
                return _mm_cmpge_pd(_mm_load_pd(iota), _mm_set1_pd(static_cast<double>(n_skip)));
            }
        }

        template <class A, class Mode>
        XSIMD_INLINE batch<float, A>
        load_head(float const* mem, std::size_t n, Mode, requires_arch<avx_128>) noexcept
        {
            return _mm_maskload_ps(mem, _mm_castps_si128(detail::headtail_lanemask128_head_ps(n)));
        }
        template <class A, class Mode>
        XSIMD_INLINE batch<double, A>
        load_head(double const* mem, std::size_t n, Mode, requires_arch<avx_128>) noexcept
        {
            return _mm_maskload_pd(mem, _mm_castpd_si128(detail::headtail_lanemask128_head_pd(n)));
        }
        template <class A, class Mode>
        XSIMD_INLINE batch<float, A>
        load_tail(float const* mem, std::size_t n, Mode, requires_arch<avx_128>) noexcept
        {
            constexpr std::size_t size = batch<float, A>::size;
            return _mm_maskload_ps(detail::offset_back(mem, size - n),
                                   _mm_castps_si128(detail::headtail_lanemask128_tail_ps(size - n)));
        }
        template <class A, class Mode>
        XSIMD_INLINE batch<double, A>
        load_tail(double const* mem, std::size_t n, Mode, requires_arch<avx_128>) noexcept
        {
            constexpr std::size_t size = batch<double, A>::size;
            return _mm_maskload_pd(detail::offset_back(mem, size - n),
                                   _mm_castpd_si128(detail::headtail_lanemask128_tail_pd(size - n)));
        }
        template <class A, class Mode>
        XSIMD_INLINE void
        store_head(float* mem, std::size_t n, batch<float, A> const& src, Mode, requires_arch<avx_128>) noexcept
        {
            _mm_maskstore_ps(mem, _mm_castps_si128(detail::headtail_lanemask128_head_ps(n)), src);
        }
        template <class A, class Mode>
        XSIMD_INLINE void
        store_head(double* mem, std::size_t n, batch<double, A> const& src, Mode, requires_arch<avx_128>) noexcept
        {
            _mm_maskstore_pd(mem, _mm_castpd_si128(detail::headtail_lanemask128_head_pd(n)), src);
        }
        template <class A, class Mode>
        XSIMD_INLINE void
        store_tail(float* mem, std::size_t n, batch<float, A> const& src, Mode, requires_arch<avx_128>) noexcept
        {
            constexpr std::size_t size = batch<float, A>::size;
            _mm_maskstore_ps(detail::offset_back(mem, size - n),
                             _mm_castps_si128(detail::headtail_lanemask128_tail_ps(size - n)), src);
        }
        template <class A, class Mode>
        XSIMD_INLINE void
        store_tail(double* mem, std::size_t n, batch<double, A> const& src, Mode, requires_arch<avx_128>) noexcept
        {
            constexpr std::size_t size = batch<double, A>::size;
            _mm_maskstore_pd(detail::offset_back(mem, size - n),
                             _mm_castpd_si128(detail::headtail_lanemask128_tail_pd(size - n)), src);
        }

        template <class A, class T, class Mode,
                  class = std::enable_if_t<std::is_integral<T>::value && (sizeof(T) == 4 || sizeof(T) == 8)>>
        XSIMD_INLINE batch<T, A>
        load_head(T const* mem, std::size_t n, Mode, requires_arch<avx_128>) noexcept
        {
            XSIMD_IF_CONSTEXPR(sizeof(T) == 4)
            {
                const auto m = _mm_castps_si128(detail::headtail_lanemask128_head_ps(n));
                return _mm_castps_si128(_mm_maskload_ps(reinterpret_cast<float const*>(mem), m));
            }
            else
            {
                const auto m = _mm_castpd_si128(detail::headtail_lanemask128_head_pd(n));
                return _mm_castpd_si128(_mm_maskload_pd(reinterpret_cast<double const*>(mem), m));
            }
        }

        template <class A, class T, class Mode,
                  class = std::enable_if_t<std::is_integral<T>::value && (sizeof(T) == 4 || sizeof(T) == 8)>>
        XSIMD_INLINE batch<T, A>
        load_tail(T const* mem, std::size_t n, Mode, requires_arch<avx_128>) noexcept
        {
            constexpr std::size_t size = batch<T, A>::size;
            const auto base = detail::offset_back(mem, size - n);
            XSIMD_IF_CONSTEXPR(sizeof(T) == 4)
            {
                const auto m = _mm_castps_si128(detail::headtail_lanemask128_tail_ps(size - n));
                return _mm_castps_si128(_mm_maskload_ps(reinterpret_cast<float const*>(base), m));
            }
            else
            {
                const auto m = _mm_castpd_si128(detail::headtail_lanemask128_tail_pd(size - n));
                return _mm_castpd_si128(_mm_maskload_pd(reinterpret_cast<double const*>(base), m));
            }
        }

        template <class A, class T, class Mode,
                  class = std::enable_if_t<std::is_integral<T>::value && (sizeof(T) == 4 || sizeof(T) == 8)>>
        XSIMD_INLINE void
        store_head(T* mem, std::size_t n, batch<T, A> const& src, Mode, requires_arch<avx_128>) noexcept
        {
            XSIMD_IF_CONSTEXPR(sizeof(T) == 4)
            {
                const auto m = _mm_castps_si128(detail::headtail_lanemask128_head_ps(n));
                _mm_maskstore_ps(reinterpret_cast<float*>(mem), m, _mm_castsi128_ps(src));
            }
            else
            {
                const auto m = _mm_castpd_si128(detail::headtail_lanemask128_head_pd(n));
                _mm_maskstore_pd(reinterpret_cast<double*>(mem), m, _mm_castsi128_pd(src));
            }
        }

        template <class A, class T, class Mode,
                  class = std::enable_if_t<std::is_integral<T>::value && (sizeof(T) == 4 || sizeof(T) == 8)>>
        XSIMD_INLINE void
        store_tail(T* mem, std::size_t n, batch<T, A> const& src, Mode, requires_arch<avx_128>) noexcept
        {
            constexpr std::size_t size = batch<T, A>::size;
            const auto base = detail::offset_back(mem, size - n);
            XSIMD_IF_CONSTEXPR(sizeof(T) == 4)
            {
                const auto m = _mm_castps_si128(detail::headtail_lanemask128_tail_ps(size - n));
                _mm_maskstore_ps(reinterpret_cast<float*>(base), m, _mm_castsi128_ps(src));
            }
            else
            {
                const auto m = _mm_castpd_si128(detail::headtail_lanemask128_tail_pd(size - n));
                _mm_maskstore_pd(reinterpret_cast<double*>(base), m, _mm_castsi128_pd(src));
            }
        }

        // swizzle (dynamic mask)
        template <class A, class T, class ITy, class = std::enable_if_t<std::is_floating_point<T>::value && sizeof(T) == sizeof(ITy)>>
        XSIMD_INLINE batch<T, A> swizzle(batch<T, A> const& self, batch<ITy, A> mask, requires_arch<avx_128>) noexcept
        {
            XSIMD_IF_CONSTEXPR(std::is_same<T, float>::value)
            {
                return _mm_permutevar_ps(self, mask);
            }
            else
            {
                // VPERMILPD's variable control reads bit 1 of each 64-bit selector
                // (bit 0 is ignored), so a {0,1} index needs to become {0,2}.
                // Negation is a cheap alternative to a left shift by 1.
                return _mm_permutevar_pd(self, -mask);
            }
        }

        // swizzle (constant mask)
        template <class A, uint32_t V0, uint32_t V1, uint32_t V2, uint32_t V3>
        XSIMD_INLINE batch<float, A> swizzle(batch<float, A> const& self, batch_constant<uint32_t, A, V0, V1, V2, V3>, requires_arch<avx_128>) noexcept
        {
            return _mm_permute_ps(self, detail::mod_shuffle(V0, V1, V2, V3));
        }

        template <class A, uint32_t V0, uint32_t V1>
        XSIMD_INLINE batch<double, A> swizzle(batch<double, A> const& self, batch_constant<uint64_t, A, V0, V1>, requires_arch<avx_128>) noexcept
        {
            return _mm_permute_pd(self, detail::mod_shuffle(V0, V1));
        }

    }
}

#endif
