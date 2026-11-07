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

#if defined(XSIMD_ENABLE_STD_SIMD)

#include "xsimd/xsimd.hpp"

#if (defined(XSIMD_WITH_STD_SIMD) || defined(XSIMD_WITH_STD_EXPERIMENTAL_SIMD)) && !defined(XSIMD_NO_SUPPORTED_ARCHITECTURE)

#include "test_utils.hpp"

#include <array>
#include <cstddef>
#include <type_traits>

// SFINAE probe: the std type of width N does not exist for every N and T in the
// TS fallback (fixed_size rejects some widths)
template <class T, std::size_t N, class = void>
struct std_simd_exists : std::false_type
{
};

template <class T, std::size_t N>
struct std_simd_exists<T, N, std::void_t<xsimd::detail::std_simd<T, N>>> : std::true_type
{
};

template <class B>
struct std_simd_test
{
    using batch_type = B;
    using value_type = typename B::value_type;
    using mask_type = xsimd::batch_bool<value_type>;
    static constexpr std::size_t size = B::size;

    using std_vec_type = xsimd::detail::std_simd<value_type, size>;
    using std_mask_type = xsimd::detail::std_simd_mask<value_type, size>;

    // distinct per-lane values so a lane swap or a dropped lane fails
    static value_type lane_value(std::size_t i) noexcept { return value_type(i + 1); }
    static bool lane_mask(std::size_t i) noexcept { return i % 2 == 0; }

    static void check_no_wide_conversion()
    {
        if constexpr (std_simd_exists<value_type, size + 1>::value)
        {
            static_assert(!std::is_convertible<batch_type, xsimd::detail::std_simd<value_type, size + 1>>::value,
                          "batch must not convert to a std_simd of width size+1");
        }
    }

    static_assert(std::is_convertible<batch_type, std_vec_type>::value,
                  "batch to std_simd must be an implicit conversion");
    static_assert(std::is_convertible<std_vec_type, batch_type>::value,
                  "std_simd to batch must be an implicit conversion");
    static_assert(std::is_convertible<mask_type, std_mask_type>::value,
                  "batch_bool to std_simd_mask must be an implicit conversion");
    static_assert(std::is_convertible<std_mask_type, mask_type>::value,
                  "std_simd_mask to batch_bool must be an implicit conversion");

    void test_batch_round_trip() const
    {
        check_no_wide_conversion();

        std::array<value_type, size> in {}, out {};
        for (std::size_t i = 0; i < size; ++i)
        {
            in[i] = lane_value(i);
        }

        // batch -> std_simd -> batch
        batch_type b = batch_type::load_unaligned(in.data());
        std_vec_type s = b;
        batch_type b2 = s;
        b2.store_unaligned(out.data());
        for (std::size_t i = 0; i < size; ++i)
        {
            CHECK_EQ(out[i], in[i]);
        }

        // std_simd -> batch -> std_simd
        std_vec_type s2 = std_vec_type([](std::size_t i) { return lane_value(i); });
        batch_type b3 = s2;
        std_vec_type s3 = b3;
        std_vec_type const& cs3 = s3;
        for (std::size_t i = 0; i < size; ++i)
        {
            value_type got = cs3[i];
            CHECK_EQ(got, in[i]);
        }
    }

    void test_batch_bool_round_trip() const
    {
        std::array<bool, size> in {}, out {};
        for (std::size_t i = 0; i < size; ++i)
        {
            in[i] = lane_mask(i);
        }

        // batch_bool -> std_simd_mask -> batch_bool
        mask_type m = mask_type::load_unaligned(in.data());
        std_mask_type sm = m;
        mask_type m2 = sm;
        m2.store_unaligned(out.data());
        for (std::size_t i = 0; i < size; ++i)
        {
            CHECK_EQ(out[i], in[i]);
        }

        // std_simd_mask -> batch_bool -> std_simd_mask
#if defined(XSIMD_WITH_STD_EXPERIMENTAL_SIMD)
        // shortcut: the lane comparison hits a libstdc++ static_assert on clang for
        // char and long lanes, load the source mask from the bool array instead
        std_mask_type sm2(in.data(), std::experimental::element_aligned);
#else
        // the source mask comes from a lane comparison so the std path is not hard-coded
        std_vec_type ones(value_type(1));
        std_vec_type shaped = std_vec_type([](std::size_t i) { return value_type(lane_mask(i)); });
        std_mask_type sm2 = (shaped == ones);
#endif
        mask_type m3 = sm2;
        std_mask_type sm3 = m3;
        std_mask_type const& csm3 = sm3;
        for (std::size_t i = 0; i < size; ++i)
        {
            bool got = csm3[i];
            CHECK_EQ(got, in[i]);
        }
    }
};

TEST_CASE_TEMPLATE("[std_simd batch interop]", B, BATCH_TYPES)
{
    // skip widths the std path has no type for
    if constexpr (std_simd_exists<typename B::value_type, B::size>::value)
    {
        std_simd_test<B> Test;
        SUBCASE("round trip") { Test.test_batch_round_trip(); }
    }
}

TEST_CASE_TEMPLATE("[std_simd batch_bool interop]", B, BATCH_TYPES)
{
    if constexpr (std_simd_exists<typename B::value_type, B::size>::value)
    {
        std_simd_test<B> Test;
        SUBCASE("round trip") { Test.test_batch_bool_round_trip(); }
    }
}

#endif

#endif // XSIMD_ENABLE_STD_SIMD
