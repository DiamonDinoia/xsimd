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

#include "xsimd/xsimd.hpp"

#if (defined(XSIMD_WITH_STD_SIMD) || defined(XSIMD_WITH_STD_EXPERIMENTAL_SIMD)) && !defined(XSIMD_NO_SUPPORTED_ARCHITECTURE)

#include "test_utils.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

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

    static value_type lane_value(std::size_t i) noexcept { return value_type(i + 1); }
    static bool lane_mask(std::size_t i) noexcept { return i % 2 == 0; }

    static void check_no_wide_conversion()
    {
        if constexpr (std_simd_exists<value_type, size + 1>::value)
        {
            static_assert(!std::is_convertible_v<batch_type, xsimd::detail::std_simd<value_type, size + 1>>,
                          "batch must not convert to a std_simd of width size+1");
        }
    }

    static_assert(std::is_convertible_v<batch_type, std_vec_type>,
                  "batch to std_simd must be an implicit conversion");
    static_assert(std::is_convertible_v<std_vec_type, batch_type>,
                  "std_simd to batch must be an implicit conversion");
    static_assert(std::is_convertible_v<mask_type, std_mask_type>,
                  "batch_bool to std_simd_mask must be an implicit conversion");
    static_assert(std::is_convertible_v<std_mask_type, mask_type>,
                  "std_simd_mask to batch_bool must be an implicit conversion");

    void test_batch_round_trip() const
    {
        check_no_wide_conversion();

        std::array<value_type, size> in {}, out {};
        for (std::size_t i = 0; i < size; ++i)
        {
            in[i] = lane_value(i);
        }

        batch_type b = batch_type::load_unaligned(in.data());
        std_vec_type s = b;
        batch_type b2 = s;
        b2.store_unaligned(out.data());
        for (std::size_t i = 0; i < size; ++i)
        {
            CHECK_EQ(out[i], in[i]);
        }

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

        mask_type m = mask_type::load_unaligned(in.data());
        std_mask_type sm = m;
        mask_type m2 = sm;
        m2.store_unaligned(out.data());
        for (std::size_t i = 0; i < size; ++i)
        {
            CHECK_EQ(out[i], in[i]);
        }

        std::uint64_t bits = 0;
        for (std::size_t i = 0; i < size; ++i)
        {
            bits |= static_cast<std::uint64_t>(in[i]) << i;
        }
        std_mask_type sm2 = xsimd::detail::std_mask_from_bits<std_mask_type, size>(bits);
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

#ifndef XSIMD_WITH_STD_SIMD
TEST_CASE("[std_simd fixed_size interop]")
{
    using batch_t = xsimd::batch<float>;
    using bool_t = typename batch_t::batch_bool_type;
    using std_vec_t = std::experimental::fixed_size_simd<float, batch_t::size>;
    using std_mask_t = typename std_vec_t::mask_type;
    static_assert(std_vec_t::size() == batch_t::size);
    static_assert(std_mask_t::size() == bool_t::size);

    static_assert(std::is_convertible_v<batch_t, std_vec_t>);
    static_assert(std::is_convertible_v<std_vec_t, batch_t>);
    static_assert(std::is_convertible_v<bool_t, std_mask_t>);
    static_assert(std::is_convertible_v<std_mask_t, bool_t>);

    batch_t const b(3.14f);
    std_vec_t const s = b;
    batch_t const b2 = s;
    for (std::size_t i = 0; i < batch_t::size; ++i)
    {
        CHECK_EQ(s[i], 3.14f);
        CHECK_EQ(b2.get(i), 3.14f);
    }

    bool_t const m(true);
    std_mask_t const k = m;
    bool_t const m2 = k;
    for (std::size_t i = 0; i < bool_t::size; ++i)
    {
        CHECK(k[i]);
        CHECK(m2.get(i));
    }
}
#endif
#if defined(XSIMD_WITH_EMULATED)
TEST_CASE("[std_simd emulated interop]")
{
    using batch_t = xsimd::batch<float, xsimd::emulated<320>>;
    using bool_t = typename batch_t::batch_bool_type;
    using std_vec_t = xsimd::detail::std_simd<float, batch_t::size>;
    using std_mask_t = xsimd::detail::std_simd_mask<float, batch_t::size>;
    static_assert(batch_t::size == 10);

    static_assert(std::is_convertible_v<batch_t, std_vec_t>);
    static_assert(std::is_convertible_v<std_vec_t, batch_t>);
    static_assert(std::is_convertible_v<bool_t, std_mask_t>);
    static_assert(std::is_convertible_v<std_mask_t, bool_t>);

    batch_t const b(3.14f);
    std_vec_t const s = b;
    batch_t const b2 = s;
    for (std::size_t i = 0; i < batch_t::size; ++i)
    {
        CHECK_EQ(s[i], 3.14f);
        CHECK_EQ(b2.get(i), 3.14f);
    }

    bool_t const m(true);
    std_mask_t const k = m;
    bool_t const m2 = k;
    for (std::size_t i = 0; i < bool_t::size; ++i)
    {
        CHECK(k[i]);
        CHECK(m2.get(i));
    }
}
#endif

#endif