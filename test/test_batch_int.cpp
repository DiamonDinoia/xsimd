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

#include "xsimd/xsimd.hpp"
#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE

#include "test_utils.hpp"

#include <climits>

namespace xsimd
{
    template <class T, std::size_t N = T::size>
    struct test_int_min_max
    {
        bool run()
        {
            return true;
        }
    };

    template <class T>
    struct test_int_min_max<batch<T>, 2>
    {
        void run()
        {
            using B = batch<T>;
            using BB = batch_bool<T>;
            using A = std::array<T, 2>;

            T max = std::numeric_limits<T>::max();
            T min = std::numeric_limits<T>::min();
            std::array<T, 2> maxmin_cmp { { max, min } };
            B maxmin = { max, min };
            INFO("numeric max and min");
            CHECK_BATCH_EQ(maxmin, maxmin_cmp);

            B a = { 1, 3 };
            B b(2);
            B c = { 2, 3 };

            auto r1 = xsimd::max(a, c);
            auto r3 = xsimd::min(a, c);

            INFO("max");
            CHECK_BATCH_EQ(r1, (A { { 2, 3 } }));
            INFO("min");
            CHECK_BATCH_EQ(r3, (A { { 1, 3 } }));

            auto r4 = a < b; // test lt
            BB e4 = { 1, 0 };
            CHECK_UNARY(xsimd::all(r4 == e4));
        }
    };

    template <class T>
    struct test_int_min_max<batch<T>, 4>
    {
        void run()
        {
            using B = batch<T>;
            using BB = batch_bool<T>;
            using A = std::array<T, 4>;

            B a = { 1, 3, 1, 1 };
            B b(2);
            B c = { 2, 3, 2, 3 };

            auto r1 = xsimd::max(a, c);
            auto r3 = xsimd::min(a, c);

            INFO("max");
            CHECK_BATCH_EQ(r1, (A { { 2, 3, 2, 3 } }));
            INFO("min");
            CHECK_BATCH_EQ(r3, (A { { 1, 3, 1, 1 } }));

            auto r4 = a < b; // test lt
            BB e4 = { 1, 0, 1, 1 };
            CHECK_UNARY(xsimd::all(r4 == e4));
        }
    };

    template <class T>
    struct test_int_min_max<batch<T>, 8>
    {
        void run()
        {
            using B = batch<T>;
            using BB = batch_bool<T>;
            using A = std::array<T, 8>;

            T max = std::numeric_limits<T>::max();
            T min = std::numeric_limits<T>::min();
            std::array<T, 8> maxmin_cmp { { 0, 0, max, 0, min, 0, 0, 0 } };
            B maxmin = { 0, 0, max, 0, min, 0, 0, 0 };
            INFO("numeric max and min");
            CHECK_BATCH_EQ(maxmin, maxmin_cmp);

            B a { 1, 3, 1, 3, 1, 1, 3, 3 };
            B b { 2 };
            B c { 2, 3, 2, 3, 2, 3, 2, 3 };

            auto r1 = xsimd::max(a, c);
            auto r3 = xsimd::min(a, c);
            auto r4 = a < b; // test lt
            INFO("max");
            CHECK_BATCH_EQ(r1, (A { { 2, 3, 2, 3, 2, 3, 3, 3 } }));
            INFO("min");
            CHECK_BATCH_EQ(r3, (A { { 1, 3, 1, 3, 1, 1, 2, 3 } }));

            BB e4 = { 1, 0, 1, 0, 1, 1, 0, 0 };
            CHECK_UNARY(xsimd::all(r4 == e4));
        }
    };

    template <class T>
    struct test_int_min_max<batch<T>, 16>
    {
        void run()
        {
            using B = batch<T>;
            using BB = batch_bool<T>;
            using A = std::array<T, 16>;

            T max = std::numeric_limits<T>::max();
            T min = std::numeric_limits<T>::min();
            std::array<T, 16> maxmin_cmp { { 0, 0, max, 0, min, 0, 0, 0, 0, 0, max, 0, min, 0, 0, 0 } };
            B maxmin = { 0, 0, max, 0, min, 0, 0, 0, 0, 0, max, 0, min, 0, 0, 0 };
            INFO("numeric max and min");
            CHECK_BATCH_EQ(maxmin, maxmin_cmp);

            B a = { 1, 3, 1, 3, 1, 3, 1, 3, 3, 3, 3, 3, min, max, max, min };
            B b(2);
            B c = { 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3 };
            auto r1 = xsimd::max(a, b);
            auto r3 = xsimd::min(a, b);
            auto r4 = a < b; // test lt
            auto r5 = a == c;
            auto r6 = a != c;

            INFO("max");
            CHECK_BATCH_EQ(r1, (A { { 2, 3, 2, 3, 2, 3, 2, 3, 3, 3, 3, 3, 2, max, max, 2 } }));
            INFO("min");
            CHECK_BATCH_EQ(r3, (A { { 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, min, 2, 2, min } }));

            BB e4 = { 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1 };
            CHECK_UNARY(xsimd::all(r4 == e4));

            BB e5 = { 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0 };
            CHECK_UNARY(xsimd::all(r5 == e5));
            CHECK_UNARY(xsimd::all(r6 == !e5));
        }
    };

    template <class T>
    struct test_int_min_max<batch<T>, 32>
    {
        void run()
        {
            using B = batch<T>;
            using BB = batch_bool<T>;
            using A = std::array<T, 32>;
            T max = std::numeric_limits<T>::max();
            T min = std::numeric_limits<T>::min();

            B a = { 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 3, 3, 3, 3, min, max, max, min };
            B b = 2;

            auto r1 = xsimd::max(a, b);
            auto r3 = xsimd::min(a, b);
            auto r4 = a < b; // test lt
            INFO("max");
            CHECK_BATCH_EQ(r1, (A { { 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 3, 3, 3, 3, 2, max, max, 2 } }));
            INFO("min");
            CHECK_BATCH_EQ(r3, (A { { 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, min, 2, 2, min } }));

            BB e4 = { 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1 };
            CHECK_UNARY(xsimd::all(r4 == e4));
        }
    };
}

template <class B>
struct batch_int_test
{
    using batch_type = B;
    using value_type = typename B::value_type;
    static constexpr size_t size = B::size;
    using array_type = std::array<value_type, size>;
    using bool_array_type = std::array<bool, size>;

    array_type lhs;
    array_type rhs;
    array_type shift;

    batch_int_test()
    {
        using signed_value_type = std::make_signed_t<value_type>;
        for (size_t i = 0; i < size; ++i)
        {
            bool negative_lhs = std::is_signed_v<value_type> && (i % 2 == 1);
            lhs[i] = value_type(i) * (negative_lhs ? -10 : 10);
            if (lhs[i] == value_type(0))
            {
                lhs[i] += value_type(1);
            }
            rhs[i] = value_type(i) + value_type(4);
            shift[i] = signed_value_type(i) % (CHAR_BIT * sizeof(value_type));
        }
    }

    void test_modulo() const
    {
        // batch % batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                           [](const value_type& l, const value_type& r)
                           { return l % r; });
            batch_type res = batch_lhs() % batch_rhs();
            INFO("batch % batch");
            CHECK_BATCH_EQ(res, expected);
        }
    }

    void test_shift() const
    {
        int32_t nb_sh = 3;
        // batch << scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(),
                           [nb_sh](const value_type& v)
                           { return xsimd::abs(v) << nb_sh; });
            batch_type res = abs(batch_lhs()) << nb_sh;
            INFO("batch << scalar");
            CHECK_BATCH_EQ(res, expected);
        }
        // batch << batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), shift.cbegin(), expected.begin(),
                           [](const value_type& l, const value_type& r)
                           { return xsimd::abs(l) << r; });
            batch_type res = abs(batch_lhs()) << batch_shift();
            INFO("batch << batch");
            CHECK_BATCH_EQ(res, expected);
        }
        // batch >> scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(),
                           [nb_sh](const value_type& v)
                           { return v >> nb_sh; });
            batch_type res = batch_lhs() >> nb_sh;
            INFO("batch >> scalar");
            CHECK_BATCH_EQ(res, expected);
        }
        // batch >> batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), shift.cbegin(), expected.begin(),
                           [](const value_type& l, const value_type& r)
                           { return l >> r; });
            batch_type res = batch_lhs() >> batch_shift();
            INFO("batch >> batch");
            CHECK_BATCH_EQ(res, expected);
        }
    }

    void test_more_shift() const
    {
        int32_t s = static_cast<int32_t>(sizeof(value_type) * 8);
        batch_type lhs = batch_type(value_type(1));
        batch_type res;

        for (int32_t i = 0; i < s; ++i)
        {
            res = lhs << i;
            batch_type expected(value_type(1) << i);
            CHECK_BATCH_EQ(res, expected);
        }
        lhs = batch_type(std::numeric_limits<value_type>::max());
        for (int32_t i = 0; i < s; ++i)
        {
            res = lhs >> i;
            batch_type expected(std::numeric_limits<value_type>::max() >> i);
            CHECK_BATCH_EQ(res, expected);
        }
    }

    void test_min_max() const
    {
        xsimd::test_int_min_max<batch_type> t;
        t.run();
    }

    // 0, ~0, single bits, prefixes, suffixes and pseudo-random words: the
    // patterns the bit ops actually branch on
    static array_type bit_patterns(size_t seed)
    {
        constexpr size_t bits = sizeof(value_type) * CHAR_BIT;
        using U = std::make_unsigned_t<value_type>;
        array_type a;
        for (size_t i = 0; i < size; ++i)
        {
            size_t k = seed * size + i;
            size_t sh = k % bits;
            U u;
            switch (k % 6)
            {
            case 0:
                u = U(0);
                break;
            case 1:
                u = U(~U(0));
                break;
            case 2:
                u = U(U(1) << sh);
                break;
            case 3:
                u = U(~U(0)) << sh;
                break;
            case 4:
                u = U(U(U(1) << sh) - U(1));
                break;
            default:
                u = U(k * 0x9E3779B9u + 0x7F4A7C15u);
                break;
            }
            a[i] = value_type(u);
        }
        return a;
    }

    void test_popcount() const
    {
        using U = std::make_unsigned_t<value_type>;
        for (size_t s = 0; s < 6; ++s)
        {
            array_type in = bit_patterns(s), expected;
            std::transform(in.cbegin(), in.cend(), expected.begin(), [](value_type v)
                           { return value_type(xsimd::detail::popcount(U(v))); });
            INFO("popcount, pattern " << s);
            CHECK_BATCH_EQ(xsimd::popcount(batch_type::load_unaligned(in.data())), expected);
        }
    }

    void test_countl_zero() const
    {
        using U = std::make_unsigned_t<value_type>;
        for (size_t s = 0; s < 6; ++s)
        {
            array_type in = bit_patterns(s), expected;
            std::transform(in.cbegin(), in.cend(), expected.begin(), [](value_type v)
                           { return value_type(xsimd::detail::countl_zero(U(v))); });
            INFO("countl_zero, pattern " << s);
            CHECK_BATCH_EQ(xsimd::countl_zero(batch_type::load_unaligned(in.data())), expected);
        }
    }

    void test_countr_zero() const
    {
        using U = std::make_unsigned_t<value_type>;
        for (size_t s = 0; s < 6; ++s)
        {
            array_type in = bit_patterns(s), expected;
            std::transform(in.cbegin(), in.cend(), expected.begin(), [](value_type v)
                           { return value_type(xsimd::detail::countr_zero(U(v))); });
            INFO("countr_zero, pattern " << s);
            CHECK_BATCH_EQ(xsimd::countr_zero(batch_type::load_unaligned(in.data())), expected);
        }
    }

    void test_bit_reverse() const
    {
        constexpr size_t bits = sizeof(value_type) * CHAR_BIT;
        using U = std::make_unsigned_t<value_type>;
        auto reverse = [](value_type v)
        {
            U u = U(v), r = 0;
            for (size_t i = 0; i < bits; ++i)
                if ((u >> i) & U(1))
                    r = U(r | U(U(1) << (bits - 1 - i)));
            return value_type(r);
        };
        for (size_t s = 0; s < 6; ++s)
        {
            array_type in = bit_patterns(s), expected;
            std::transform(in.cbegin(), in.cend(), expected.begin(), reverse);
            INFO("bit_reverse, pattern " << s);
            CHECK_BATCH_EQ(xsimd::bit_reverse(batch_type::load_unaligned(in.data())), expected);

            // an involution: reversing twice is the identity
            INFO("bit_reverse round trip, pattern " << s);
            CHECK_BATCH_EQ(xsimd::bit_reverse(xsimd::bit_reverse(batch_type::load_unaligned(in.data()))), in);
        }
    }

    void test_bit_deposit_extract() const
    {
        constexpr size_t bits = sizeof(value_type) * CHAR_BIT;
        using U = std::make_unsigned_t<value_type>;
        auto deposit = [](value_type v, value_type m)
        {
            U u = U(v), r = 0;
            size_t k = 0;
            for (size_t i = 0; i < bits; ++i)
                if ((U(m) >> i) & U(1))
                {
                    if ((u >> k) & U(1))
                        r = U(r | U(U(1) << i));
                    ++k;
                }
            return value_type(r);
        };
        auto extract = [](value_type v, value_type m)
        {
            U u = U(v), r = 0;
            size_t k = 0;
            for (size_t i = 0; i < bits; ++i)
                if ((U(m) >> i) & U(1))
                {
                    if ((u >> i) & U(1))
                        r = U(r | U(U(1) << k));
                    ++k;
                }
            return value_type(r);
        };

        for (size_t s = 0; s < 6; ++s)
        {
            array_type in = bit_patterns(s);
            array_type mask = bit_patterns(s + 3);
            array_type expected;
            batch_type bin = batch_type::load_unaligned(in.data());
            batch_type bmask = batch_type::load_unaligned(mask.data());

            std::transform(in.cbegin(), in.cend(), mask.cbegin(), expected.begin(), deposit);
            INFO("bit_deposit, pattern " << s);
            CHECK_BATCH_EQ(xsimd::bit_deposit(bin, bmask), expected);

            std::transform(in.cbegin(), in.cend(), mask.cbegin(), expected.begin(), extract);
            INFO("bit_extract, pattern " << s);
            CHECK_BATCH_EQ(xsimd::bit_extract(bin, bmask), expected);

            // extract undoes deposit for the bits the mask keeps
            std::transform(in.cbegin(), in.cend(), mask.cbegin(), expected.begin(),
                           [](value_type v, value_type m)
                           {
                               size_t kept = size_t(xsimd::detail::popcount(U(m)));
                               U keep = kept >= bits ? U(~U(0)) : U(U(U(1) << kept) - U(1));
                               return value_type(U(v) & keep);
                           });
            INFO("bit_extract(bit_deposit(x)), pattern " << s);
            CHECK_BATCH_EQ(xsimd::bit_extract(xsimd::bit_deposit(bin, bmask), bmask), expected);
        }
    }

    void test_less_than_underflow() const
    {
        batch_type test_negative_compare = batch_type(5) - 6;
        if (std::is_unsigned_v<value_type>)
        {
            CHECK_FALSE(xsimd::any(test_negative_compare < 1));
        }
        else
        {
            CHECK_UNARY(xsimd::all(test_negative_compare < 1));
        }
    }

private:
    batch_type batch_lhs() const
    {
        return batch_type::load_unaligned(lhs.data());
    }

    batch_type batch_rhs() const
    {
        return batch_type::load_unaligned(rhs.data());
    }

    batch_type batch_shift() const
    {
        return batch_type::load_unaligned(shift.data());
    }
};

TEST_CASE_TEMPLATE("[batch int tests]", B, BATCH_INT_TYPES)
{
    batch_int_test<B> Test;

    SUBCASE("modulo")
    {
        Test.test_modulo();
    }

    SUBCASE("shift")
    {
        Test.test_shift();
    }

    SUBCASE("more_shift")
    {
        Test.test_more_shift();
    }

    SUBCASE("min_max")
    {
        Test.test_min_max();
    }

    SUBCASE("less_than_underflow")
    {
        Test.test_less_than_underflow();
    }

    SUBCASE("popcount")
    {
        Test.test_popcount();
    }

    SUBCASE("countl_zero")
    {
        Test.test_countl_zero();
    }

    SUBCASE("countr_zero")
    {
        Test.test_countr_zero();
    }

    SUBCASE("bit_reverse")
    {
        Test.test_bit_reverse();
    }

    SUBCASE("bit_deposit_extract")
    {
        Test.test_bit_deposit_extract();
    }
}

TEST_CASE("[batch bit_permute]")
{
    using batch_type = xsimd::batch<uint8_t>;
    constexpr size_t size = batch_type::size;
    using array_type = std::array<uint8_t, size>;

    array_type in;
    for (size_t i = 0; i < size; ++i)
        in[i] = uint8_t(i * 7 + 1);
    batch_type b = batch_type::load_unaligned(in.data());

    auto reference = [&in](std::array<uint8_t, 8> const& src)
    {
        array_type out;
        for (size_t i = 0; i < size; ++i)
        {
            uint8_t r = 0;
            for (size_t k = 0; k < 8; ++k)
                if ((in[i] >> src[k]) & 1)
                    r = uint8_t(r | (1u << k));
            out[i] = r;
        }
        return out;
    };

    INFO("bit_permute reverse");
    CHECK_BATCH_EQ(xsimd::bit_permute(b, xsimd::bit_permute_constant<7, 6, 5, 4, 3, 2, 1, 0> {}),
                   reference({ { 7, 6, 5, 4, 3, 2, 1, 0 } }));

    INFO("bit_permute nibble swap");
    CHECK_BATCH_EQ(xsimd::bit_permute(b, xsimd::bit_permute_constant<4, 5, 6, 7, 0, 1, 2, 3> {}),
                   reference({ { 4, 5, 6, 7, 0, 1, 2, 3 } }));

    INFO("bit_permute identity");
    CHECK_BATCH_EQ(xsimd::bit_permute(b, xsimd::bit_permute_constant<0, 1, 2, 3, 4, 5, 6, 7> {}), in);

    // not a permutation: every destination reads the same source bit
    INFO("bit_permute broadcast");
    CHECK_BATCH_EQ(xsimd::bit_permute(b, xsimd::bit_permute_constant<0, 0, 0, 0, 0, 0, 0, 0> {}),
                   reference({ { 0, 0, 0, 0, 0, 0, 0, 0 } }));

    INFO("bit_permute matches bit_reverse");
    CHECK_BATCH_EQ(xsimd::bit_permute(b, xsimd::bit_permute_constant<7, 6, 5, 4, 3, 2, 1, 0> {}),
                   xsimd::bit_reverse(b));
}

TEST_CASE("[batch bit_matmul]")
{
    using byte_batch = xsimd::batch<uint8_t>;
    using word_batch = xsimd::batch<uint64_t>;
    constexpr size_t nbytes = byte_batch::size;
    constexpr size_t nwords = word_batch::size;

    // out bit k of a byte is the parity of that byte ANDed with matrix byte 7-k
    auto reference = [](uint8_t v, uint64_t m, uint8_t x)
    {
        uint8_t r = 0;
        for (int k = 0; k < 8; ++k)
        {
            uint8_t row = uint8_t(m >> (8 * (7 - k)));
            uint8_t t = uint8_t(row & v);
            int parity = 0;
            for (int b = 0; b < 8; ++b)
                parity ^= (t >> b) & 1;
            if (parity ^ ((x >> k) & 1))
                r = uint8_t(r | (1u << k));
        }
        return r;
    };

    std::array<uint8_t, nbytes> in;
    std::array<uint64_t, nwords> mat;
    for (size_t i = 0; i < nbytes; ++i)
        in[i] = uint8_t(i * 37 + 11);

    // identity, bit reversal, all-zero, all-ones and an arbitrary matrix
    const uint64_t identity = 0x0102040810204080ull;
    const uint64_t reversal = 0x8040201008040201ull;
    for (uint64_t m : { identity, reversal, uint64_t(0), ~uint64_t(0), uint64_t(0x0123456789ABCDEF) })
    {
        for (size_t i = 0; i < nwords; ++i)
            mat[i] = m;
        byte_batch b = byte_batch::load_unaligned(in.data());
        word_batch bm = word_batch::load_unaligned(mat.data());

        std::array<uint8_t, nbytes> expected;
        for (size_t i = 0; i < nbytes; ++i)
            expected[i] = reference(in[i], m, 0);
        INFO("bit_matmul, matrix " << m);
        CHECK_BATCH_EQ(xsimd::bit_matmul(b, bm), expected);

        for (size_t i = 0; i < nbytes; ++i)
            expected[i] = reference(in[i], m, 0xA5);
        INFO("bit_matmul with xor, matrix " << m);
        CHECK_BATCH_EQ(xsimd::bit_matmul<0xA5>(b, bm), expected);
    }

    // the identity matrix is a no-op and the reversal matrix agrees with
    // bit_reverse, which is implemented independently
    byte_batch b = byte_batch::load_unaligned(in.data());
    INFO("bit_matmul identity");
    CHECK_BATCH_EQ(xsimd::bit_matmul(b, word_batch(identity)), in);
    INFO("bit_matmul reversal matches bit_reverse");
    CHECK_BATCH_EQ(xsimd::bit_matmul(b, word_batch(reversal)), xsimd::bit_reverse(b));

    // a per-lane matrix must be applied per lane, not broadcast
    if (nwords >= 2)
    {
        for (size_t i = 0; i < nwords; ++i)
            mat[i] = (i % 2) ? reversal : identity;
        word_batch bm = word_batch::load_unaligned(mat.data());
        std::array<uint8_t, nbytes> expected;
        for (size_t i = 0; i < nbytes; ++i)
            expected[i] = reference(in[i], mat[i / 8], 0);
        INFO("bit_matmul with a different matrix per lane");
        CHECK_BATCH_EQ(xsimd::bit_matmul(b, bm), expected);
    }
}

TEST_CASE("[batch multishift]")
{
    using byte_batch = xsimd::batch<uint8_t>;
    using word_batch = xsimd::batch<uint64_t>;
    constexpr size_t nbytes = byte_batch::size;
    constexpr size_t nwords = word_batch::size;

    std::array<uint8_t, nbytes> ctrl;
    std::array<uint64_t, nwords> data;
    for (size_t i = 0; i < nwords; ++i)
        data[i] = 0x0123456789ABCDEFull * (i + 1) + i;

    // offset 0 and offset 64 must both mean "no rotation"
    for (int round = 0; round < 3; ++round)
    {
        for (size_t i = 0; i < nbytes; ++i)
            ctrl[i] = round == 0 ? uint8_t((i % 2) ? 0 : 64)
                                 : uint8_t(i * 11 + round * 5);

        std::array<uint8_t, nbytes> expected;
        for (size_t i = 0; i < nbytes; ++i)
        {
            uint64_t q = data[i / 8];
            int off = ctrl[i] & 63;
            uint64_t rot = off == 0 ? q : ((q >> off) | (q << (64 - off)));
            expected[i] = uint8_t(rot & 0xff);
        }

        INFO("multishift, round " << round);
        CHECK_BATCH_EQ(xsimd::multishift(byte_batch::load_unaligned(ctrl.data()),
                                         word_batch::load_unaligned(data.data())),
                       expected);
    }
}
#endif
