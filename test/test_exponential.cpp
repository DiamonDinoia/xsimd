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

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <utility>
#include <vector>

template <class B>
struct exponential_test
{
    using batch_type = B;
    using value_type = typename B::value_type;
    static constexpr size_t size = B::size;
    using vector_type = std::vector<value_type>;

    size_t nb_input;
    vector_type exp_input;
    vector_type log_input;
    vector_type expected;

    exponential_test()
    {
        nb_input = size * 10000;
        exp_input.resize(nb_input);
        log_input.resize(nb_input);
        for (size_t i = 0; i < nb_input; ++i)
        {
            exp_input[i] = value_type(-1.5) + i * value_type(3) / nb_input;
            log_input[i] = value_type(0.001 + i * 100 / nb_input);
        }
        expected.resize(nb_input);
    }

    void test_exponential_functions()
    {
        // exp
        {
            std::transform(exp_input.cbegin(), exp_input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::exp(v); });
            for (size_t i = 0; i < nb_input; i += size)
            {
                batch_type in, out, ref;
                detail::load_batch(in, exp_input, i);
                out = exp(in);
                detail::load_batch(ref, expected, i);
                INFO("exp");
                CHECK_BATCH_EQ(ref, out);
            }
        }

        // exp2
        {
            std::transform(exp_input.cbegin(), exp_input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::exp2(v); });
            for (size_t i = 0; i < nb_input; i += size)
            {
                batch_type in, out, ref;
                detail::load_batch(in, exp_input, i);
                out = exp2(in);
                detail::load_batch(ref, expected, i);
                INFO("exp2");
                CHECK_BATCH_EQ(ref, out);
            }
        }

        // exp10
        {
            std::transform(exp_input.cbegin(), exp_input.cend(), expected.begin(),
                           /* imprecise but enough for testing version of exp10 */
                           [](const value_type& v)
                           { return exp(log(10) * v); });
            for (size_t i = 0; i < nb_input; i += size)
            {
                batch_type in, out, ref;
                detail::load_batch(in, exp_input, i);
                out = exp10(in);
                detail::load_batch(ref, expected, i);
                INFO("exp10");
                CHECK_BATCH_EQ(ref, out);
            }
        }

        // expm1
        {
            std::transform(exp_input.cbegin(), exp_input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::expm1(v); });
            for (size_t i = 0; i < nb_input; i += size)
            {
                batch_type in, out, ref;
                detail::load_batch(in, exp_input, i);
                out = expm1(in);
                detail::load_batch(ref, expected, i);
                INFO("expm1");
                CHECK_BATCH_EQ(ref, out);
            }
        }
    }

    void test_log_functions()
    {
        // log
        {
            std::transform(log_input.cbegin(), log_input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::log(v); });
            for (size_t i = 0; i < nb_input; i += size)
            {
                batch_type in, out, ref;
                detail::load_batch(in, log_input, i);
                out = log(in);
                detail::load_batch(ref, expected, i);
                INFO("log");
                CHECK_BATCH_EQ(ref, out);
            }
        }

        // log2
        {
            std::transform(log_input.cbegin(), log_input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::log2(v); });
            for (size_t i = 0; i < nb_input; i += size)
            {
                batch_type in, out, ref;
                detail::load_batch(in, log_input, i);
                out = log2(in);
                detail::load_batch(ref, expected, i);
                INFO("log2");
                CHECK_BATCH_EQ(ref, out);
            }
        }

        // log10
        {
            std::transform(log_input.cbegin(), log_input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::log10(v); });
            for (size_t i = 0; i < nb_input; i += size)
            {
                batch_type in, out, ref;
                detail::load_batch(in, log_input, i);
                out = log10(in);
                detail::load_batch(ref, expected, i);
                INFO("log10");
                CHECK_BATCH_EQ(ref, out);
            }
        }

        // log1p
        {
            std::transform(log_input.cbegin(), log_input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::log1p(v); });
            for (size_t i = 0; i < nb_input; i += size)
            {
                batch_type in, out, ref;
                detail::load_batch(in, log_input, i);
                out = log1p(in);
                detail::load_batch(ref, expected, i);
                CHECK_BATCH_EQ(ref, out);
            }
        }
    }
    static std::int64_t bit_of(value_type v)
    {
        if constexpr (sizeof(value_type) == 4)
        {
            std::int32_t b;
            std::memcpy(&b, &v, 4);
            return b;
        }
        else
        {
            std::int64_t b;
            std::memcpy(&b, &v, 8);
            return b;
        }
    }

    // +inf as integer bits: a float +inf value is undefined under __FAST_MATH__, and the
    // clang-cl /fp:fast CI job runs this file
    static std::int64_t inf_bits()
    {
        return sizeof(value_type) == 4 ? 0x7f800000LL : 0x7ff0000000000000LL;
    }

    template <class F>
    void check_overflow_rows(const char* name, const std::vector<std::pair<value_type, std::int64_t>>& rows, std::int64_t tol, F simd_fn)
    {
        // wants are the correctly rounded results as bit patterns; rows near the overflow bound
        // catch +inf while the result is finite
        constexpr std::int64_t sign_mask = sizeof(value_type) == 4 ? 0x7fffffffLL : 0x7fffffffffffffffLL;
        for (const auto& r : rows)
        {
            batch_type in = batch_type(r.first);
            alignas(64) value_type got[size];
            simd_fn(in).store_aligned(got);
            for (size_t lane = 0; lane < size; ++lane)
            {
                const std::int64_t g = bit_of(got[lane]);
                bool ok;
                if (r.second == 0)
                {
                    ok = (g & sign_mask) <= tol;
                }
                else if (r.second == inf_bits())
                {
                    ok = g == inf_bits();
                }
                else
                {
                    ok = std::llabs(g - r.second) <= tol;
                }
                if (!ok)
                {
                    std::fprintf(stderr, "%s: x=%a lane %zu got=%a want_bits=0x%llx\n", name, double(r.first), lane, double(got[lane]), (unsigned long long)r.second);
                }
                CHECK(ok);
            }
        }
    }

    void test_exp_overflow()
    {
        // regression for DiamonDinoia/xsimd#7: exp/exp2/exp10 must go to +inf only past the true
        // overflow bound and must stay within tol ulp of the correctly rounded value at the top
        std::vector<std::pair<value_type, std::int64_t>> exp_rows, exp2_rows, exp10_rows;
        if constexpr (std::is_same<value_type, float>::value)
        {
            exp_rows = { { 88.5f, bit_of(2.723087918012828e+38f) },
                         { 88.72283172607422f, bit_of(3.402798519021476e+38f) }, // last float below the overflow bound
                         { 88.72283935546875f, inf_bits() },
                         { 89.0f, inf_bits() } };
            exp2_rows = { { 127.0f, bit_of(1.7014118346046923e+38f) },
                          { 127.9f, bit_of(3.17495018884007e+38f) },
                          { 128.0f, inf_bits() } };
            exp10_rows = { { 38.3f, bit_of(1.995258719988189e+38f) },
                           { 38.531837463378906f, bit_of(3.40280805175399e+38f) },
                           { 38.53184127807617f, inf_bits() } };
        }
        else
        {
            exp_rows = { { 0x1.62e3d70a3d70ap+9, bit_of(0x1.fe9ce5c4c52b4p+1023) }, // 709.78, k = 1024 scaling
                         { 0x1.62e42fefa39a0p+9, bit_of(0x1.ffffffffec32ap+1023) },
                         { 0x1.62e42fefa39bdp+9, bit_of(0x1.fffffffff372ap+1023) }, // top exp_ref row below the bound
                         { 0x1.62e42fefa39efp+9, bit_of(0x1.fffffffffff2ap+1023) }, // the old maxlog clamp row
                         { 710.0, inf_bits() },
#ifndef __FAST_MATH__
                         // the maxlog clamp is off under __FAST_MATH__; k far past maxexponent + 1 is garbage there
                         { 800.0, inf_bits() },
#endif
                         { -0x1.74910d52d3084p+9, 0 }, // -745.1332191019469, rounds to +0
                         { -0x1.74910d52d3020p+9, bit_of(0x1p-1074) } }; // -745.13321910193555, 1 subnormal ulp
            exp2_rows = { { 1023.0, bit_of(0x1p+1023) },
                          { 1023.5, bit_of(0x1.6a09e667f3bcdp+1023) },
                          { 1023.9, bit_of(0x1.ddb680117aa8ep+1023) },
                          { 1024.0, inf_bits() } };
            exp10_rows = { { 308.25, bit_of(0x1.fa788589d81d3p+1023) },
                           { 0x1.34413509f79fep+8, bit_of(0x1.ffffffffffba1p+1023) }, // the old maxlog10 clamp row
                           { 308.3, inf_bits() } };
        }
        check_overflow_rows("exp", exp_rows, 2, [](const batch_type& b)
                            { return exp(b); });
        check_overflow_rows("exp2", exp2_rows, 2, [](const batch_type& b)
                            { return exp2(b); });
        check_overflow_rows("exp10", exp10_rows, 2, [](const batch_type& b)
                            { return exp10(b); });
    }
};

TEST_CASE_TEMPLATE("[exponential]", B, BATCH_FLOAT_TYPES)
{
    exponential_test<B> Test;

    SUBCASE("exp")
    {
        Test.test_exponential_functions();
    }

    SUBCASE("log")
    {
        Test.test_log_functions();
    }

    SUBCASE("exp overflow")
    {
        Test.test_exp_overflow();
    }
}
#endif
