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

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
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

    template <class F>
    void check_overflow_rows(const char* name, const std::vector<std::pair<value_type, value_type>>& rows, std::int64_t tol, F simd_fn)
    {
        // wants are correctly rounded; rows near the overflow bound catch inf while the result is finite
        const value_type grid = (std::numeric_limits<value_type>::denorm_min)();
        for (const auto& r : rows)
        {
            batch_type in = batch_type(r.first);
            alignas(64) value_type got[size];
            simd_fn(in).store_aligned(got);
            for (size_t lane = 0; lane < size; ++lane)
            {
                bool ok;
                if (r.second == value_type(0))
                {
                    ok = std::abs(got[lane]) <= value_type(tol) * grid;
                }
                else if (std::isinf(r.second))
                {
                    ok = got[lane] == r.second;
                }
                else
                {
                    ok = !std::isinf(got[lane]) && std::llabs(bit_of(got[lane]) - bit_of(r.second)) <= tol;
                }
                if (!ok)
                {
                    std::fprintf(stderr, "%s: x=%a lane %zu got=%a want=%a\n", name, double(r.first), lane, double(got[lane]), double(r.second));
                }
                CHECK(ok);
            }
        }
    }

    void test_exp_overflow()
    {
        // regression for DiamonDinoia/xsimd#7: exp/exp2/exp10 must go to +inf only past the true
        // overflow bound and must stay within tol ulp of the correctly rounded value at the top
        const value_type pinf = std::numeric_limits<value_type>::infinity();
        std::vector<std::pair<value_type, value_type>> exp_rows, exp2_rows, exp10_rows;
        if constexpr (std::is_same<value_type, float>::value)
        {
            exp_rows = { { 88.5f, 2.723087918012828e+38f },
                         { 88.72283172607422f, 3.402798519021476e+38f }, // last float below the overflow bound
                         { 88.72283935546875f, pinf },
                         { 89.0f, pinf } };
            exp2_rows = { { 127.0f, 1.7014118346046923e+38f },
                          { 127.9f, 3.17495018884007e+38f },
                          { 128.0f, pinf } };
            exp10_rows = { { 38.3f, 1.995258719988189e+38f },
                           { 38.531837463378906f, 3.40280805175399e+38f },
                           { 38.53184127807617f, pinf } };
        }
        else
        {
            exp_rows = { { 0x1.62e3d70a3d70ap+9, 0x1.fe9ce5c4c52b4p+1023 }, // 709.78, k = 1024 scaling
                         { 0x1.62e42fefa39a0p+9, 0x1.ffffffffec32ap+1023 },
                         { 0x1.62e42fefa39bdp+9, 0x1.fffffffff372ap+1023 }, // top exp_ref row below the bound
                         { 0x1.62e42fefa39efp+9, 0x1.fffffffffff2ap+1023 }, // the old maxlog clamp row
                         { 710.0, pinf },
                         { 800.0, pinf },
                         { -0x1.74910d52d3084p+9, value_type(0) }, // -745.1332191019469, rounds to +0
                         { -0x1.74910d52d3020p+9, 0x1p-1074 } }; // -745.13321910193555, 1 subnormal ulp
            exp2_rows = { { 1023.0, 0x1p+1023 },
                          { 1023.5, 0x1.6a09e667f3bcdp+1023 },
                          { 1023.9, 0x1.ddb680117aa8ep+1023 },
                          { 1024.0, pinf } };
            exp10_rows = { { 308.25, 0x1.fa788589d81d3p+1023 },
                           { 0x1.34413509f79fep+8, 0x1.ffffffffffba1p+1023 }, // the old maxlog10 clamp row
                           { 308.3, pinf } };
        }
        check_overflow_rows("exp", exp_rows, 2, [](const batch_type& b)
                            { return exp(b); });
        check_overflow_rows("exp2", exp2_rows, 2, [](const batch_type& b)
                            { return exp2(b); });
        check_overflow_rows("exp10", exp10_rows, 2, [](const batch_type& b)
                            { return exp10(b); });
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
};

TEST_CASE_TEMPLATE("[exponential]", B, BATCH_FLOAT_TYPES)
{
    exponential_test<B> Test;

    SUBCASE("exp")
    {
        Test.test_exponential_functions();
    }

    SUBCASE("exp overflow")
    {
        Test.test_exp_overflow();
    }

    SUBCASE("log")
    {
        Test.test_log_functions();
    }
}
#endif
