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

#include <cfenv>
#include <cstdio>
#include <limits>
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

    static bool subnormals_survive_simd()
    {
        // some targets flush subnormal SIMD results to zero (e.g. ARMv7 NEON) and cannot hold them
        alignas(64) value_type buf[size];
        volatile value_type tiny = (std::numeric_limits<value_type>::denorm_min)();
        for (size_t i = 0; i < size; ++i)
        {
            buf[i] = tiny;
        }
        (B::load_aligned(buf) + B::load_aligned(buf)).store_aligned(buf);
        return buf[0] != value_type(0);
    }

    void test_exp_subnormal()
    {
        // regression for DiamonDinoia/xsimd#3: exp/exp2/exp10 must not flush subnormal results to +0
        std::vector<value_type> exp_points;
        std::vector<value_type> exp2_points;
        std::vector<value_type> exp10_points;
        if constexpr (std::is_same<value_type, float>::value)
        {
            exp_points = { -87.0f, -87.4f, -88.5f, -92.0f, -98.0f, -103.5f, -103.9f, -104.0f, -104.5f, -std::numeric_limits<float>::infinity() };
            exp2_points = { -126.5f, -127.0f, -140.0f, -149.5f, -150.0f, -151.0f, -std::numeric_limits<float>::infinity() };
            exp10_points = { -38.2f, -39.0f, -42.0f, -45.1f, -45.2f, -45.5f, -std::numeric_limits<float>::infinity() };
        }
        else
        {
            exp_points = { -708.0, -708.4, -709.5, -720.0, -740.0, -744.5, -745.1, -745.2, -746.0, -std::numeric_limits<double>::infinity() };
            exp2_points = { -1022.5, -1023.0, -1050.0, -1074.5, -1075.0, -1076.0, -std::numeric_limits<double>::infinity() };
            exp10_points = { -308.3, -309.0, -315.0, -323.5, -323.7, -324.0, -std::numeric_limits<double>::infinity() };
        }
        if (!subnormals_survive_simd())
        {
            return;
        }
        const value_type grid = (std::numeric_limits<value_type>::denorm_min)();
        const auto check = [&](const char* name, const std::vector<value_type>& pts, value_type tol, value_type (*ref)(value_type), auto&& simd_fn)
        {
            for (size_t i = 0; i < pts.size(); ++i)
            {
                batch_type in(pts[i]);
                alignas(64) value_type got[size];
                simd_fn(in).store_aligned(got);
                for (size_t lane = 0; lane < size; ++lane)
                {
                    value_type want = ref(pts[i]);
                    // libm and the kernel may round the smallest subnormals differently; -inf must give exactly +0
                    bool ok = (std::abs(got[lane] - want) <= tol * grid)
                        && !(std::isinf(pts[i]) && pts[i] < value_type(0) && got[lane] != value_type(0));
                    if (!ok)
                    {
                        std::fprintf(stderr, "%s: x=%g lane %zu got=%a want=%a\n", name, double(pts[i]), lane, double(got[lane]), double(want));
                        CHECK(ok);
                    }
                }
            }
        };
        check("exp", exp_points, value_type(2), [](value_type v)
              { return std::exp(v); }, [](const batch_type& b)
              { return exp(b); });
        check("exp2", exp2_points, value_type(2), [](value_type v)
              { return std::exp2(v); }, [](const batch_type& b)
              { return exp2(b); });
#ifndef __FAST_MATH__ // exp10's compensated range reduction loses accuracy when reassociated, not a subnormal-formation defect
        check("exp10", exp10_points, value_type(4), [](value_type v)
              { return std::pow(value_type(10), v); }, // exp10 is 2 ulp in the normal range
              [](const batch_type& b)
              { return exp10(b); });
#endif
#if defined(FE_INVALID) && !defined(EMSCRIPTEN) // emscripten has no FP exception flags
        for (size_t i = 0; i + 1 < exp_points.size(); ++i) // -inf is excluded: the range reduction of +-inf is not exact
        {
            batch_type in(exp_points[i]);
            alignas(64) value_type got[size];
            std::feclearexcept(FE_ALL_EXCEPT);
            exp(in).store_aligned(got);
            CHECK_FALSE(std::fetestexcept(FE_INVALID) != 0);
            CHECK_FALSE(std::fetestexcept(FE_OVERFLOW) != 0);
        }
#endif
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
    SUBCASE("exp subnormal")
    {
        Test.test_exp_subnormal();
    }

    SUBCASE("log")
    {
        Test.test_log_functions();
    }
}
#endif
