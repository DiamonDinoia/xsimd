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
#include <cstring>
#include <limits>
#include <string>

// The flag checks need FP exception flags and flag-free rounding kernels. wasm has
// no flags, the vsx/vxe kernels (ppc, s390x) are audited as raising or unverifiable,
// MSVC does not track FP exceptions reliably (its ARM64 and x86 runs fail here), and
// fast-math may reorder the FP ops past the fenv calls.
#if defined(FE_INEXACT) && !defined(__EMSCRIPTEN__) && !defined(__powerpc__) && !defined(__s390__) && !defined(_MSC_VER) && !defined(__FAST_MATH__)
#define XSIMD_ROUNDING_FENV_TEST 1
#endif

template <class B>
struct rounding_test
{
    using batch_type = B;
    using arch_type = typename B::arch_type;
    using value_type = typename B::value_type;
    using int_value_type = xsimd::as_integer_t<value_type>;
    using int_batch_type = xsimd::batch<int_value_type, arch_type>;
    static constexpr size_t size = B::size;
    static constexpr size_t nb_input = 16;
    static constexpr size_t nb_batches = nb_input / size;

    std::array<value_type, nb_input> input;
    std::array<value_type, nb_input> expected;

    rounding_test()
    {
        input[0] = value_type(-3.7);
        input[1] = value_type(-3.5);
        input[2] = value_type(-3.3);
        input[3] = value_type(-3.1);
        input[4] = value_type(-2.9);
        input[5] = value_type(-2.0);
        input[6] = value_type(-1.9);
        input[7] = value_type(-0.5);
        input[8] = value_type(0.5);
        input[9] = value_type(1.9);
        input[10] = value_type(2.0);
        input[11] = value_type(2.9);
        input[12] = value_type(3.1);
        input[13] = value_type(3.3);
        input[14] = value_type(3.5);
        input[15] = value_type(3.7);
    }

    void test_rounding_functions()
    {
        // ceil
        {
            std::transform(input.cbegin(), input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::ceil(v); });
            for (size_t i = 0; i < nb_batches; i++)
            {
                batch_type in, out, ref;
                detail::load_batch(in, input, i * size);
                out = ceil(in);
                detail::load_batch(ref, expected, i * size);
                INFO("ceil");
                CHECK_BATCH_EQ(ref, out);
            }
        }
        // floor
        {
            std::transform(input.cbegin(), input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::floor(v); });
            for (size_t i = 0; i < nb_batches; i++)
            {
                batch_type in, out, ref;
                detail::load_batch(in, input, i * size);
                out = floor(in);
                detail::load_batch(ref, expected, i * size);
                INFO("floor");
                CHECK_BATCH_EQ(ref, out);
            }
        }
        // trunc
        {
            std::transform(input.cbegin(), input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::trunc(v); });
            for (size_t i = 0; i < nb_batches; i++)
            {
                batch_type in, out, ref;
                detail::load_batch(in, input, i * size);
                out = trunc(in);
                detail::load_batch(ref, expected, i * size);
                INFO("trunc");
                CHECK_BATCH_EQ(ref, out);
            }
        }
        // round
        {
            std::transform(input.cbegin(), input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::round(v); });
            for (size_t i = 0; i < nb_batches; i++)
            {
                batch_type in, out, ref;
                detail::load_batch(in, input, i * size);
                out = round(in);
                detail::load_batch(ref, expected, i * size);
                INFO("round");
                CHECK_BATCH_EQ(ref, out);
            }
        }
        // nearbyint
        {
            std::transform(input.cbegin(), input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::nearbyint(v); });
            for (size_t i = 0; i < nb_batches; i++)
            {
                batch_type in, out, ref;
                detail::load_batch(in, input, i * size);
                out = nearbyint(in);
                detail::load_batch(ref, expected, i * size);
                INFO("nearbyint");
                CHECK_BATCH_EQ(ref, out);
            }
        }
        // nearbyint_as_int
        {
            std::array<int_value_type, nb_input> expected;
            std::transform(input.cbegin(), input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return xsimd::nearbyint_as_int(v); });
            for (size_t i = 0; i < nb_batches; i++)
            {
                batch_type in;
                int_batch_type out, ref;
                detail::load_batch(in, input, i * size);
                out = nearbyint_as_int(in);
                detail::load_batch(ref, expected, i * size);
                INFO("nearbyint_as_int");
                CHECK_BATCH_EQ(ref, out);
            }
        }
        // rint
        {
            std::transform(input.cbegin(), input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::rint(v); });
            for (size_t i = 0; i < nb_batches; i++)
            {
                batch_type in, out, ref;
                detail::load_batch(in, input, i * size);
                out = rint(in);
                detail::load_batch(ref, expected, i * size);
                INFO("rint");
                CHECK_BATCH_EQ(ref, out);
            }
        }
    }
};

TEST_CASE_TEMPLATE("[rounding]", B, BATCH_FLOAT_TYPES)
{

    rounding_test<B> Test;
    Test.test_rounding_functions();
}

#ifdef XSIMD_ROUNDING_FENV_TEST

namespace rounding_fenv_detail
{
    inline void barrier() noexcept
    {
#if defined(__GNUC__) || defined(__clang__)
        asm volatile("" ::: "memory");
#endif
    }

#if defined(_MSC_VER)
#define XSIMD_TEST_NOINLINE __declspec(noinline)
#else
#define XSIMD_TEST_NOINLINE __attribute__((noinline))
#endif

    // The simd call must run at runtime and stay between the two fenv calls. A
    // call cannot be reordered across them and cannot be constant folded away.
    template <class F, class B>
    XSIMD_TEST_NOINLINE B call(F f, B in) noexcept
    {
        return f(in);
    }

    // NaN payloads are unspecified across implementations, so compare the class only
    template <class T>
    inline bool same_bits(T a, T b) noexcept
    {
        if (std::isnan(a) && std::isnan(b))
        {
            return true;
        }
        return std::memcmp(&a, &b, sizeof(T)) == 0;
    }
}

template <class B>
struct rounding_fenv_test
{
    using batch_type = B;
    using value_type = typename B::value_type;
    static constexpr size_t size = B::size;
    static constexpr size_t nb_input = 16;
    static constexpr size_t nb_batches = nb_input / size;

    std::array<value_type, nb_input> input;

    rounding_fenv_test()
    {
        // The values cross a volatile load so the compiler cannot fold the simd
        // calls at compile time: folded FP math raises no flags at runtime.
        volatile value_type one = value_type(1);
        const value_type mf = std::ldexp(value_type(1), std::numeric_limits<value_type>::digits - 1);
        const value_type qnan = std::numeric_limits<value_type>::quiet_NaN();
        const value_type inf = std::numeric_limits<value_type>::infinity();
        input[0] = value_type(0.5);
        input[1] = value_type(-2.5);
        input[2] = value_type(3.75);
        input[3] = value_type(-3.75);
        input[4] = value_type(2.0);
        input[5] = value_type(-3.0);
        input[6] = value_type(0.0);
        input[7] = value_type(-0.0);
        input[8] = inf;
        input[9] = -inf;
        input[10] = qnan;
        input[11] = mf - value_type(0.5);
        input[12] = mf;
        input[13] = mf * value_type(2);
        input[14] = -(mf - value_type(0.5));
        input[15] = value_type(1.0);
        for (size_t i = 0; i < nb_input; ++i)
        {
            input[i] = input[i] * one;
        }
    }

    template <class F, class G>
    void check_flag_free(const char* name, F simd_fn, G std_fn)
    {
        for (size_t i = 0; i < nb_batches; ++i)
        {
            batch_type in;
            detail::load_batch(in, input, i * size);
            std::feclearexcept(FE_ALL_EXCEPT);
            rounding_fenv_detail::barrier();
            batch_type out = rounding_fenv_detail::call(simd_fn, in);
            rounding_fenv_detail::barrier();
            const int flags = std::fetestexcept(FE_ALL_EXCEPT);
            INFO(std::string(name), ": FE_INEXACT must stay clear");
            CHECK((flags & FE_INEXACT) == 0);
            value_type got[size];
            out.store_unaligned(got);
            for (size_t j = 0; j < size; ++j)
            {
                const value_type ref = std_fn(input[i * size + j]);
                INFO(std::string(name), ": lane ", i * size + j);
                CHECK(rounding_fenv_detail::same_bits(ref, got[j]));
            }
        }
    }

    void test_rounding_functions()
    {
        check_flag_free("ceil", [](batch_type x)
                        { return ceil(x); }, [](value_type v)
                        { return std::ceil(v); });
        check_flag_free("floor", [](batch_type x)
                        { return floor(x); }, [](value_type v)
                        { return std::floor(v); });
        check_flag_free("trunc", [](batch_type x)
                        { return trunc(x); }, [](value_type v)
                        { return std::trunc(v); });
        check_flag_free("round", [](batch_type x)
                        { return round(x); }, [](value_type v)
                        { return std::round(v); });
        check_flag_free("nearbyint", [](batch_type x)
                        { return nearbyint(x); }, [](value_type v)
                        { return std::nearbyint(v); });
        // 7.12.9.6: rint may raise inexact and must keep doing so when the result differs
        {
            batch_type in;
            detail::load_batch(in, input, 0);
            std::feclearexcept(FE_ALL_EXCEPT);
            rounding_fenv_detail::barrier();
            batch_type out = rounding_fenv_detail::call([](batch_type x)
                                                        { return rint(x); },
                                                        in);
            rounding_fenv_detail::barrier();
            INFO("rint: FE_INEXACT must be raised for non-integer input");
            CHECK((std::fetestexcept(FE_ALL_EXCEPT) & FE_INEXACT) != 0);
            value_type got[size];
            out.store_unaligned(got);
            for (size_t j = 0; j < size; ++j)
            {
                INFO("rint: lane ", j);
                CHECK(rounding_fenv_detail::same_bits(std::rint(input[j]), got[j]));
            }
            std::feclearexcept(FE_ALL_EXCEPT);
        }
    }
};

TEST_CASE_TEMPLATE("[rounding] fp exception flags", B, BATCH_FLOAT_TYPES)
{
    rounding_fenv_test<B> Test;
    Test.test_rounding_functions();
}
#endif
#endif
