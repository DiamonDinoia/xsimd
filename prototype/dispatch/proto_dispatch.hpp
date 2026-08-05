// C++17 generic dispatch prototype for xsimd (follow-up to PR #1394).
//
// Two compile-time tables per operation:
//   by_type<T>(...)   -- <f32, f64, i8, i16, i32, i64>, one entry per element type
//   by_width<T, A>(..)-- <512, 256, 128>, one entry per register width
// A missing entry is `nullptr`; the dispatcher services it by splitting the
// batch in halves and recursing, or fails with a named static_assert.
//
// Debug knobs:
//   PROTO_ASSERT_COVERAGE(Op, A)  -- every element type resolves on A
//   PROTO_STRICT_NATIVE           -- make any halving fallback a hard error
#ifndef PROTO_DISPATCH_HPP
#define PROTO_DISPATCH_HPP

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <xsimd/xsimd.hpp>

namespace proto
{
    using namespace xsimd;

    // ----------------------------------------------------------------- table

    /// A missing table entry is `nullptr`: no callable here.
    template <class F>
    inline constexpr bool is_none_v = std::is_same_v<F, std::nullptr_t>;

    /// Keep @p f only when the intrinsic exists on this architecture.
    template <bool Cond, class F>
    XSIMD_INLINE constexpr auto only_if(F f) noexcept
    {
        if constexpr (Cond)
            return f;
        else
            return nullptr;
    }

    template <std::size_t I, class F0, class... Fs>
    XSIMD_INLINE constexpr auto nth(F0 f0, Fs... fs) noexcept
    {
        if constexpr (I == 0)
            return f0;
        else
            return nth<I - 1>(fs...);
    }

    template <class T>
    constexpr std::size_t type_slot() noexcept
    {
        if constexpr (std::is_same_v<T, float>)
            return 0;
        else if constexpr (std::is_same_v<T, double>)
            return 1;
        else
            return sizeof(T) == 1 ? 2 : sizeof(T) == 2 ? 3 : sizeof(T) == 4 ? 4 : 5;
    }

    /// Table indexed by element type: <f32, f64, i8, i16, i32, i64>.
    /// Signedness-agnostic; ops where it matters use by_type_signed (below).
    template <class T, class... Fs>
    XSIMD_INLINE constexpr auto by_type(Fs... fs) noexcept
    {
        static_assert(sizeof...(Fs) == 6, "expected <f32, f64, i8, i16, i32, i64>");
        return nth<type_slot<T>()>(fs...);
    }

    template <class T>
    constexpr std::size_t type_slot_signed() noexcept
    {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>)
            return type_slot<T>();
        else
            return 2 * type_slot<T>() - 2 + std::is_unsigned_v<T>;
    }

    /// Table indexed by element type: <f32, f64, i8, u8, i16, u16, i32, u32, i64, u64>.
    template <class T, class... Fs>
    XSIMD_INLINE constexpr auto by_type_signed(Fs... fs) noexcept
    {
        static_assert(sizeof...(Fs) == 10, "expected <f32, f64, i8, u8, ..., i64, u64>");
        return nth<type_slot_signed<T>()>(fs...);
    }

    /// Table indexed by register width: <512, 256, 128>.
    template <class A, class F512, class F256, class F128>
    XSIMD_INLINE constexpr auto by_width(F512 f512, F256 f256, F128 f128) noexcept
    {
        if constexpr (std::is_base_of_v<avx512f, A>)
            return f512;
        else if constexpr (std::is_base_of_v<avx, A>)
            return f256;
        else if constexpr (std::is_base_of_v<sse2, A>)
            return f128;
        else
            return nullptr;
    }

    // -------------------------------------------------------------- fallback

    template <class T, class A>
    constexpr void unsupported() noexcept
    {
        static_assert(!std::is_same_v<A, A>, "unsupported element type for this architecture");
    }

    template <class A>
    using half_arch_t = std::conditional_t<
        std::is_base_of_v<avx512f, A>, avx2,
        std::conditional_t<std::is_base_of_v<avx, A>, sse4_2, void>>;

    template <class A>
    inline constexpr bool splittable_v = !std::is_void_v<half_arch_t<A>>;

    template <class F, class T, class A>
    XSIMD_INLINE auto apply_on_halves(F f, batch<T, A> a, batch<T, A> b) noexcept
    {
        namespace d = kernel::detail;
#if XSIMD_WITH_AVX
        using half = batch<T, half_arch_t<A>>;
        auto lo = f(half(d::lower_half(a.data)), half(d::lower_half(b.data))).data;
        auto hi = f(half(d::upper_half(a.data)), half(d::upper_half(b.data))).data;
        if constexpr (sizeof(lo) == 16)
            return d::merge_sse(lo, hi);
#if XSIMD_WITH_AVX512F
        else
            return d::merge_avx(lo, hi);
#endif
#endif
    }

    // -------------------------------------------------------------- dispatch

    /// True when Op has a native intrinsic for this element type and register width.
    template <class Op, class T, class A>
    inline constexpr bool has_native_v = !is_none_v<decltype(Op::template native<T, A>())>;

    /// Pick the native op, else split in halves and recurse, else fail.
    template <class Op, class T, class A, class... Bs>
    XSIMD_INLINE batch<T, A> dispatch(Bs... bs) noexcept
    {
        if constexpr (has_native_v<Op, T, A>)
        {
            return batch<T, A> { Op::template native<T, A>()(bs...) };
        }
        else if constexpr (splittable_v<A>)
        {
#ifdef PROTO_STRICT_NATIVE
            static_assert(has_native_v<Op, T, A>, "no native intrinsic: falling back to halves");
#endif
            return batch<T, A> { apply_on_halves([](auto l, auto r)
                                                 { return Op::apply(l, r); },
                                                 bs...) };
        }
        else
        {
            unsupported<T, A>();
        }
    }

    // -------------------------------------------------------------- coverage

    template <class...>
    struct type_list
    {
    };

    using element_types = type_list<float, double, int8_t, uint8_t, int16_t, uint16_t,
                                    int32_t, uint32_t, int64_t, uint64_t>;

    /// Does (Op, T, A) resolve -- natively, or by splitting all the way down?
    template <class Op, class T, class A>
    constexpr bool resolves() noexcept
    {
        if constexpr (has_native_v<Op, T, A>)
            return true;
        else if constexpr (splittable_v<A>)
            return resolves<Op, T, half_arch_t<A>>();
        else
            return false;
    }

    template <class Op, class A, class... Ts>
    constexpr bool covers(type_list<Ts...>) noexcept
    {
        return (resolves<Op, Ts, A>() && ...);
    }

    template <class Op, class A>
    constexpr bool covers() noexcept
    {
        return covers<Op, A>(element_types {});
    }
}

/// Compile-time check that Op has no hole on architecture A.
#define PROTO_ASSERT_COVERAGE(Op, A) \
    static_assert(::proto::covers<Op, A>(), #Op " has an unresolvable element type on " #A)

#endif
