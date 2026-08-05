// C++17 table-driven kernel dispatch for xsimd (follow-up to PR #1394).
//
// An operation is an empty tag type. Each arch header registers a table for it
// by defining one overload of
//
//     template <class T> constexpr auto table(op_tag, tag<T>, arch)
//
// returning by_type<T>(...) -- <f32, f64, i8, i16, i32, i64>. A missing entry
// is `nullptr`; an arch with no overload at all is simply not registered. Both
// are serviced by the fallback tiers below, or fail with a static_assert naming
// T and A.
//
// Fallback tiers, tried in order (see `enum class tier`):
//     native  an intrinsic registered for this element type and width
//     halves  split into two narrower batches and recurse
//     lanes   the op's scalar form, registered as scalar(op_tag, tag<T>)
//     none    hard error
//
// Overload resolution on the arch tag does width selection and refinement
// (avx2 refines avx, avx512bw refines avx512f) for free, so there is no central
// width table and no macro registration step. One file per width; an arch that
// only refines an existing one lists just what it adds.
//
// Debug predicates (compile-time -- nothing is decided at runtime):
//     covers<Op, A>()      every element type resolves on A, recursively
//     all_native<Op, A>()  ... and none of them go through the halving fallback
//     PROTO_NO_IMPLICIT_FALLBACK   build-wide: any halving fallback is an error
//     has_native_v<Op, T, A>   per-pair, for a targeted static_assert
#ifndef PROTO_DISPATCH_HPP
#define PROTO_DISPATCH_HPP

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <xsimd/xsimd.hpp>

namespace proto
{
    using namespace xsimd;

    /// Carries an element type through overload resolution without needing a value.
    template <class T>
    struct tag
    {
    };

    // ----------------------------------------------------------------- table

    /// A missing table entry is `nullptr`: no callable here.
    template <class F>
    inline constexpr bool is_none_v = std::is_same_v<std::decay_t<F>, std::nullptr_t>;

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
    /// Signedness-agnostic; ops where it matters use by_type_signed.
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

    static_assert(type_slot_signed<float>() == 0 && type_slot_signed<double>() == 1
                      && type_slot_signed<int8_t>() == 2 && type_slot_signed<uint8_t>() == 3
                      && type_slot_signed<int16_t>() == 4 && type_slot_signed<uint16_t>() == 5
                      && type_slot_signed<int32_t>() == 6 && type_slot_signed<uint32_t>() == 7
                      && type_slot_signed<int64_t>() == 8 && type_slot_signed<uint64_t>() == 9,
                  "type_slot_signed must map the ten element types to distinct ordered slots");

    /// Table indexed by element type: <f32, f64, i8, u8, i16, u16, i32, u32, i64, u64>.
    template <class T, class... Fs>
    XSIMD_INLINE constexpr auto by_type_signed(Fs... fs) noexcept
    {
        static_assert(sizeof...(Fs) == 10, "expected <f32, f64, i8, u8, ..., i64, u64>");
        return nth<type_slot_signed<T>()>(fs...);
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
        {
            return d::merge_sse(lo, hi);
        }
#if XSIMD_WITH_AVX512F
        else if constexpr (sizeof(lo) == 32)
        {
            return d::merge_avx(lo, hi);
        }
#endif
        else
        {
            unsupported<T, A>();
        }
#else
        (void)f, (void)a, (void)b;
        static_assert(!std::is_same_v<A, A>, "halving requires AVX: no 256/512-bit register to split");
#endif
    }

    /// Elementwise fallback: run the op's scalar form over the lanes. Reached only
    /// when no register-level intrinsic exists at this width and the batch cannot
    /// be split any further (e.g. an unregistered element type on sse2).
    template <class Op, class T, class A>
    XSIMD_INLINE batch<T, A> apply_scalar(batch<T, A> a, batch<T, A> b) noexcept
    {
        constexpr std::size_t n = batch<T, A>::size;
        alignas(A::alignment()) T xa[n], xb[n];
        a.store_aligned(xa);
        b.store_aligned(xb);
        constexpr auto f = scalar(Op {}, tag<T> {});
        for (std::size_t i = 0; i < n; ++i)
            xa[i] = f(xa[i], xb[i]);
        return batch<T, A>::load_aligned(xa);
    }

    // -------------------------------------------------------------- dispatch

    /// The registered table entry for (Op, T, A). Resolved by ADL at the point of
    /// instantiation, so arch headers may be included after this one. There is no
    /// catch-all overload on purpose: a catch-all is an exact match on the arch
    /// parameter and would beat every registration that needs a derived-to-base
    /// conversion (sse4_2 -> sse2). Absence is detected instead.
    template <class Op, class T, class A>
    inline constexpr auto native_v = table(Op {}, tag<T> {}, A {});

    template <class Op, class T, class A, class = void>
    inline constexpr bool registered_v = false;

    template <class Op, class T, class A>
    inline constexpr bool registered_v<Op, T, A, std::void_t<decltype(table(Op {}, tag<T> {}, A {}))>> = true;

    template <class Op, class T, class A>
    constexpr bool has_native() noexcept
    {
        if constexpr (registered_v<Op, T, A>)
            return !is_none_v<decltype(native_v<Op, T, A>)>;
        else
            return false;
    }

    template <class Op, class T, class A>
    inline constexpr bool has_native_v = has_native<Op, T, A>();

    /// Is a scalar form of Op registered for T?
    template <class Op, class T, class = void>
    inline constexpr bool has_scalar_v = false;

    template <class Op, class T>
    inline constexpr bool has_scalar_v<Op, T, std::void_t<decltype(scalar(Op {}, tag<T> {}))>> = true;

    /// Fallback tiers, in the order the dispatcher tries them.
    enum class tier
    {
        native, ///< an intrinsic registered for this element type and width
        halves, ///< split into two narrower batches and recurse
        lanes,  ///< the op's scalar form, one element at a time
        none,   ///< nothing applies: hard error
    };

    template <class Op, class T, class A>
    constexpr tier tier_of() noexcept
    {
        if constexpr (has_native_v<Op, T, A>)
            return tier::native;
        else if constexpr (splittable_v<A>)
            return tier::halves;
        else if constexpr (has_scalar_v<Op, T>)
            return tier::lanes;
        else
            return tier::none;
    }

    /// Pick the native intrinsic, else split in halves, else go elementwise, else fail.
    template <class Op, class T, class A>
    XSIMD_INLINE batch<T, A> apply(batch<T, A> a, batch<T, A> b) noexcept
    {
        constexpr tier t = tier_of<Op, T, A>();

        // Anything below tier::native is correct but slower than a register-level
        // intrinsic, and silent. Define PROTO_NO_IMPLICIT_FALLBACK to turn every
        // such degradation -- halving or elementwise -- into an error naming the
        // (op, T, arch) triple that is not running natively.
#ifdef PROTO_NO_IMPLICIT_FALLBACK
        static_assert(t == tier::native,
                      "implicit fallback: no native intrinsic registered for this element type and width");
#endif

        if constexpr (t == tier::native)
        {
            return batch<T, A> { native_v<Op, T, A>(a, b) };
        }
        else if constexpr (t == tier::halves)
        {
            return batch<T, A> { apply_on_halves([](auto l, auto r)
                                                 { return apply<Op>(l, r); },
                                                 a, b) };
        }
        else if constexpr (t == tier::lanes)
        {
            return apply_scalar<Op>(a, b);
        }
        else
        {
            unsupported<T, A>();
            return batch<T, A> {};
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
            return has_scalar_v<Op, T>;
    }

    template <class Op, class A, class... Ts>
    constexpr bool covers(type_list<Ts...>) noexcept
    {
        return (resolves<Op, Ts, A>() && ...);
    }

    /// Every element type resolves on A. A hole is a compile error naming T and A.
    template <class Op, class A>
    constexpr bool covers() noexcept
    {
        return covers<Op, A>(element_types {});
    }

    template <class Op, class A, class... Ts>
    constexpr bool all_native(type_list<Ts...>) noexcept
    {
        return (has_native_v<Op, Ts, A> && ...);
    }

    /// Every element type runs natively on A -- no silent halving fallback.
    /// This is the check that catches a missing table as a performance bug
    /// rather than letting it degrade quietly into a split.
    template <class Op, class A>
    constexpr bool all_native() noexcept
    {
        return all_native<Op, A>(element_types {});
    }
}

#endif
