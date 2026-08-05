#include "proto_add.hpp"

#include <cstdint>
#include <cstdio>

using A = xsimd::default_arch;

// Every element type resolves on this architecture.
static_assert(proto::covers<proto::add_t, A>(), "add has an unresolvable element type");

// Native-op expectations: these would have caught float-on-plain-AVX.
#if XSIMD_WITH_AVX
static_assert(proto::has_native_v<proto::add_t, float, xsimd::avx>, "vaddps is native on avx");
static_assert(proto::has_native_v<proto::add_t, double, xsimd::avx>, "vaddpd is native on avx");
static_assert(!proto::has_native_v<proto::add_t, int8_t, xsimd::avx>, "no vpaddb before avx2");
static_assert(!proto::all_native<proto::add_t, xsimd::avx>(), "avx splits integers");
#endif
#if XSIMD_WITH_AVX2
static_assert(proto::all_native<proto::add_t, xsimd::avx2>(), "avx2 is fully native");
#endif
#if XSIMD_WITH_AVX512F
static_assert(!proto::has_native_v<proto::add_t, int16_t, xsimd::avx512f>, "needs avx512bw");
static_assert(proto::has_native_v<proto::add_t, int32_t, xsimd::avx512f>, "vpaddd on avx512f");
#endif
#if XSIMD_WITH_AVX512BW
static_assert(proto::all_native<proto::add_t, xsimd::avx512bw>(), "avx512bw is fully native");
#endif
static_assert(proto::all_native<proto::add_t, xsimd::sse2>(), "sse2 is fully native");

// Fallback tiers are visible and asserted, not implicit.
static_assert(proto::tier_of<proto::add_t, float, xsimd::sse2>() == proto::tier::native, "");
#if XSIMD_WITH_AVX
static_assert(proto::tier_of<proto::add_t, int8_t, xsimd::avx>() == proto::tier::halves, "");
#endif
// An op with no table anywhere falls to the elementwise tier on sse2. Op tag and
// its scalar form must share a namespace, so ADL finds the registration.
namespace probe
{
    struct mul_t
    {
    };
    template <class T>
    constexpr auto scalar(mul_t, proto::tag<T>) noexcept
    {
        return [](T x, T y) noexcept { return T(x * y); };
    }
}
static_assert(proto::tier_of<probe::mul_t, float, xsimd::sse2>() == proto::tier::lanes, "");
static_assert(proto::covers<probe::mul_t, xsimd::sse2>(), "elementwise tier covers everything");
static_assert(!proto::all_native<probe::mul_t, xsimd::sse2>(), "elementwise is not native");

template <class T>
using B = xsimd::batch<T, A>;

#define GEN(T, name)                                                        \
    B<T> proto_##name(B<T> a, B<T> b) noexcept { return proto::add(a, b); } \
    B<T> ref_##name(B<T> a, B<T> b) noexcept { return xsimd::kernel::add<A>(a, b, A {}); }

GEN(float, f32)
GEN(double, f64)
GEN(int8_t, i8)
GEN(uint8_t, u8)
GEN(int16_t, i16)
GEN(uint16_t, u16)
GEN(int32_t, i32)
GEN(uint32_t, u32)
GEN(int64_t, i64)
GEN(uint64_t, u64)

int main()
{
    B<int32_t> a = 3, b = 4;
    B<float> x = 1.5f, y = 2.25f;
    std::printf("%d %f\n", (int)proto_i32(a, b).get(0), (double)proto_f32(x, y).get(0));
    return 0;
}
