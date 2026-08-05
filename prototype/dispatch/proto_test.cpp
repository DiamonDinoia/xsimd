#include "proto_add.hpp"

#include <cstdint>
#include <cstdio>

using A = xsimd::default_arch;

// Every element type resolves on this architecture.
PROTO_ASSERT_COVERAGE(proto::add_op, A);

// Native-op expectations: these would have caught float-on-plain-AVX.
#if XSIMD_WITH_AVX
static_assert(proto::has_native_v<proto::add_op, float, xsimd::avx>, "vaddps is native on avx");
static_assert(proto::has_native_v<proto::add_op, double, xsimd::avx>, "vaddpd is native on avx");
static_assert(!proto::has_native_v<proto::add_op, int8_t, xsimd::avx>, "no vpaddb before avx2");
static_assert(proto::has_native_v<proto::add_op, int8_t, xsimd::avx2>, "vpaddb is native on avx2");
#endif
#if XSIMD_WITH_AVX512F
static_assert(!proto::has_native_v<proto::add_op, int16_t, xsimd::avx512f>, "needs avx512bw");
static_assert(proto::has_native_v<proto::add_op, int16_t, xsimd::avx512bw>, "vpaddw on avx512bw");
static_assert(proto::has_native_v<proto::add_op, int32_t, xsimd::avx512f>, "vpaddd on avx512f");
#endif

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
GEN(int32_t, i32)
GEN(int64_t, i64)

int main()
{
    B<int32_t> a = 3, b = 4;
    B<float> x = 1.5f, y = 2.25f;
    std::printf("%d %f\n", (int)proto_i32(a, b).get(0), (double)proto_f32(x, y).get(0));
    return 0;
}
