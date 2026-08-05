// Operation tags and their public entry points. No intrinsics here: the tables
// live in the per-arch headers, which may be included after this one.
#ifndef PROTO_OPS_HPP
#define PROTO_OPS_HPP

#include "proto_dispatch.hpp"

namespace proto
{
    struct add_t
    {
    };

    template <class T, class A>
    XSIMD_INLINE batch<T, A> add(batch<T, A> a, batch<T, A> b) noexcept
    {
        return apply<add_t>(a, b);
    }
}

#endif
