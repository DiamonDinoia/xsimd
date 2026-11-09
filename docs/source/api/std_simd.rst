.. Copyright (c) 2016, Johan Mabille and Sylvain Corlay

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Interoperability with std::simd
===============================

`xsimd` can convert its batch types to and from the standard SIMD types
``std::simd::vec`` and ``std::simd::mask``, or ``std::experimental::simd``
and ``std::experimental::simd_mask`` where the C++26 type is absent. The
conversions are implicit, so a kernel written against the standard API and
a driver written with `xsimd` can call each other without casts.

Enabling the interop
--------------------

Define ``XSIMD_ENABLE_STD_SIMD`` for the whole build. With CMake, pass the
macro through ``CMAKE_CXX_FLAGS``:

.. code-block:: sh

    cmake -D CMAKE_CXX_FLAGS="-DXSIMD_ENABLE_STD_SIMD" ..

Without the macro the interop is off and the code compiles as before.

Which std type is used
----------------------

The header ``xsimd/types/xsimd_std_simd.hpp`` picks the standard type at
compile time:

- with ``__cplusplus > 202302L``, it includes ``<simd>`` and uses
  ``std::simd::vec`` and ``std::simd::mask`` when the standard library
  defines ``__cpp_lib_simd`` or ``__glibcxx_simd``;
- otherwise it includes ``<experimental/simd>`` and uses the
  ``std::experimental`` types when ``__cpp_lib_experimental_parallel_simd``
  is defined;
- when neither type exists, the header stops the build with ``#error``.

The CI enables the macro with GCC 11 and newer under libstdc++, because
``<experimental/simd>`` exists in libstdc++ from GCC 11, with clang on
libstdc++, and with GCC 16 in C++26 mode. No CI job enables the macro on
libc++: the macOS jobs do not pass the flag.

The conversions
---------------

When the macro is on, ``xsimd::batch<T, A>`` converts implicitly to and
from the std vector type of the same lane count, and
``xsimd::batch_bool<T, A>`` converts implicitly to and from the std mask
type of the same lane count. A ``(T, N)`` pair whose std type does not
exist gets no conversion, and a std type with a different lane count never
converts.

When the std type has a native register at the batch width, the conversion
goes through the SIMD register instead of memory. Other widths convert
through a load/store round trip. Masks use the same split: register to
register where the mask layouts match, a movemask-style bit pivot where the
std mask exposes no register, and load/store otherwise.

Example
-------

The example below shows the interop in practice. The ``soft_threshold``
kernel is pure std::simd code: it uses only the standard API and knows
nothing of xsimd. The driver loop in ``main`` uses ``xsimd::batch`` for
the loads, the arithmetic and the scalar tail. The call
``soft_threshold(b, threshold)`` passes a batch where a std vector is
expected, so the batch converts to the std type on entry and the result
converts back to a batch on return, without any explicit cast.

.. literalinclude:: ../../../examples/std_simd.cpp
   :start-at: // Soft-threshold denoising

The example closes with a scalar reference pass and checks the relative
error against a tolerance, so it doubles as a correctness test.
