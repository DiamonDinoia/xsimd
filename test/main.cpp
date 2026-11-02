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
#ifndef EMSCRIPTEN

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>
#include <cstdio>

// unbuffered stdout: if a run dies, the log ends at the last test that ran
int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    return context.run();
}
#else

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>
#include <emscripten/bind.h>

int run_tests()
{
    doctest::Context context;
    return context.run();
}

EMSCRIPTEN_BINDINGS(my_module)
{
    emscripten::function("run_tests", &run_tests);
}

#endif
