#! /usr/bin/env bash

azeban_root="$(realpath "$(dirname "$(readlink -f "$0")")"/..)"

find ${azeban_root}/benchmarks -regex '.*\.\(cpp\|hpp\|cu\|cuh\)' -exec clang-format -i {} \;
find ${azeban_root}/include -regex '.*\.\(cpp\|hpp\|cu\|cuh\)' -exec clang-format -i {} \;
find ${azeban_root}/src -regex '.*\.\(cpp\|hpp\|cu\|cuh\)' -exec clang-format -i {} \;
find ${azeban_root}/test -regex '.*\.\(cpp\|hpp\|cu\|cuh\)' -exec clang-format -i {} \;
