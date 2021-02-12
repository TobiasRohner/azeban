#/usr/bin/env bash

if [[ "$#" -ne 1 ]]
then
    echo "Usage: $0 CC"
fi

cc=$(basename "$1")
prefix=$(dirname "$1")
if [[ "$prefix" == "." ]]
then
    prefix=""
else
    prefix="${prefix}/"
fi

if [[ "$cc" == "gcc" ]]
then
    echo ${prefix}g++
elif [[ "$cc" == "clang" ]]
then
    echo ${prefix}clang++
else
    echo "Heuristics for this compiler need to be implemented." >&2
    exit -1
fi
