#! /usr/bin/env bash

# echo "${@:2}"
adb push $1 /data/local/tmp/`basename $1` && adb shell "data/local/tmp/`basename $1` ${@:2}"
