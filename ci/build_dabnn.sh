#! /usr/bin/env bash
set -e

echo "y" | $ANDROID_HOME/tools/bin/sdkmanager --install 'ndk-bundle'
nproc=$(ci/get_cores.sh)

mkdir build_dabnn && cd build_dabnn
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_HOME/ndk-bundle/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -- -j$nproc
cd -
