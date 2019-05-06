#! /usr/bin/env bash

set -e

MY_ANDROID_HOME="${ANDROID_HOME:-$HOME/Android/Sdk}"
MY_ANDROID_NDK_HOME="${ANDROID_NDK_HOME:-$MY_ANDROID_HOME/ndk-bundle}"
JNI_BUILD_DIR=build_jni_tmp
rm -rf ${JNI_BUILD_DIR} && mkdir ${JNI_BUILD_DIR} && pushd ${JNI_BUILD_DIR}
cmake -DCMAKE_SYSTEM_NAME=Android -DCMAKE_TOOLCHAIN_FILE=${MY_ANDROID_NDK_HOME}/build/cmake/android.toolchain.cmake -DANDROID_CPP_FEATURES=exceptions -DANDROID_PLATFORM=android-21 -DANDROID_ABI=arm64-v8a -DBNN_BUILD_JNI=ON -DBNN_BUILD_TEST=OFF -DBNN_BUILD_BENCHMARK=OFF ..
cmake --build . -- "-j$(nproc)"
popd
mkdir -p ci/android_aar/dabnn/src/main/jniLibs/arm64-v8a
cp ${JNI_BUILD_DIR}/dabnn/jni/libdabnn-jni.so ci/android_aar/dabnn/src/main/jniLibs/arm64-v8a/

# Increase version code and update version name
if (( $# >= 1 )); then
    NEW_VER_NUM=`sed -nE 's/versionCode ([0-9]+)/\1+1/p' ci/android_aar/dabnn/build.gradle | bc`
    sed -i -E "s/versionCode [0-9]+/versionCode $NEW_VER_NUM/" ci/android_aar/dabnn/build.gradle
    sed -i -E "s/versionName .+/versionName \"v$1\"/" ci/android_aar/dabnn/build.gradle
    sed -i -E "s/publishVersion = .+/publishVersion = \'$1\'/" ci/android_aar/dabnn/build.gradle
fi

pushd ci/android_aar
ANDROID_HOME=$MY_ANDROID_HOME ./gradlew clean build

# Publishing is only for myself
if (( $# == 2 )); then
	echo "Publishing.."
	ANDROID_HOME=$MY_ANDROID_HOME ./gradlew bintrayUpload -PbintrayUser=daquexian566 -PbintrayKey=$2 -PdryRun=false
fi
popd
