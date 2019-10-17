## Build

We use CMake build system like most C++ projects. 

### Clone the project

Clone dabnn recursively

```bash
git clone --recursive https://github.com/JDAI-CV/dabnn
```

### Build dabnn

Cross-compiling for ARMv8 Android:

1. Download and unzip Android NDK from https://developer.android.com/ndk/downloads.

2. Run cmake with the toolchain file, which determine the proper compiling toolchains. **If this step fails, please check whether the toolchain file really exists on the path you set.**

```bash
mkdir build-dabnn
cd build-dabnn
cmake -DCMAKE_TOOLCHAIN_FILE=the_path_to_android_ndk/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DCMAKE_BUILD_TYPE=Release ..
```

3. Build

```bash
cmake --build .
```

For ARMv7, just replace `-DANDROID_ABI=arm64-v8a` with `-DANDROID_ABI=armeabi-v7a` in step 2.

For non-Android ARM devices, use the proper toolchain file for your device instead of the Android NDK toolchain file, or compile natively on your ARM device.

For non-ARM devices, only the unoptimized code will work. If you still want to build dabnn for non-ARM devices, pass `-DBNN_BUILD_MAIN_LIB=ON` in step 2.

### Build onnx2bnn

On non-ARM devices, just run cmake and build the project directly.

1. Run cmake

```bash
mkdir build-onnx2bnn
cd build-onnx2bnn
cmake ..
```

2. Build

```bash
cmake --build .
```

On ARM devices, pass `-DBNN_BUILD_MAIN_LIB=OFF` in step 1.
