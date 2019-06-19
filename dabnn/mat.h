// Copyright 2019 JD.com Inc. JD AI

// Tencent is pleased to support the open source community by making ncnn
// available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#ifndef NCNN_MAT_H
#define NCNN_MAT_H

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#if __ARM_NEON
#include <arm_neon.h>
#endif
#include <common/helper.h>
#include "allocator.h"

namespace bnn {

enum class DataType { Float, Bit };

// the three dimension matrix
// ncnn Mat is CHW, our mat is NHWC
class Mat {
   public:
    // empty
    Mat();
    // vec
    Mat(int w, DataType data_type);
    // image
    Mat(int w, int h, DataType data_type);
    // dim
    Mat(int w, int h, int c, DataType data_type, std::string name = "");
    // Conv weight or multi-batch blob
    Mat(int n, int w, int h, int c, DataType data_type,
        bool require_align = true);
    Mat(int n, int w, int h, int c, DataType data_type, size_t data_num,
        bool require_align = true);
    // external vec
    Mat(int w, void *data, DataType data_type);
    // external image
    Mat(int w, int h, void *data, DataType data_type);
    // external dim
    Mat(int w, int h, int c, void *data, DataType data_type);
    // external dim
    Mat(int n, int w, int h, int c, void *data, DataType data_type,
        bool require_align = true);
    Mat(int n, int w, int h, int c, void *data, DataType data_type,
        size_t data_num, bool require_align = true);

    Mat subMat(int w1, int w2, int h1, int h2);
    // release
    ~Mat();
    // delete copy constructor and copy assignment
    Mat(const Mat &) = delete;
    Mat(Mat &&) = default;
    Mat &operator=(const Mat &) = delete;
    // equality
    bool operator==(const Mat &m) const;
    // set all
    void fill(float v);
    void fill(int v);
    template <typename T>
    void fill(T v);
    // deep copy
    Mat clone() const;
    // allocate vec
    void create(int w, DataType data_type);
    // allocate image
    void create(int w, int h, DataType data_type);
    // allocate dim
    void create(int w, int h, int c, DataType data_type);

    void create(int n, int w, int h, int c, DataType data_type,
                bool require_align = true);
    void release();

    bool empty() const;
    size_t total() const;

    void dump(css &filename);

    bool external_memory = false;

    // data reference
    template <typename T>
    inline const T *point(int _n, int _h, int _w) const;
    template <typename T>
    inline const T *point(int _h, int _w) const;
    template <typename T>
    inline T *point(int _n, int _h, int _w);
    template <typename T>
    inline T *point(int _h, int _w);

    // access raw data
    template <typename T>
    operator T *();
    template <typename T>
    operator const T *() const;

    // convenient access float vec element
    float &operator[](int i);
    const float &operator[](int i) const;

    Mat flatten();

    friend std::ostream &operator<<(std::ostream &os, const Mat &mat);

    // pointer to the data
    void *data;

    // pointer to the reference counter
    // when points to user-allocated data, the pointer is NULL
    int *refcount;

    // element size in bytes
    // 4 = float32/int32
    // 2 = float16
    // 1 = int8/uint8
    // 0 = empty
    size_t elemsize;

    // the dimensionality
    int dims;

    int n;
    int w;
    int h;
    // TODO: A less error-prone way to distinguish the channel in bit and the
    // channel in uint64_t
    int c;  // since for bit mat, a uint64_t number stands for 64 unit, so
            // elem_c = c * 64
    int elem_c;

    size_t hstep;

    DataType data_type;

    size_t data_num_ = 0;

    std::string name;
};

inline Mat::Mat()
    : data(nullptr),
      elemsize(0),
      dims(0),
      w(0),
      h(0),
      c(0),
      hstep(0),
      data_type(DataType::Float) {}

inline Mat::Mat(int _w, DataType data_type)
    : data(nullptr), dims(0), data_type(data_type) {
    create(_w, data_type);
}

inline Mat::Mat(int _w, int _h, DataType data_type)
    : data(nullptr), dims(0), data_type(data_type) {
    if (data_type == DataType::Bit) {
        _h /= 64;
    }
    create(_w, _h, data_type);
}

inline Mat::Mat(int _w, int _h, int _c, DataType data_type, std::string name)
    : data(nullptr), dims(0), data_type(data_type), name(name) {
    elem_c = _c;
    if (data_type == DataType::Bit) {
        _c /= 64;
    }
    create(_w, _h, _c, data_type);
}

inline Mat::Mat(int _n, int _w, int _h, int _c, DataType data_type,
                bool require_align)
    : Mat(_n, _w, _h, _c, data_type, 0, require_align) {}

inline Mat::Mat(int _n, int _w, int _h, int _c, DataType data_type,
                size_t data_num, bool require_align)
    : data(nullptr), dims(0), data_type(data_type) {
    if (data_num != 0) {
        data_num_ = data_num;
    }
    elem_c = _c;
    if (data_type == DataType::Bit) {
        _c /= 64;
    }
    create(_n, _w, _h, _c, data_type, require_align);
}

inline Mat::Mat(int _w, void *_data, DataType data_type)
    : data(_data), dims(1), data_type(data_type) {
    n = 1;
    w = _w;
    h = 1;
    c = 1;
    elemsize = data_type == DataType::Float ? sizeof(float) : sizeof(uint64_t);

    hstep = w;

    external_memory = true;
}

inline Mat::Mat(int _w, int _h, void *_data, DataType data_type)
    : data(_data), dims(2), data_type(data_type) {
    n = 1;
    w = _w;
    h = _h;
    c = 1;
    elemsize = data_type == DataType::Float ? sizeof(float) : sizeof(uint64_t);

    hstep = w * 1;

    external_memory = true;
}

inline Mat::Mat(int _w, int _h, int _c, void *_data, DataType data_type)
    : data(_data), dims(3), data_type(data_type) {
    n = 1;
    w = _w;
    h = _h;
    c = _c;
    elem_c = _c;
    if (data_type == DataType::Bit) {
        c /= 64;
    }
    elemsize = data_type == DataType::Float ? sizeof(float) : sizeof(uint64_t);

    std::stringstream ss;
    ss << "Not align, w: " << w << ", c: " << c << ", elemsize: " << elemsize;
    BNN_ASSERT(w * c == 1 || w * c * elemsize % 16 == 0, ss.str());
    hstep = ncnn::alignSize(w * c * elemsize, 16) / elemsize;
    BNN_ASSERT(hstep > 0, hstep);

    external_memory = true;
}

inline Mat::Mat(int _n, int _w, int _h, int _c, void *_data, DataType data_type,
                bool require_align)
    : Mat(_n, _w, _h, _c, _data, data_type, 0, require_align) {}

inline Mat::Mat(int _n, int _w, int _h, int _c, void *_data, DataType data_type,
                size_t data_num, bool require_align)
    : data(_data), dims(4), data_type(data_type) {
    n = _n;
    w = _w;
    h = _h;
    c = _c;
    elem_c = _c;
    if (data_type == DataType::Bit) {
        c /= 64;
    }
    if (data_num != 0) {
        data_num_ = data_num;
        BNN_ASSERT(data_num_ > static_cast<size_t>(n * w * h * c), "data_num_ ",
                   data_num_, " shoule be larger than n * w * h * c, ", n, ", ", w,
                   ", ", h, ", ", c);
    }
    elemsize = data_type == DataType::Float ? sizeof(float) : sizeof(uint64_t);
    BNN_ASSERT(c > 0, c);
    std::stringstream ss;
    ss << "Not align, w: " << w << ", c: " << c << ", elemsize: " << elemsize;
    BNN_ASSERT(!require_align || w * c == 1 || w * c * elemsize % 16 == 0,
               ss.str());
    if (require_align) {
        hstep = ncnn::alignSize(w * c * elemsize, 16) / elemsize;
    } else {
        hstep = w * c;
    }
    BNN_ASSERT(hstep > 0, hstep);

    external_memory = true;
}

inline Mat::~Mat() { release(); }

inline bool Mat::operator==(const Mat &m) const {
    if (this == &m || data == m.data) {
        return true;
    }
    if (!(dims == m.dims && elemsize == m.elemsize && n == m.n && w == m.w &&
          h == m.h && c == m.c && data_type == m.data_type)) {
        return false;
    }
    if (m.data_type == DataType::Float) {
        FORZ(i, total()) {
            const auto elem = static_cast<float *>(data)[i];
            if (std::isnan(elem) && !std::isnan(m[i])) {
                PNT(elem, m[i]);
                return false;
            }
            if (!std::isnan(elem) && std::isnan(m[i])) {
                PNT(elem, m[i]);
                return false;
            }
            if (std::abs(elem - m[i]) > 1e-5) {
                PNT(i, elem, m[i]);
                return false;
            }
        }
    } else if (m.data_type == DataType::Bit) {
        FORZ(i, total()) {
            const auto elem = static_cast<uint64_t *>(data)[i];
            if (elem != m[i]) {
                PNT(elem, m[i]);
                return false;
            }
        }
    } else {
        throw std::invalid_argument("Unknown datatype");
    }
    return true;
}

inline std::ostream &operator<<(std::ostream &os, const Mat &mat) {
    os << "n: " << mat.n << ", width: " << mat.w << ", height: " << mat.h
       << ", channels: " << mat.c << std::endl;
    if (mat.data_type == DataType::Bit) {
        return os << binrep(static_cast<char *>(mat.data),
                            std::min(mat.total(), size_t{10}) * mat.elemsize,
                            true);
    } else {
        for (size_t i = 0;
             i < std::min(static_cast<decltype(mat.total())>(10), mat.total());
             i++) {
            // for (size_t i = 0; i < mat.total(); i++) {
            os << mat[i] << ", ";
        }
        return os;
    }
}

template <>
inline void Mat::fill(float _v) {
    int size = total();
    float *ptr = (float *)data;

#if __ARM_NEON
    int nn = size >> 2;
    int remain = size - (nn << 2);
#else
    int remain = size;
#endif  // __ARM_NEON

#if __ARM_NEON
    float32x4_t _c = vdupq_n_f32(_v);
#if __aarch64__
    if (nn > 0) {
        asm volatile(
            "0:                             \n"
            "subs       %w0, %w0, #1        \n"
            "st1        {%4.4s}, [%1], #16  \n"
            "bne        0b                  \n"
            : "=r"(nn),  // %0
              "=r"(ptr)  // %1
            : "0"(nn), "1"(ptr),
              "w"(_c)  // %4
            : "cc", "memory");
    }
#else
    if (nn > 0) {
        asm volatile(
            "0:                             \n"
            "subs       %0, #1              \n"
            "vst1.f32   {%e4-%f4}, [%1 :128]!\n"
            "bne        0b                  \n"
            : "=r"(nn),  // %0
              "=r"(ptr)  // %1
            : "0"(nn), "1"(ptr),
              "w"(_c)  // %4
            : "cc", "memory");
    }
#endif  // __aarch64__
#endif  // __ARM_NEON
    for (; remain > 0; remain--) {
        *ptr++ = _v;
    }
}

template <>
inline void Mat::fill(int _v) {
    int size = total();
    int *ptr = (int *)data;

#if __ARM_NEON
    int nn = size >> 2;
    int remain = size - (nn << 2);
#else
    int remain = size;
#endif  // __ARM_NEON

#if __ARM_NEON
    int32x4_t _c = vdupq_n_s32(_v);
#if __aarch64__
    if (nn > 0) {
        asm volatile(
            "0:                             \n"
            "subs       %w0, %w0, #1        \n"
            "st1        {%4.4s}, [%1], #16  \n"
            "bne        0b                  \n"
            : "=r"(nn),  // %0
              "=r"(ptr)  // %1
            : "0"(nn), "1"(ptr),
              "w"(_c)  // %4
            : "cc", "memory");
    }
#else
    if (nn > 0) {
        asm volatile(
            "0:                             \n"
            "subs       %0, #1              \n"
            "vst1.s32   {%e4-%f4}, [%1 :128]!\n"
            "bne        0b                  \n"
            : "=r"(nn),  // %0
              "=r"(ptr)  // %1
            : "0"(nn), "1"(ptr),
              "w"(_c)  // %4
            : "cc", "memory");
    }
#endif  // __aarch64__
#endif  // __ARM_NEON
    for (; remain > 0; remain--) {
        *ptr++ = _v;
    }
}

template <typename T>
inline void Mat::fill(T _v) {
    int size = total();
    T *ptr = (T *)data;
    for (int i = 0; i < size; i++) {
        ptr[i] = _v;
    }
}

inline Mat Mat::clone() const {
    if (empty()) return Mat();

    Mat m;
    if (dims == 1)
        m.create(w, data_type);
    else if (dims == 2)
        m.create(w, h, data_type);
    else if (dims == 3)
        m.create(w, h, c, data_type);
    else if (dims == 4)
        m.create(n, w, h, c, data_type);

    if (total() > 0) {
        BNN_ASSERT(false, "Clone is not implemented correctly");
        memcpy(m.data, data, total() * elemsize);
    }

    return m;
}

inline void Mat::create(int _w, DataType _data_type) {
    if (dims == 1 && w == _w && data_type == _data_type) return;

    release();

    data_type = _data_type;
    elemsize = data_type == DataType::Float ? sizeof(float) : sizeof(uint64_t);

    dims = 1;
    n = 1;
    w = _w;
    h = 1;
    c = 1;

    hstep = w;

    if (total() > 0) {
        size_t totalsize = ncnn::alignSize(total() * elemsize, 4);
        data = ncnn::fastMalloc(totalsize);
    }
}

inline void Mat::create(int _w, int _h, DataType _data_type) {
    if (dims == 2 && w == _w && h == _h && data_type == _data_type) return;

    release();

    data_type = _data_type;
    elemsize = data_type == DataType::Float ? sizeof(float) : sizeof(uint64_t);

    dims = 2;
    n = 1;
    w = _w;
    h = _h;
    c = 1;

    hstep = w;

    if (total() > 0) {
        size_t totalsize = ncnn::alignSize(total() * elemsize, 4);
        data = ncnn::fastMalloc(totalsize);
    }
}

inline void Mat::create(int _w, int _h, int _c, DataType _data_type) {
    if (dims == 3 && w == _w && h == _h && c == _c && data_type == _data_type)
        return;

    release();

    data_type = _data_type;
    elemsize = data_type == DataType::Float ? sizeof(float) : sizeof(uint64_t);

    dims = 3;
    n = 1;
    w = _w;
    h = _h;
    c = _c;

    if (w * c != 1 && w * c * elemsize % 16 != 0) {
        LOG(FATAL) << "Not align, w: " << w << ", c: " << c
                   << ", elemsize: " << elemsize;
        throw std::invalid_argument("Not align!");
    }
    hstep = ncnn::alignSize(w * c * elemsize, 16) / elemsize;

    if (total() > 0) {
        size_t totalsize = ncnn::alignSize(total() * elemsize, 4);
        data = ncnn::fastMalloc(totalsize);
    }
}

inline void Mat::create(int _n, int _w, int _h, int _c, DataType _data_type,
                        bool require_align) {
    if (dims == 4 && n == _n && w == _w && h == _h && c == _c &&
        data_type == _data_type)
        return;

    release();

    data_type = _data_type;
    elemsize = data_type == DataType::Float ? sizeof(float) : sizeof(uint64_t);

    dims = 0;
    n = _n;
    w = _w;
    h = _h;
    c = _c;
    if (n != 0) dims++;
    if (w != 0) dims++;
    if (h != 0) dims++;
    if (c != 0) dims++;

    if (require_align && w * c != 1 && w * c * elemsize % 16 != 0) {
        LOG(FATAL) << "Not align, w: " << w << ", c: " << c
                   << ", elemsize: " << elemsize;
        throw std::invalid_argument("Not align!");
    }
    if (require_align) {
        hstep = ncnn::alignSize(w * c * elemsize, 16) / elemsize;
    } else {
        hstep = w * c;
    }
    BNN_ASSERT(hstep > 0, hstep);

    if (total() > 0) {
        size_t totalsize = ncnn::alignSize(total() * elemsize, 4);
        data = ncnn::fastMalloc(totalsize);
    }
}

inline void Mat::release() {
    if (!external_memory) {
        ncnn::fastFree(data);
    }

    data = nullptr;

    elemsize = 0;

    dims = 0;
    w = 0;
    h = 0;
    c = 0;

    hstep = 0;

    refcount = 0;
}

inline bool Mat::empty() const { return data == nullptr || total() == 0; }

inline size_t Mat::total() const {
    if (data_num_ != 0) {
        return data_num_;
    } else {
        return n * h * w * c;
    }
}

template <typename T>
inline const T *Mat::point(int _n, int _h, int _w) const {
    BNN_ASSERT((_n == 0 && _h == 0 && _w == 0) || hstep > 0, hstep);
    return (T *)data + _n * h * hstep + _h * hstep + _w * c;
}

template <typename T>
inline const T *Mat::point(int _h, int _w) const {
    BNN_ASSERT((_h == 0 && _w == 0) || hstep > 0, hstep);
    return (T *)data + _h * hstep + _w * c;
}

template <typename T>
inline T *Mat::point(int _n, int _h, int _w) {
    BNN_ASSERT((_n == 0 && _h == 0 && _w == 0) || hstep > 0, hstep);
    return (T *)data + _n * h * hstep + _h * hstep + _w * c;
}

template <typename T>
inline T *Mat::point(int _h, int _w) {
    BNN_ASSERT((_h == 0 && _w == 0) || hstep > 0, hstep);
    return (T *)data + _h * hstep + _w * c;
}

template <typename T>
inline Mat::operator T *() {
    return (T *)data;
}

template <typename T>
inline Mat::operator const T *() const {
    return (const T *)data;
}

inline float &Mat::operator[](int i) { return ((float *)data)[i]; }

inline const float &Mat::operator[](int i) const {
    return ((const float *)data)[i];
}

inline Mat Mat::flatten() { return Mat(total(), data, data_type); }

inline void Mat::dump(css &filename) {
    std::ofstream ofs(filename);
    FORZ(i, total()) { ofs << (*this)[i] << std::endl; }
}

}  // namespace bnn

#endif  // NCNN_MAT_H
