// Copyright 2019 JD.com Inc. JD AI

#ifndef HELPER_H
#define HELPER_H

#include <bitset>
#include <climits>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>

#include "log_helper.h"

// Make a FOREACH macro
#define FE_1(WHAT, X) WHAT(X)
#define FE_2(WHAT, X, ...) WHAT(X) FE_1(WHAT, __VA_ARGS__)
#define FE_3(WHAT, X, ...) WHAT(X) FE_2(WHAT, __VA_ARGS__)
#define FE_4(WHAT, X, ...) WHAT(X) FE_3(WHAT, __VA_ARGS__)
#define FE_5(WHAT, X, ...) WHAT(X) FE_4(WHAT, __VA_ARGS__)
#define FE_6(WHAT, X, ...) WHAT(X) FE_5(WHAT, __VA_ARGS__)
#define FE_7(WHAT, X, ...) WHAT(X) FE_6(WHAT, __VA_ARGS__)
#define FE_8(WHAT, X, ...) WHAT(X) FE_7(WHAT, __VA_ARGS__)
#define FE_9(WHAT, X, ...) WHAT(X) FE_8(WHAT, __VA_ARGS__)
#define FE_10(WHAT, X, ...) WHAT(X) FE_9(WHAT, __VA_ARGS__)
#define FE_11(WHAT, X, ...) WHAT(X) FE_10(WHAT, __VA_ARGS__)
//... repeat as needed

#define GET_MACRO(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, NAME, ...) NAME
#define FOR_EACH(action, ...)                                              \
    GET_MACRO(__VA_ARGS__, FE_11, FE_10, FE_9, FE_8, FE_7, FE_6, FE_5, FE_4, FE_3, FE_2, FE_1) \
    (action, __VA_ARGS__)

#define M_A_1(_1, ...) _1
#define M_A_2(_1, _2, ...) _2
#define M_A_3(_1, _2, _3, ...) _3
#define M_A_4(_1, _2, _3, _4, ...) _4
#define M_A_5(_1, _2, _3, _4, _5, ...) _5
#define M_A_6(_1, _2, _3, _4, _5, _6, ...) _6
#define M_A_7(_1, _2, _3, _4, _5, _6, _7, ...) _7
#define M_A_8(_1, _2, _3, _4, _5, _6, _7, _8, ...) _8
#define M_A_9(_1, _2, _3, _4, _5, _6, _7, _8, _9, ...) _9
#define M_A_10(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, ...) _10
#define M_A_11(_1, _2, _3, _4, _5, _6, _7, _8, _10, _11, ...) _11

#define FIRST_ARG(...) M_A_1(__VA_ARGS__)

#define LAST_ARG(...)                                                       \
    GET_MACRO(__VA_ARGS__, M_A_11, M_A_10, M_A_9, M_A_8, M_A_7, M_A_6, M_A_5, M_A_4, M_A_3, M_A_2, \
              M_A_1)                                                        \
    (__VA_ARGS__)

#define FORZS(var, end, step) \
    for (auto var = decltype(end){0}; var < end; var += (step))

#define FORZ(var, end) for (auto var = decltype(end){0}; var < end; var++)

#define FOR(var, start, end) \
    for (auto var = decltype(end){start}; var < end; var++)

#define STR(a) #a
#define XSTR(a) STR(a)

#define PNT_STR(s) << s
#define PNT_VAR(var) << XSTR(var) << " = " << (var) << ", "
#define PNT_TO(stream, ...) stream FOR_EACH(PNT_VAR, __VA_ARGS__);
#define PNT(...) PNT_TO(LOG(INFO), __VA_ARGS__)

#define BNN_ASSERT(condition, ...)                \
    if (!(condition)) {                           \
        std::stringstream ss;                     \
        ss << std::string(XSTR(condition))        \
           << std::string(" is not satisfied! ")  \
                  FOR_EACH(PNT_STR, __VA_ARGS__); \
        LOG(INFO) << ss.str();                    \
        throw std::runtime_error(ss.str());       \
    }

inline float random_float() {
    static std::random_device
        rd;  // Get a random seed from the OS entropy device, or whatever
    static std::mt19937_64 eng(
        rd());  // Use the 64-bit Mersenne Twister 19937 generator
    // and seed it with entropy.

    static std::normal_distribution<float> distr;

    float rand_float = distr(eng) / 10;
    if (rand_float == 0) {
        return random_float();
    }
    // LOG(INFO) << "Random float: " << rand_float;

    return rand_float;
}

inline uint64_t random_uint64() {
    static std::random_device
        rd;  // Get a random seed from the OS entropy device, or whatever
    static std::mt19937_64 eng(
        rd());  // Use the 64-bit Mersenne Twister 19937 generator
    // and seed it with entropy.

    // Define the distribution, by default it goes from 0 to MAX(unsigned long
    // long) or what have you.
    static std::uniform_int_distribution<unsigned long long> distr;

    auto rand_uint64 = distr(eng);
    // LOG(INFO) << "Random num: " << rand_uint64;

    return rand_uint64;
}

inline void fill_rand_float(float *data, size_t num) {
    FORZ(i, num) {
        data[i] = random_float();  // (1243.52 - i) * 125.512;     fails
        // data[i] = (1243.52 - i) * 125.512;//     fails for bconv_float
    }
}

inline void fill_rand_uint64(uint64_t *data, size_t num) {
    FORZ(i, num) { *(data + i) = random_uint64(); }
}

/**
 * parameter human will make the output on little endian machines human-readable
 */
inline std::string binrep(const void *a, const size_t size, bool reverse) {
    const char *beg = static_cast<const char *>(a);
    const char *end = beg + size;

    std::stringstream ss;

    if (reverse) {
        while (beg != end) ss << std::bitset<CHAR_BIT>(*(end-- - 1)) << ' ';
    } else {
        while (beg != end) ss << std::bitset<CHAR_BIT>(*beg++) << ' ';
    }
    return ss.str();
}

template <typename T>
T Product(const std::vector<T> &v) {
    return static_cast<T>(
        accumulate(v.begin(), v.end(), 1, std::multiplies<T>()));
}

namespace bnn {
using bin_t = uint64_t;
}

using css = const std::string;
#endif /* HELPER_H */
