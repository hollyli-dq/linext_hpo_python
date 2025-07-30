#pragma once

#include <condition_variable>
#include <unordered_map>
#include <functional>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <climits>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <limits>
#include <memory>
#include <random>
#include <thread>
#include <vector>
#include <array>
#include <deque>
#include <mutex>
#include <queue>
#include <set>

#include <boost/math/special_functions/gamma.hpp>

using namespace std;

extern thread_local mt19937 rng;

typedef long long ll;
typedef unsigned long long ull;

#ifdef __NVCC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

template <typename T>
using UnifInt = uniform_int_distribution<T>;

template <typename T>
using UnifReal = uniform_real_distribution<T>;

constexpr double Infinity = numeric_limits<double>::infinity();

extern const clock_t StartCPUTime;
extern const chrono::steady_clock::time_point StartWallTime;

extern const int HardwareThreadCount;

extern mutex outputMutex;

inline void stderrPrint() {
    cerr << "\n";
}
template <typename F, typename... T>
void stderrPrint(const F& f, const T&... p) {
    cerr << f;
    stderrPrint(p...);
}
template <typename... T>
void msg(const T&... p) {
    double cpuTime = (double)(clock() - StartCPUTime) / (double)CLOCKS_PER_SEC;
    double wallTime = 1e-9 * (double)chrono::duration_cast<chrono::nanoseconds>(chrono::steady_clock::now() - StartWallTime).count();
    lock_guard<mutex> lock(outputMutex);
    stderrPrint(cpuTime, " ", wallTime, " -- ", p...);
}
template <typename... T>
void fail(const T&... p) {
    msg("FAIL -- ", p...);
    abort();
}

template <typename T>
T fromString(const string& str) {
    T ret;
    stringstream ss(str);
    ss >> ret;
    if(ss.fail()) fail("fromString: Could not convert string '", str, "' to given type (typeid name '", typeid(T).name(), "').");
    return ret;
}

template <typename F>
void parallelize(F f, bool quiet = false) {
    if(!quiet) msg("PARALLELIZE_START");
    vector<thread> threads;
    for(int threadIdx = 0; threadIdx < HardwareThreadCount; ++threadIdx) {
        threads.emplace_back([threadIdx, &f, quiet]() {
            if(!quiet) msg("PARALLELIZE_THREAD_START ", threadIdx);
            f((const int&)threadIdx);
            if(!quiet) msg("PARALLELIZE_THREAD_END ", threadIdx);
        });
    }
    for(thread& t : threads) {
        t.join();
    }
    if(!quiet) msg("PARALLELIZE_END");
}

#ifdef NDEBUG
#define assume(expr) \
    do { \
        if(!(expr)) { \
            __builtin_unreachable(); \
        } \
    } while(false)
#else
#define assume(expr) \
    do { \
        if(!(expr)) { \
            fail("Assumption '", #expr, "' failed in ", __FILE__, ":", __LINE__); \
            __builtin_unreachable(); \
        } \
    } while(false)
#endif

#define CUDACHECK(cmd) \
    do { \
        cudaError_t err = cmd; \
        if(err != cudaSuccess) { \
            fail("CUDA error ", __FILE__, ":", __LINE__, " '", cudaGetErrorString(err)); \
        } \
    } while(false)

template <typename T>
struct Range {
    T a;
    T b;
    
    Range() {}
    Range(T a, T b) : a(a), b(b) {}

    struct Iterator {
        T i;

        Iterator(T i) : i(i) {}
        bool operator==(Iterator o) const {
            return i == o.i;
        }
        bool operator!=(Iterator o) const {
            return i != o.i;
        }
        Iterator operator++() {
            return Iterator(++i);
        }
        Iterator operator++(int) {
            return Iterator(i++);
        }
        T operator*() const {
            return i;
        }
    };

    Iterator begin() const {
        return Iterator(a);
    }
    Iterator end() const {
        return Iterator(b);
    }
};
