
#pragma once

#include "common.hpp"

#include <immintrin.h>

inline __m256i avx2Mul64(__m256i a, __m256i b) {
    __m256i cross = _mm256_mullo_epi32(_mm256_shuffle_epi32(a, 0xB1), b);
    __m256i ret = _mm256_add_epi64(
        _mm256_mul_epu32(a, b),
        _mm256_and_si256(
            _mm256_add_epi32(
                _mm256_shuffle_epi32(cross, 0xB1),
                cross
            ),
            _mm256_set_epi32(-1, 0, -1, 0, -1, 0, -1, 0)
        )
    );
    return ret;
};

struct AVX2Vector {
    static constexpr int N = 8;

    struct Bool {
        __m256i data;

        Bool() {}

        explicit Bool(__m256i data)
            : data(data)
        {}

        explicit Bool(bool val)
            : Bool(_mm256_set1_epi32(-(int)val))
        {}

        Bool operator!() const {
            return Bool(_mm256_xor_si256(data, _mm256_set1_epi32(-1)));
        }

        Bool operator&&(Bool b) const {
            return Bool(_mm256_and_si256(data, b.data));
        }

        template <typename F>
        void iterate(F f) const {
            int mask = _mm256_movemask_epi8(data) & 0x11111111;
            while(mask) {
                int b = __builtin_ctz(mask);
                mask ^= 1 << b;
                f(b >> 2);
            }
        }
        
        bool any() const {
            return (bool)_mm256_movemask_epi8(data);
        }

        void set(int i, bool val) {
            assume(i >= 0 && i < 8);
            data = _mm256_blendv_epi8(
                data,
                _mm256_set1_epi32(-(int)val),
                _mm256_cmpeq_epi32(
                    _mm256_set1_epi32(i),
                    _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0)
                )
            );
        }
        bool get(int i) const {
            assume(i >= 0 && i < 8);
            return (bool)(_mm256_movemask_epi8(data) & (1 << (i << 2)));
        }
    };
    struct Int {
        __m256i data;

        Int() {}

        explicit Int(__m256i data)
            : data(data)
        {}

        explicit Int(int val)
            : Int(_mm256_set1_epi32(val))
        {}

        static pair<unique_ptr<char[]>, array<Int, 2>*> allocPairs(int size) {
            static const size_t Alignment = 64;

            size_t dataSize = (size_t)size * sizeof(array<Int, 2>);
            size_t bufSize = dataSize + Alignment - 1;

            unique_ptr<char[]> buf(new char[bufSize]);
            void* ptr = (void*)buf.get();
            void* aligned = align(Alignment, dataSize, ptr, bufSize);
            assume(aligned != nullptr);

            return make_pair(move(buf), (array<Int, 2>*)aligned);
        }

        static Int choose(Bool pred, Int a, Int b) {
            return Int(_mm256_blendv_epi8(b.data, a.data, pred.data));
        }

        Int operator+(Int b) const {
            return Int(_mm256_add_epi32(data, b.data));
        }
        Int operator-(Int b) const {
            return Int(_mm256_sub_epi32(data, b.data));
        }

        static Int max(Int a, Int b) {
            return Int(_mm256_max_epi32(a.data, b.data));
        }
        static Int min(Int a, Int b) {
            return Int(_mm256_min_epi32(a.data, b.data));
        }

        Bool operator==(Int b) const {
            return Bool(_mm256_cmpeq_epi32(data, b.data));
        }
        Bool operator<(Int b) const {
            return Bool(_mm256_cmpgt_epi32(b.data, data));
        }

        int get(int i) const {
            assume(i >= 0 && i < 8);
            return _mm_cvtsi128_si32(
                _mm256_castsi256_si128(
                    _mm256_permutevar8x32_epi32(
                        data,
                        _mm256_castsi128_si256(
                            _mm_cvtsi32_si128(i)
                        )
                    )
                )
            );
        }
    };
    struct PCG32Step {
        __m256i mulLo;
        __m256i mulHi;
        __m256i addLo;
        __m256i addHi;

        PCG32Step() {}

        static PCG32Step zero() {
            PCG32Step ret;
            ret.mulLo = _mm256_set1_epi64x(1);
            ret.mulHi = _mm256_set1_epi64x(1);
            ret.addLo = _mm256_setzero_si256();
            ret.addHi = _mm256_setzero_si256();
            return ret;
        }

        PCG32Step operator+(PCG32Step b) const {
            PCG32Step ret;
            ret.mulLo = avx2Mul64(mulLo, b.mulLo);
            ret.mulHi = avx2Mul64(mulHi, b.mulHi);
            ret.addLo = _mm256_add_epi64(avx2Mul64(addLo, b.mulLo), b.addLo);
            ret.addHi = _mm256_add_epi64(avx2Mul64(addHi, b.mulHi), b.addHi);
            return ret;
        }

        static PCG32Step choose(Bool pred, PCG32Step a, PCG32Step b) {
            __m256i predLo = _mm256_shuffle_epi32(pred.data, 0xA0);
            __m256i predHi = _mm256_shuffle_epi32(pred.data, 0xF5);
            PCG32Step ret;
            ret.mulLo = _mm256_blendv_epi8(b.mulLo, a.mulLo, predLo);
            ret.mulHi = _mm256_blendv_epi8(b.mulHi, a.mulHi, predHi);
            ret.addLo = _mm256_blendv_epi8(b.addLo, a.addLo, predLo);
            ret.addHi = _mm256_blendv_epi8(b.addHi, a.addHi, predHi);
            return ret;
        }

    };
    struct PCG32 {
        __m256i addLo;
        __m256i addHi;
        __m256i stateLo;
        __m256i stateHi;

        PCG32() {}

        static PCG32 randomSeeded() {
            alignas(32) array<uint64_t, 4> seqLo;
            alignas(32) array<uint64_t, 4> seqHi;
            for(uint64_t& x : seqLo) {
                x = (UnifInt<uint64_t>()(rng) << 1) | (uint64_t)1; 
            }
            for(uint64_t& x : seqHi) {
                x = (UnifInt<uint64_t>()(rng) << 1) | (uint64_t)1;
            }

            alignas(32) array<uint64_t, 4> seedLo;
            alignas(32) array<uint64_t, 4> seedHi;
            for(uint64_t& x : seedLo) {
                x = UnifInt<uint64_t>()(rng);
            }
            for(uint64_t& x : seedHi) {
                x = UnifInt<uint64_t>()(rng);
            }
            
            PCG32 ret;
            ret.stateLo = _mm256_setzero_si256();
            ret.stateHi = _mm256_setzero_si256();
            ret.addLo = _mm256_load_si256((__m256i*)seqLo.data());
            ret.addHi = _mm256_load_si256((__m256i*)seqHi.data());
            ret.sampleCoord();
            ret.stateLo = _mm256_add_epi64(ret.stateLo, _mm256_load_si256((__m256i*)seedLo.data()));
            ret.stateHi = _mm256_add_epi64(ret.stateHi, _mm256_load_si256((__m256i*)seedHi.data()));
            ret.sampleCoord();
            return ret;
        }

        Int sampleCoord() {
            __m256i xLo = stateLo;
            __m256i xHi = stateHi;
            stateLo = _mm256_add_epi64(avx2Mul64(xLo, _mm256_set1_epi64x(PCG32_MULTIPLIER)), addLo);
            stateHi = _mm256_add_epi64(avx2Mul64(xHi, _mm256_set1_epi64x(PCG32_MULTIPLIER)), addHi);
            __m256i rotLo = _mm256_srli_epi64(xLo, 59);
            __m256i rotHi = _mm256_srli_epi64(xHi, 59);
            __m256i rot = _mm256_or_si256(rotLo, _mm256_slli_epi64(rotHi, 32));
            xLo = _mm256_xor_si256(xLo, _mm256_srli_epi64(xLo, 18));
            xHi = _mm256_xor_si256(xHi, _mm256_srli_epi64(xHi, 18));
            __m256i rLo = _mm256_srli_epi64(xLo, 27);
            __m256i rHiShift = _mm256_slli_epi64(xHi, 5);
            __m256i r = _mm256_blendv_epi8(rLo, rHiShift, _mm256_set_epi32(-1, 0, -1, 0, -1, 0, -1, 0));
            __m256i negRot = _mm256_and_si256(
                _mm256_sub_epi32(_mm256_setzero_si256(), rot),
                _mm256_set1_epi32(31)
            );
            __m256i result = _mm256_or_si256(
                _mm256_srlv_epi32(r, rot),
                _mm256_sllv_epi32(r, negRot)
            );
            return Int(_mm256_srli_epi32(result, 1));
        }

        void skip(PCG32Step step) {
            stateLo = _mm256_add_epi64(avx2Mul64(step.mulLo, stateLo), step.addLo);
            stateHi = _mm256_add_epi64(avx2Mul64(step.mulHi, stateHi), step.addHi);
        }

        PCG32Step backwardStep(uint64_t c) const {
            c = -c;

            PCG32Step ret = PCG32Step::zero();
            PCG32Step x;
            x.mulLo = _mm256_set1_epi64x(PCG32_MULTIPLIER);
            x.mulHi = _mm256_set1_epi64x(PCG32_MULTIPLIER);
            x.addLo = addLo;
            x.addHi = addHi;

            while(c) {
                if(c & (uint64_t)1) {
                    ret = ret + x;
                }
                x = x + x;
                c >>= 1;
            }
            return ret;
        }
    };
};
