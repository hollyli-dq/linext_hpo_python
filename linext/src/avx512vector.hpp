#pragma once

#include "common.hpp"

#include <immintrin.h>

struct AVX512Vector {
    static constexpr int N = 16;

    struct Bool {
        __mmask16 data;

        Bool() {}

        explicit Bool(__mmask16 data)
            : data(data)
        {}

        explicit Bool(bool val)
            : Bool(_cvtu32_mask16((unsigned)(-(int)val)))
        {}

        Bool operator!() const {
            return Bool(_knot_mask16(data));
        }

        Bool operator&&(Bool b) const {
            return Bool(_kand_mask16(data, b.data));
        }

        template <typename F>
        void iterate(F f) const {
            int mask = _cvtmask16_u32(data);
            while(mask) {
                int b = __builtin_ctz(mask);
                mask ^= 1 << b;
                f(b);
            }
        }
        
        bool any() const {
            return (bool)_cvtmask16_u32(data);
        }

        void set(int i, bool val) {
            assume(i >= 0 && i < 16);
            if(val) {
                data = _kor_mask16(data, _cvtu32_mask16(1u << i));
            } else {
                data = _kand_mask16(data, _cvtu32_mask16(~(1u << i)));
            }
        }
        bool get(int i) const {
            assume(i >= 0 && i < 16);
            return (bool)(_cvtmask16_u32(data) & (1 << i));
        }
    };
    struct Int {
        __m512i data;

        Int() {}

        explicit Int(__m512i data)
            : data(data)
        {}

        explicit Int(int val)
            : Int(_mm512_set1_epi32(val))
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
            return Int(_mm512_mask_blend_epi32(pred.data, b.data, a.data));
        }

        Int operator+(Int b) const {
            return Int(_mm512_add_epi32(data, b.data));
        }
        Int operator-(Int b) const {
            return Int(_mm512_sub_epi32(data, b.data));
        }

        static Int max(Int a, Int b) {
            return Int(_mm512_max_epi32(a.data, b.data));
        }
        static Int min(Int a, Int b) {
            return Int(_mm512_min_epi32(a.data, b.data));
        }

        Bool operator==(Int b) const {
            return Bool(_mm512_cmpeq_epi32_mask(data, b.data));
        }
        Bool operator<(Int b) const {
            return Bool(_mm512_cmpgt_epi32_mask(b.data, data));
        }

        int get(int i) const {
            assume(i >= 0 && i < 16);
            return _mm_cvtsi128_si32(
                _mm512_castsi512_si128(
                    _mm512_permutexvar_epi32(
                        _mm512_castsi128_si512(
                            _mm_cvtsi32_si128(i)
                        ),
                        data
                    )
                )
            );
        }
    };
    struct PCG32Step {
        __m512i mulLo;
        __m512i mulHi;
        __m512i addLo;
        __m512i addHi;

        PCG32Step() {}

        static PCG32Step zero() {
            PCG32Step ret;
            ret.mulLo = _mm512_set1_epi64(1);
            ret.mulHi = _mm512_set1_epi64(1);
            ret.addLo = _mm512_setzero_si512();
            ret.addHi = _mm512_setzero_si512();
            return ret;
        }

        PCG32Step operator+(PCG32Step b) const {
            PCG32Step ret;
            ret.mulLo = _mm512_mullo_epi64(mulLo, b.mulLo);
            ret.mulHi = _mm512_mullo_epi64(mulHi, b.mulHi);
            ret.addLo = _mm512_add_epi64(_mm512_mullo_epi64(addLo, b.mulLo), b.addLo);
            ret.addHi = _mm512_add_epi64(_mm512_mullo_epi64(addHi, b.mulHi), b.addHi);
            return ret;
        }

        static PCG32Step choose(Bool pred, PCG32Step a, PCG32Step b) {
            int predInt = _cvtmask16_u32(pred.data);
            int predLoInt = 3 * (predInt & 0x5555);
            int predHiInt = 3 * ((predInt >> 1) & 0x5555);
            __mmask16 predLo = _cvtu32_mask16(predLoInt);
            __mmask16 predHi = _cvtu32_mask16(predHiInt);

            PCG32Step ret;
            ret.mulLo = _mm512_mask_blend_epi32(predLo, b.mulLo, a.mulLo);
            ret.mulHi = _mm512_mask_blend_epi32(predHi, b.mulHi, a.mulHi);
            ret.addLo = _mm512_mask_blend_epi32(predLo, b.addLo, a.addLo);
            ret.addHi = _mm512_mask_blend_epi32(predHi, b.addHi, a.addHi);
            return ret;
        }

    };
    struct PCG32 {
        __m512i addLo;
        __m512i addHi;
        __m512i stateLo;
        __m512i stateHi;

        PCG32() {}

        static PCG32 randomSeeded() {
            alignas(64) array<uint64_t, 8> seqLo;
            alignas(64) array<uint64_t, 8> seqHi;
            for(uint64_t& x : seqLo) {
                x = (UnifInt<uint64_t>()(rng) << 1) | (uint64_t)1; 
            }
            for(uint64_t& x : seqHi) {
                x = (UnifInt<uint64_t>()(rng) << 1) | (uint64_t)1;
            }

            alignas(64) array<uint64_t, 8> seedLo;
            alignas(64) array<uint64_t, 8> seedHi;
            for(uint64_t& x : seedLo) {
                x = UnifInt<uint64_t>()(rng);
            }
            for(uint64_t& x : seedHi) {
                x = UnifInt<uint64_t>()(rng);
            }
            
            PCG32 ret;
            ret.stateLo = _mm512_setzero_si512();
            ret.stateHi = _mm512_setzero_si512();
            ret.addLo = _mm512_load_si512((__m512i*)seqLo.data());
            ret.addHi = _mm512_load_si512((__m512i*)seqHi.data());
            ret.sampleCoord();
            ret.stateLo = _mm512_add_epi64(ret.stateLo, _mm512_load_si512((__m512i*)seedLo.data()));
            ret.stateHi = _mm512_add_epi64(ret.stateHi, _mm512_load_si512((__m512i*)seedHi.data()));
            ret.sampleCoord();
            return ret;
        }

        Int sampleCoord() {
            __m512i xLo = stateLo;
            __m512i xHi = stateHi;
            stateLo = _mm512_add_epi64(_mm512_mullo_epi64(xLo, _mm512_set1_epi64(PCG32_MULTIPLIER)), addLo);
            stateHi = _mm512_add_epi64(_mm512_mullo_epi64(xHi, _mm512_set1_epi64(PCG32_MULTIPLIER)), addHi);
            __m512i rotLo = _mm512_srli_epi64(xLo, 59);
            __m512i rotHi = _mm512_srli_epi64(xHi, 59);
            __m512i rot = _mm512_or_si512(rotLo, _mm512_slli_epi64(rotHi, 32));
            xLo = _mm512_xor_si512(xLo, _mm512_srli_epi64(xLo, 18));
            xHi = _mm512_xor_si512(xHi, _mm512_srli_epi64(xHi, 18));
            __m512i rLo = _mm512_srli_epi64(xLo, 27);
            __m512i rHiShift = _mm512_slli_epi64(xHi, 5);
            __m512i r = _mm512_mask_blend_epi32(_cvtu32_mask16(0xAAAA), rLo, rHiShift);
            __m512i negRot = _mm512_and_si512(
                _mm512_sub_epi32(_mm512_setzero_si512(), rot),
                _mm512_set1_epi32(31)
            );
            __m512i result = _mm512_or_si512(
                _mm512_srlv_epi32(r, rot),
                _mm512_sllv_epi32(r, negRot)
            );
            return Int(_mm512_srli_epi32(result, 1));
        }

        void skip(PCG32Step step) {
            stateLo = _mm512_add_epi64(_mm512_mullo_epi64(step.mulLo, stateLo), step.addLo);
            stateHi = _mm512_add_epi64(_mm512_mullo_epi64(step.mulHi, stateHi), step.addHi);
        }

        PCG32Step backwardStep(uint64_t c) const {
            c = -c;

            PCG32Step ret = PCG32Step::zero();
            PCG32Step x;
            x.mulLo = _mm512_set1_epi64(PCG32_MULTIPLIER);
            x.mulHi = _mm512_set1_epi64(PCG32_MULTIPLIER);
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
