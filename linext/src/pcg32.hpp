#pragma once

#include "common.hpp"

const uint64_t PCG32_MULTIPLIER = 6364136223846793005ULL;

class PCG32Step {
public:
    CUDA_HOSTDEV PCG32Step() : mul_(1), add_(0) {}

    CUDA_HOSTDEV PCG32Step operator+(PCG32Step other) const {
        return PCG32Step(
            mul_ * other.mul_,
            add_ * other.mul_ + other.add_
        );
    }

private:
    CUDA_HOSTDEV PCG32Step(uint64_t mul, uint64_t add)
        : mul_(mul),
          add_(add)
    {}

    uint64_t mul_;
    uint64_t add_;

    friend class PCG32;
};

class PCG32 {
public:
    PCG32() : PCG32(UnifInt<uint64_t>()(rng), UnifInt<uint64_t>()(rng)) { }
    
    CUDA_HOSTDEV PCG32(uint64_t seed, uint64_t seq) {
        state_ = 0;
        add_ = (seq << 1) | 1;
        (*this)();
        state_ += seed;
        (*this)();
    }

    CUDA_HOSTDEV PCG32Step forwardStep() const {
        return PCG32Step(PCG32_MULTIPLIER, add_);
    }
    CUDA_HOSTDEV PCG32Step backwardStep() const {
        PCG32Step ret;
        PCG32Step x = forwardStep();
        for(int i = 0; i < 64; ++i) {
            ret = ret + x;
            x = x + x;
        }
        return ret;
    }
    CUDA_HOSTDEV PCG32Step forwardStep(uint64_t c) const {
        PCG32Step ret;
        PCG32Step x = forwardStep();
        while(c) {
            if(c & (uint64_t)1) {
                ret = ret + x;
            }
            x = x + x;
            c >>= 1;
        }
        return ret;
    }
    CUDA_HOSTDEV PCG32Step backwardStep(uint64_t c) const {
        return forwardStep(-c);
    }

    CUDA_HOSTDEV void skip(PCG32Step x) {
        state_ = x.mul_ * state_ + x.add_;
    }

    CUDA_HOSTDEV uint32_t operator()() {
        uint64_t x = state_;
        state_ = x * PCG32_MULTIPLIER + add_;
        uint32_t rot = (uint32_t)(x >> 59);
        x ^= x >> 18;
        uint32_t r = (uint32_t)(x >> 27);
        return (r >> rot) | (r << ((-rot) & 31));
    }

#ifdef __NVCC__
    __device__ uint32_t genGPU() {
        uint64_t x = state_;
        state_ = x * PCG32_MULTIPLIER + add_;
        uint32_t rot = (uint32_t)(x >> 59);
        x ^= x >> 18;
        uint32_t r = (uint32_t)(x >> 27);
        return __funnelshift_r(r, r, rot);
    }
#endif

private:
    uint64_t add_;
    uint64_t state_;
};
