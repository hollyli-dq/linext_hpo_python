#pragma once

#include "relaxtpa_common.hpp"
#include "tpa.hpp"

namespace relaxtpa_basic_ {

template <typename ParTPA>
struct Context {
    int n;
    RelaxationSamplingContext relax;
    ParTPA tpa;

    typedef vector<array<int, 2>> State;

    template <int W>
    Context(const Poset<W>& poset, double epsilon, double delta)
        : n(poset.size()),
          relax(poset),
          tpa(log(1.0 + epsilon), delta)
    {}

    void step(State& state, PCG32& pcg32, int beta, int v) {
        int p = sampleCoord(pcg32);
        for(int i = 0; i < 2; ++i) {
            if(p < state[v][i]) {
                for(const int* x : relax.hardPred(v)) {
                    if(p <= state[*x][i]) goto skip;
                }
                for(const int* x : relax.softPred(v)) {
                    if(state[*x][i] - p >= beta) goto skip;
                }
            } else {
                for(const int* x : relax.hardSucc(v)) {
                    if(p >= state[*x][i]) goto skip;
                }
                for(const int* x : relax.softSucc(v)) {
                    if(p - state[*x][i] >= beta) goto skip;
                }
            }
            state[v][i] = p;
            skip: {}
        }
    }

    void iteration(State& state, PCG32& pcg32, int beta) {
        for(int v = 0; v < n; ++v) {
            step(state, pcg32, beta, v);
        }
        for(int v = n - 2; v >= 0; --v) {
            step(state, pcg32, beta, v);
        }
    }

    void sample(
        State& state,
        PCG32& pcg32,
        PCG32Step initialBackStep,
        int beta,
        ll& totalIterCount
    ) {
        PCG32Step backStep = initialBackStep;
        int iterCount = InitialIterationCount;
        while(true) {
            pcg32.skip(backStep);
            fill(state.begin(), state.end(), array<int, 2>{0, MaxCoord});

            for(int iterIdx = 0; iterIdx < iterCount; ++iterIdx) {
                iteration(state, pcg32, beta);
            }
            totalIterCount += iterCount;

            bool isCoupled = true;
            for(int v = 0; v < n; ++v) {
                if(state[v][0] != state[v][1]) {
                    isCoupled = false;
                    break;
                }
            }
            if(isCoupled) {
                break;
            }
            backStep = backStep + backStep;
            iterCount += iterCount;
        }
        pcg32.skip(backStep);
    }

    bool handleWalkJob(
        State& state,
        PCG32& pcg32,
        PCG32Step initialBackStep,
        ll& totalIterCount
    ) {
        bool prelim;
        if(!tpa.popWalkJob(prelim)) {
            return false;
        }
        int beta = MaxCoord;
        int hitCount = 0;
        while(true) {
            sample(state, pcg32, initialBackStep, beta, totalIterCount);

            beta = 0;
            for(int v = 0; v < n; ++v) {
                for(const int* x : relax.softSucc(v)) {
                    beta = max(beta, state[v][0] - state[*x][0]);
                }
            }
            if(beta) {
                ++hitCount;
            } else {
                break;
            }
        }
        tpa.pushWalkResult(prelim, hitCount);
        return true;
    }

    void run() {
        parallelize([&](int threadIdx) {
            ll totalIterCount = 0;

            State state(n);
            PCG32 pcg32;
            ll stepsPerIteration = max(2 * n - 1, 0);
            PCG32Step initialBackStep =
                pcg32.backwardStep(InitialIterationCount * stepsPerIteration);
            while(handleWalkJob(state, pcg32, initialBackStep, totalIterCount)) {}

            msg("THREAD_TOTAL_ITER_COUNT ", threadIdx, " ", totalIterCount);
        });

        msg("LINEXT_LOG_COUNT ", relax.relaxationLogCount() - tpa.getResult());
    }
};

}

template <int W>
void runBasicRelaxationTPA(const Poset<W>& poset, double epsilon, double delta) {
    using namespace relaxtpa_basic_;
    Context<ParallelTPAContext> ctx(poset, epsilon, delta);
    ctx.run();
}

template <int W, typename ParTPA>
void runBasicRelaxationTPAWithAlternativeParTPA(const Poset<W>& poset, double epsilon, double delta) {
    using namespace relaxtpa_basic_;
    Context<ParTPA> ctx(poset, epsilon, delta);
    ctx.run();
}
