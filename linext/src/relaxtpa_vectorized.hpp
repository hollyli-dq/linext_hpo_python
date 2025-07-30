#pragma once

#include "relaxtpa_common.hpp"
#include "tpa.hpp"

namespace relaxtpa_vectorized_ {

template <typename V, typename UpdFuncs>
struct VectorizedRelaxationTPA {
    static const int VecSize = V::N;
    static_assert(VecSize > 0, "Invalid vector size");

    typedef typename V::Bool BoolVec;
    typedef typename V::Int IntVec;
    typedef typename V::PCG32Step PCG32StepVec;
    typedef typename V::PCG32 PCG32Vec;

    int n;
    UpdFuncs updFuncs;
    ParallelTPAContext tpa;
    double relaxationLogCount;

    VectorizedRelaxationTPA(
        int n,
        UpdFuncs updFuncs,
        double epsilon,
        double delta,
        double relaxationLogCount
    )
        : n(n),
          updFuncs(move(updFuncs)),
          tpa(log(1.0 + epsilon), delta),
          relaxationLogCount(relaxationLogCount)
    {}

    void run() {
        msg("SAMPLING_VECTOR_SIZE ", VecSize);

        parallelize([&](int threadIdx) {
            ll totalIterCount = 0;

            unique_ptr<char[]> stateBuf;
            array<IntVec, 2>* state;
            tie(stateBuf, state) = IntVec::allocPairs(n);

            PCG32Vec pcg32 = PCG32Vec::randomSeeded();
            PCG32StepVec initialBackStep =
                pcg32.backwardStep(InitialIterationCount * updFuncs.stepsPerIteration());
            PCG32StepVec zeroStep = PCG32StepVec::zero();
            
            BoolVec active = BoolVec(false);
            BoolVec prelim = BoolVec(false);
            for(int i = 0; i < VecSize; ++i) {
                bool isPrelim;
                active.set(i, tpa.popWalkJob(isPrelim));
                prelim.set(i, isPrelim);
            }

            IntVec beta(MaxCoord);
            IntVec hitCount(0);
            PCG32StepVec backStep = initialBackStep;
            IntVec iterCount(InitialIterationCount);
            IntVec itersLeft(InitialIterationCount);
            pcg32.skip(backStep);
            for(int v = 0; v < n; ++v) {
                state[v][0] = IntVec(0);
                state[v][1] = IntVec(MaxCoord);
            }

            while(active.any()) {
                updFuncs.iteration(state, pcg32, beta);
                itersLeft = itersLeft - IntVec(1);
                totalIterCount += VecSize;

                BoolVec itersDone = active && itersLeft == IntVec(0);
                if(itersDone.any()) {
                    BoolVec isCoupled = itersDone;
                    for(int v = 0; v < n; ++v) {
                        isCoupled = isCoupled && state[v][0] == state[v][1];
                    }
                    BoolVec isNotCoupled = itersDone && !isCoupled;
                    
                    pcg32.skip(PCG32StepVec::choose(isCoupled, backStep, zeroStep));
                    beta = IntVec::choose(isCoupled, IntVec(0), beta);
                    updFuncs.betaUpd(state, isCoupled, beta);

                    BoolVec isBetaZero = beta == IntVec(0);
                    BoolVec isWalkDone = isCoupled && isBetaZero;
                    BoolVec isWalkNotDone = isCoupled && !isBetaZero;

                    if(isWalkDone.any()) {
                        isWalkDone.iterate([&](int i) {
                            tpa.pushWalkResult(prelim.get(i), hitCount.get(i));
                            bool isPrelim;
                            active.set(i, tpa.popWalkJob(isPrelim));
                            prelim.set(i, isPrelim);
                        });

                        beta = IntVec::choose(isWalkDone, IntVec(MaxCoord), beta);
                        hitCount = IntVec::choose(isWalkDone, IntVec(0), hitCount);
                    }

                    hitCount = hitCount + IntVec::choose(isWalkNotDone, IntVec(1), IntVec(0));

                    backStep = PCG32StepVec::choose(isCoupled, initialBackStep, backStep);
                    iterCount = IntVec::choose(isCoupled, IntVec(InitialIterationCount), iterCount);

                    backStep = PCG32StepVec::choose(isNotCoupled, backStep + backStep, backStep);
                    iterCount = iterCount + IntVec::choose(isNotCoupled, iterCount, IntVec(0));

                    itersLeft = IntVec::choose(itersDone, iterCount, itersLeft);
                    pcg32.skip(PCG32StepVec::choose(itersDone, backStep, zeroStep));
                    for(int v = 0; v < n; ++v) {
                        state[v][0] = IntVec::choose(itersDone, IntVec(0), state[v][0]);
                        state[v][1] = IntVec::choose(itersDone, IntVec(MaxCoord), state[v][1]);
                    }
                }
            }

            msg("THREAD_TOTAL_ITER_COUNT ", threadIdx, " ", totalIterCount);
        });

        msg("LINEXT_LOG_COUNT ", relaxationLogCount - tpa.getResult());
    }
};

template <typename V, typename UpdFuncs>
const int VectorizedRelaxationTPA<V, UpdFuncs>::VecSize;

template <typename V>
struct DynamicUpdFuncs {
    typedef typename V::Bool BoolVec;
    typedef typename V::Int IntVec;
    typedef typename V::PCG32 PCG32Vec;

    const RelaxationSamplingContext& relax;

    DynamicUpdFuncs(const RelaxationSamplingContext& relax) : relax(relax) {}

    ll stepsPerIteration() const {
        return max(2 * relax.vertCount() - 1, 0);
    }

    void step(array<IntVec, 2>* __restrict__ state, PCG32Vec& pcg32, IntVec beta, int v) const {
        IntVec p = pcg32.sampleCoord();
        BoolVec ok[2] = {BoolVec(true), BoolVec(true)};
        for(const int* x : relax.hardPred(v)) {
            for(int i = 0; i < 2; ++i) {
                ok[i] = ok[i] && state[*x][i] < p;
            }
        }
        for(const int* x : relax.softPred(v)) {
            for(int i = 0; i < 2; ++i) {
                ok[i] = ok[i] && state[*x][i] - p < beta;
            }
        }
        for(const int* x : relax.hardSucc(v)) {
            for(int i = 0; i < 2; ++i) {
                ok[i] = ok[i] && p < state[*x][i];
            }
        }
        for(const int* x : relax.softSucc(v)) {
            for(int i = 0; i < 2; ++i) {
                ok[i] = ok[i] && p - state[*x][i] < beta;
            }
        }
        for(int i = 0; i < 2; ++i) {
            state[v][i] = IntVec::choose(ok[i], p, state[v][i]);
        }
    }

    void iteration(array<IntVec, 2>* __restrict__ state, PCG32Vec& pcg32, IntVec beta) const {
        int n = relax.vertCount();
        for(int v = 0; v < n; ++v) {
            step(state, pcg32, beta, v);
        }
        for(int v = n - 2; v >= 0; --v) {
            step(state, pcg32, beta, v);
        }
    }

    void betaUpd(const array<IntVec, 2>* state, BoolVec mask, IntVec& beta) const {
        for(int v = 0; v < relax.vertCount(); ++v) {
            for(const int* x : relax.softSucc(v)) {
                beta = IntVec::choose(mask, IntVec::max(beta, state[v][0] - state[*x][0]), beta);
            }
        }
    }
};

struct RunInfo {
    const RelaxationSamplingContext& relax;
    double epsilon;
    double delta;
};

}

template <int W, typename V>
void runVectorizedRelaxationTPA(const Poset<W>& poset, double epsilon, double delta) {
    using namespace relaxtpa_vectorized_;
    RelaxationSamplingContext relax(poset);
    VectorizedRelaxationTPA<V, DynamicUpdFuncs<V>> ctx(
        relax.vertCount(),
        DynamicUpdFuncs<V>(relax),
        epsilon,
        delta,
        relax.relaxationLogCount()
    );
    ctx.run();
}
