#pragma once

#include "common.hpp"

// Number of independent TPA random walks needed in the preliminary and final
// phases of the algorithm to get an approximation with absolute error at most
// epsilon with probability at least 1 - delta.
ll tpaPreliminaryWalkCount(double delta);
ll tpaFinalWalkCount(double epsilon, double delta, double prelimEstimate);

// Context that allows running TPA in multiple threads. Each thread should pop
// jobs using popWalk and return their results using pushWalkResult. After all
// the jobs have finished and a memory fence has been passed, the result can be
// obtained by getResult.
class ParallelTPAContext {
public:
    ParallelTPAContext(double epsilon, double delta);

    // If there are still walks remaining, pops one from the queue and returns
    // true; otherwise, returns false. If a walk is popped, sets prelim to true
    // if it belongs to the preliminary phase and false if it belongs to the
    // final phase.
    bool popWalkJob(bool& prelim);

    // Pushes the result (hitCount) of a walk job obtained from popWalk. The
    // parameter prelim should match the value set by popWalk.
    void pushWalkResult(bool prelim, ll hitCount);

    double getResult() const;

private:
    double epsilon_;
    double delta_;
    ll prelimWalkCount_;
    ll finalWalkCount_;
    atomic<ll> prelimResults_;
    atomic<ll> nextWalkIdx_;
    atomic<ll> finalHitCount_;
#ifndef NDEBUG
    atomic<ll> finalResultCount_;
#endif
};
