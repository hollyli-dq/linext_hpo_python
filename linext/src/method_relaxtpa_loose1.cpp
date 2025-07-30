#include "relaxtpa_basic.hpp"

namespace loose1_ {

ll safePositiveCeil(double x) {
    if(!isfinite(x) || x < 0.0 || x > (double)numeric_limits<ll>::max()) {
        fail("Number representation error when determining the number of TPA walks");
    }
    return max((ll)ceil(x), (ll)1);
}

const ll InitialFinalWalkCount = (ll)1 << 62;

ll tpaPreliminaryWalkCount(double delta) {
    assume(isfinite(delta) && delta > 0.0);
    delta = min(delta, 1.0);
    return safePositiveCeil(2.0 * log(2.0 / delta));
}

ll tpaFinalWalkCount(double epsilon, double delta, double prelimEstimate) {
    assume(isfinite(epsilon) && epsilon > 0.0);
    assume(isfinite(delta) && delta > 0.0);
    if(!isfinite(prelimEstimate) || prelimEstimate < 0.0) {
        fail("Invalid preliminary estimate in TPA");
    }
    
    epsilon = min(epsilon, 2.0 / 3.0);
    return safePositiveCeil(
        2.0 * (prelimEstimate + sqrt(prelimEstimate) + 2.0)
        * log(4.0 / delta)
        / (epsilon * epsilon * (1.0 - epsilon))
    );
}

class ParallelTPAContext {
public:
    ParallelTPAContext(double epsilon, double delta) {
        epsilon_ = epsilon;
        delta_ = delta;
        prelimWalkCount_ = tpaPreliminaryWalkCount(delta_);
        if(prelimWalkCount_ > 65535) {
            fail("Too large preliminary walk count for parallel TPA context");
        }
        finalWalkCount_ = -1;
        prelimResults_ = 0;
        nextWalkIdx_ = InitialFinalWalkCount + prelimWalkCount_ - 1;
        finalHitCount_ = 0;
    #ifndef NDEBUG
        finalResultCount_ = 0;
    #endif

        msg("PARALLEL_TPA_EPSILON ", epsilon_);
        msg("PARALLEL_TPA_DELTA ", delta_);
        msg("PARALLEL_TPA_PRELIM_WALK_COUNT ", prelimWalkCount_);
        msg("PARALLEL_TPA_START");
    }

    // If there are still walks remaining, pops one from the queue and returns
    // true; otherwise, returns false. If a walk is popped, sets prelim to true
    // if it belongs to the preliminary phase and false if it belongs to the
    // final phase.
    bool popWalkJob(bool& prelim) {
        ll walkIdx = nextWalkIdx_.fetch_sub(1, memory_order_relaxed);
        if(walkIdx >= 0) {
            prelim = walkIdx >= InitialFinalWalkCount;
            return true;
        } else {
            return false;
        }
    }

    // Pushes the result (hitCount) of a walk job obtained from popWalk. The
    // parameter prelim should match the value set by popWalk.
    void pushWalkResult(bool prelim, ll hitCount) {
        assume(hitCount >= 0);
        if(prelim) {
            // prelimResults_ stores number of results in top 16 bits
            ll increment = ((ll)1 << 48) | hitCount;
            ll results = prelimResults_.fetch_add(increment, memory_order_relaxed);
            results += increment;

            ll prelimResultCount = results >> 48;
            assume(prelimResultCount <= prelimWalkCount_);

            if(prelimResultCount == prelimWalkCount_) {
                ll prelimHitCount = results & (((ll)1 << 48) - (ll)1);
                double prelimEstimate = (double)prelimHitCount / (double)prelimWalkCount_;

                msg("PARALLEL_TPA_PRELIM_DONE");
                msg("PARALLEL_TPA_PRELIM_ESTIMATE ", prelimEstimate);

                assume(finalWalkCount_ == -1);
                finalWalkCount_ = tpaFinalWalkCount(epsilon_, delta_, prelimEstimate);

                msg("PARALLEL_TPA_FINAL_WALK_COUNT_TARGET ", finalWalkCount_);

                ll decrement = InitialFinalWalkCount - finalWalkCount_;
                ll idx = nextWalkIdx_.fetch_sub(decrement, memory_order_relaxed);
                idx -= decrement;

                if(idx < -1) {
                    finalWalkCount_ += -1 - idx;
                }
                msg("PARALLEL_TPA_FINAL_WALK_COUNT ", finalWalkCount_);
            }
        } else {
            finalHitCount_.fetch_add(hitCount, memory_order_relaxed);
    #ifndef NDEBUG
            finalResultCount_.fetch_add(1, memory_order_relaxed);
    #endif
        }
    }

    double getResult() const {
        assume(finalWalkCount_ >= 1);
    #ifndef NDEBUG
        assume(finalResultCount_.load() == finalWalkCount_);
    #endif
        msg("PARALLEL_TPA_DONE");
        double finalEstimate = (double)finalHitCount_.load() / (double)finalWalkCount_;
        msg("PARALLEL_TPA_FINAL_ESTIMATE ", finalEstimate);
        return finalEstimate;
    }

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

}

template <int W>
void method_relaxtpa_loose1(const Poset<W>& poset, double epsilon, double delta) {
    runBasicRelaxationTPAWithAlternativeParTPA<W, loose1_::ParallelTPAContext>(poset, epsilon, delta);
}

#define FOR_EACH_POSET_W_ELEM(W) \
    template void method_relaxtpa_loose1<W>(const Poset<W>&, double, double);
FOR_EACH_POSET_W
