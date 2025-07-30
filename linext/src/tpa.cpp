#include "tpa.hpp"

namespace {

// Allow failure:
//  - With probability TPADeltaSplit * delta in preliminary phase
//  - With probability (1 - TPADeltaSplit) * delta in final phase
const double TPADeltaSplit = 0.01;

ll safePositiveCeil(double x) {
    if(!isfinite(x) || x < 0.0 || x > (double)numeric_limits<ll>::max()) {
        fail("Number representation error when determining the number of TPA walks");
    }
    return max((ll)ceil(x), (ll)1);
}

// Assume X ~ Pois(L) where 0 <= L <= maxL. Returns an upper bound for the
// probability that |X - L| is greater than epsilon.
double poissonDeviationBound(double epsilon, double maxL) {
    assume(epsilon > 0.0);
    assume(maxL >= 0.0);
    assume(isfinite(epsilon));
    assume(isfinite(maxL));
    if(maxL <= 0.0) {
        return 0.0;
    }
    if(epsilon <= 0.5) {
        return 1.0;
    }

    double inBound = boost::math::gamma_q(maxL + epsilon, maxL);
    if(maxL >= epsilon) {
        inBound -= boost::math::gamma_q(maxL - epsilon + 1.0, maxL);
    }
    return 1.0 - inBound;
}

const ll InitialFinalWalkCount = (ll)1 << 62;

}

ll tpaPreliminaryWalkCount(double delta) {
    assume(isfinite(delta) && delta > 0.0);
    
    return safePositiveCeil(2.0 * log(1.0 / (TPADeltaSplit * delta)));
}

ll tpaFinalWalkCount(double epsilon, double delta, double prelimEstimate) {
    assume(isfinite(epsilon) && epsilon > 0.0);
    assume(isfinite(delta) && delta > 0.0);
    if(!isfinite(prelimEstimate) || prelimEstimate < 0.0) {
        fail("Invalid preliminary estimate in TPA");
    }
    
    double maxL = prelimEstimate + sqrt(prelimEstimate) + 2.0;

    double maxFailProb = (1.0 - TPADeltaSplit) * delta;
    auto isEnoughWalks = [&](ll walkCount) {
        double walkCountDbl = (double)walkCount;
        return poissonDeviationBound(walkCountDbl * epsilon, walkCountDbl * maxL) <= maxFailProb;
    };

    ll A = 1;
    ll B = 1;
    while(!isEnoughWalks(B)) {
        if(B > numeric_limits<ll>::max() / 2) {
            fail("Overflow error when determining the number of TPA walks");
        }
        B *= 2;
    }
    while(A != B) {
        ll M = A + (B - A) / 2;
        if(isEnoughWalks(M)) {
            B = M;
        } else {
            A = M + 1;
        }
    }
    return A;
}

ParallelTPAContext::ParallelTPAContext(double epsilon, double delta) {
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

bool ParallelTPAContext::popWalkJob(bool& prelim) {
    ll walkIdx = nextWalkIdx_.fetch_sub(1, memory_order_relaxed);
    if(walkIdx >= 0) {
        prelim = walkIdx >= InitialFinalWalkCount;
        return true;
    } else {
        return false;
    }
}

void ParallelTPAContext::pushWalkResult(bool prelim, ll hitCount) {
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

double ParallelTPAContext::getResult() const {
    assume(finalWalkCount_ >= 1);
#ifndef NDEBUG
    assume(finalResultCount_.load() == finalWalkCount_);
#endif
    msg("PARALLEL_TPA_DONE");
    double finalEstimate = (double)finalHitCount_.load() / (double)finalWalkCount_;
    msg("PARALLEL_TPA_FINAL_ESTIMATE ", finalEstimate);
    return finalEstimate;
}
