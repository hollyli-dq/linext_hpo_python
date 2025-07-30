#include "poset.hpp"
#include "swap_linext_order_sampler.hpp"
#include "gibbs_linext_order_sampler.hpp"

namespace {

template <int W>
struct Factor {
    Poset<W> poset;
    Bitset<W> verts;
    int a;
    int b;
};

template <int W, typename Sampler>
double estimateOrderProbsUsingSampler(
    const vector<Factor<W>>& factors, ll samplesPerFactor, bool quiet
) {
    if(factors.empty()) {
        return 0.0;
    }

    assume(samplesPerFactor >= 1);
    ll totalSampleCount = factors.size() * samplesPerFactor;
    std::atomic<ll> nextSampleIdx(0);
    vector<std::atomic<ll>> positiveSampleCounts(factors.size());

    parallelize([&](int) {
        int factorIdx = 0;
        Sampler sampler(
            factors[factorIdx].poset,
            factors[factorIdx].verts,
            factors[factorIdx].a,
            factors[factorIdx].b
        );
        while(true) {
            ll sampleIdx = nextSampleIdx.fetch_add(1, memory_order_relaxed);
            if(sampleIdx >= totalSampleCount) {
                break;
            }
            int newFactorIdx = sampleIdx / samplesPerFactor;
            if(newFactorIdx != factorIdx) {
                factorIdx = newFactorIdx;
                sampler = Sampler(
                    factors[factorIdx].poset,
                    factors[factorIdx].verts,
                    factors[factorIdx].a,
                    factors[factorIdx].b
                );
            }
            if(sampler.sample()) {
                positiveSampleCounts[factorIdx].fetch_add(1, memory_order_relaxed);
            }
        }
    }, quiet);

    double ret = 0.0;
    for(std::atomic<ll>& positiveCount : positiveSampleCounts) {
        ret += log((double)positiveCount.load() / (double)samplesPerFactor);
    }
    return ret;
}

template <int W>
double estimateOrderProbsSwap(
    const vector<Factor<W>>& factors, ll samplesPerFactor, bool quiet
) {
    return estimateOrderProbsUsingSampler<W, SwapLinextOrderSampler<W>>(
        factors, samplesPerFactor, quiet
        );
}

template <int W>
double estimateOrderProbsGibbs(
    const vector<Factor<W>>& factors, ll samplesPerFactor, bool quiet
) {
    return estimateOrderProbsUsingSampler<W, GibbsLinextOrderSampler<W>>(
        factors, samplesPerFactor, quiet
    );
}

static const int PreliminarySampleCount = 300;

template <int W>
pair<vector<Factor<W>>, double> factorizeBasic(
    const Poset<W>& poset,
    double (*estimateOrderProbs)(const vector<Factor<W>>&, ll, bool)
) {
    constexpr double logHalf = -0.6931471805599453; //log(0.5);

    struct Context {
        Poset<W> poset;
        double (*estimateOrderProbs)(const vector<Factor<W>>&, ll, bool);
        vector<Factor<W>> estimateFactor;
        vector<Factor<W>> factors;

        bool order(int a, int b) {
            if(poset.has(a, b)) {
                return true;
            }
            if(poset.has(b, a)) {
                return false;
            }
            estimateFactor[0].a = a;
            estimateFactor[0].b = b;
            double logProb = estimateOrderProbs(estimateFactor, PreliminarySampleCount, true);
            bool ret = true;
            if(logProb < logHalf) {
                swap(a, b);
                ret = false;
            }
            factors.push_back({poset, poset.allVerts(), a, b});
            poset.add(a, b);
            return ret;
        }
        void mergeSort(vector<int>& verts) {
            if(verts.size() <= 1) {
                return;
            }

            vector<int> A(verts.begin(), verts.begin() + verts.size() / 2);
            mergeSort(A);
            vector<int> B(verts.begin() + verts.size() / 2, verts.end());
            mergeSort(B);

            verts.clear();
            int a = 0;
            int b = 0;
            while(a != (int)A.size() && b != (int)B.size()) {
                if(order(A[a], B[b])) {
                    verts.push_back(A[a++]);
                } else {
                    verts.push_back(B[b++]);
                }
            }
            while(a != (int)A.size()) {
                verts.push_back(A[a++]);
            }
            while(b != (int)B.size()) {
                verts.push_back(B[b++]);
            }
        }
    };
    
    Factor<W> factor = {poset, poset.allVerts(), -1, -1};
    Context ctx = {poset, estimateOrderProbs, {factor}};

    vector<int> verts(poset.size());
    for(int v = 0; v < poset.size(); ++v) {
        verts[v] = v;
    }
    ctx.mergeSort(verts);

    return {move(ctx.factors), 0.0};
}

template <int W>
pair<vector<Factor<W>>, double> factorizeDecomposition(
    const Poset<W>& poset,
    double (*estimateOrderProbs)(const vector<Factor<W>>&, ll, bool)
) {
    constexpr double logHalf = -0.6931471805599453; //log(0.5);

    struct Context {
        Poset<W> poset;
        double (*estimateOrderProbs)(const vector<Factor<W>>&, ll, bool);
        vector<Factor<W>> estimateFactor;
        double logCoef;
        vector<Factor<W>> factors;

        bool order(Bitset<W> verts, int a, int b) {
            if(poset.has(a, b)) {
                return true;
            }
            if(poset.has(b, a)) {
                return false;
            }
            estimateFactor[0].verts = verts;
            estimateFactor[0].a = a;
            estimateFactor[0].b = b;
            double logProb = estimateOrderProbs(estimateFactor, PreliminarySampleCount, true);
            bool ret = true;
            if(logProb < logHalf) {
                swap(a, b);
                ret = false;
            }
            factors.push_back({poset, verts, a, b});
            poset.add(a, b);
            return ret;
        }
        void sort(Bitset<W> verts, int pivot) {
            logCoef += lgamma(verts.count() + 1);
            poset.components(verts, [&](Bitset<W> comp) {
                sortComponent(comp, pivot);
                logCoef -= lgamma(comp.count() + 1);
            });
        }
        bool trySplit(Bitset<W> verts, int pivot) {
            Bitset<W> A = Bitset<W>::empty();
            Bitset<W> B = verts;
            int bestVal = -1;
            Bitset<W> bestA = Bitset<W>::empty();
            Bitset<W> bestB = Bitset<W>::empty();
            poset.topoSortWhile(verts, [&](int v) {
                A.add(v);
                B = intersection(B, poset.succ(v));
                if(!B.isEmpty() && verts == unio(A, B)) {
                    int val = min(A.count(), B.count());
                    if(val > bestVal) {
                        bestVal = val;
                        bestA = A;
                        bestB = B;
                    }
                }
                return true;
            });
            if(bestVal != -1) {
                sort(bestA, pivot);
                sort(bestB, pivot);
                return true;
            }
            return false;
        }
        void sortComponent(Bitset<W> verts, int pivot) {
            if(verts.count() <= 1) {
                return;
            }
            if(trySplit(verts, pivot)) {
                return;
            }

            if(pivot == -1 || !verts[pivot]) {
                int bestVal = -1;
                verts.iterate([&](int v) {
                    int pred = intersection(poset.pred(v), verts).count();
                    int succ = intersection(poset.succ(v), verts).count();
                    int val = min(pred, succ);
                    if(val > bestVal) {
                        bestVal = val;
                        pivot = v;
                    }
                });
            }

            Bitset<W> cand = verts;
            cand.del(pivot);
            cand = difference(cand, poset.pred(pivot));
            cand = difference(cand, poset.succ(pivot));
            int v = cand.random();
            order(verts, pivot, v);
            sortComponent(verts, pivot);
        }
    };
    
    Factor<W> factor = {poset, poset.allVerts(), -1, -1};
    Context ctx = {poset, estimateOrderProbs, {factor}, 0.0};

    ctx.sort(poset.allVerts(), -1);

    return {move(ctx.factors), ctx.logCoef};
}

ll computeSamplesPerFactor(
    int factorCount,
    double epsilon,
    double delta
) {
    assume(isfinite(epsilon) && epsilon > 0.0);
    assume(isfinite(delta) && delta > 0.0);

    epsilon = epsilon / (1.0 + epsilon);

    double k = factorCount;

    auto failureProb2 = [&](double samples, double minProb) {
        double probDiff = 0.5 - minProb;
        return
            (pow(1.0 + (1.0 - minProb) / (minProb * samples), k) - 1.0) / (epsilon * epsilon) +
            2.0 * k * exp(-2.0 * probDiff * probDiff * (double)PreliminarySampleCount);
    };

    auto failureProb = [&](double samples) {
        double A = 0.0;
        double B = 0.5;
        double V = 1.0 / 0.0;
        while(B - A > 1e-4) {
            double M1 = (2.0 * A + B) / 3.0;
            double M2 = (A + 2.0 * B) / 3.0;
            double V1 = failureProb2(samples, M1);
            double V2 = failureProb2(samples, M2);
            V = min(V, V1);
            V = min(V, V2);
            if(V1 < V2) {
                B = M2;
            } else {
                A = M1;
            }
        }
        return V;
    };

    double A = 0.0;
    double B = 1.0;
    while(failureProb(B) > delta) {
        B *= 2.0;
    }

    while(B - A >= 1.0) {
        double M = 0.5 * (A + B);
        if(failureProb(M) > delta) {
            A = M;
        } else {
            B = M;
        }
    }

    return max((ll)ceil(B), (ll)1);
}

template <int W>
void runTelescope(
    const Poset<W>& poset,
    double epsilon,
    double delta,
    pair<vector<Factor<W>>, double> (*factorize)(
        const Poset<W>&, double (*)(const vector<Factor<W>>&, ll, bool)
    ),
    double (*estimateOrderProbs)(const vector<Factor<W>>&, ll, bool)
) {
    msg("PRELIMINARY_SAMPLE_COUNT ", PreliminarySampleCount);

    msg("FACTORIZATION_START");
    vector<Factor<W>> factors;
    double logCoef;
    tie(factors, logCoef) = factorize(poset, estimateOrderProbs);
    msg("FACTORIZATION_END");

    msg("FACTOR_COUNT ", factors.size());

    msg("SAMPLES_PER_FACTOR_COMPUTE_START");
    ll samplesPerFactor = computeSamplesPerFactor(factors.size(), epsilon, delta);
    msg("SAMPLES_PER_FACTOR_COMPUTE_END");

    msg("SAMPLES_PER_FACTOR ", samplesPerFactor);

    msg("SAMPLE_ESTIMATION_START");
    double logProb = estimateOrderProbs(factors, samplesPerFactor, false);
    msg("SAMPLE_ESTIMATION_END");

    msg("LINEXT_LOG_COUNT ", logCoef - logProb);
}

}

template <int W>
void method_telescope_basic_swap(const Poset<W>& poset, double epsilon, double delta) {
    runTelescope(poset, epsilon, delta, factorizeBasic, estimateOrderProbsSwap);
}

template <int W>
void method_telescope_basic_gibbs(const Poset<W>& poset, double epsilon, double delta) {
    runTelescope(poset, epsilon, delta, factorizeBasic, estimateOrderProbsGibbs);
}

template <int W>
void method_telescope_decomposition_gibbs(const Poset<W>& poset, double epsilon, double delta) {
    runTelescope(poset, epsilon, delta, factorizeDecomposition, estimateOrderProbsGibbs);
}

#define FOR_EACH_POSET_W_ELEM(W) \
    template void method_telescope_basic_swap<W>(const Poset<W>&, double, double); \
    template void method_telescope_basic_gibbs<W>(const Poset<W>&, double, double); \
    template void method_telescope_decomposition_gibbs<W>(const Poset<W>&, double, double);
FOR_EACH_POSET_W
