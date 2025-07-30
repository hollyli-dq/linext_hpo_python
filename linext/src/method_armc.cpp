#include "exactcount.hpp"

namespace {

const int ARMCInitK = 20;
const int ARMCKappa = 5;
const int ARMCInvAlpha = 10;
const double ARMCBeta = 10;
const double ARMCMemUsageRatioLimit = 0.25;

struct StoppingRuleEstimatorConf {
    ll positivesRequired;
    double numerator;

    StoppingRuleEstimatorConf(double epsilon, double delta) {
        assume(isfinite(epsilon) && epsilon > 0.0);
        assume(isfinite(delta) && delta > 0.0);

        epsilon = 1.0 - 1.0 / (1.0 + epsilon);
        delta = min(delta, 1.0);

        double lambda = exp(1.0) - 2.0;
        double Y = 4.0 * lambda * log(2.0 / delta) / (epsilon * epsilon);
        double Y1 = 1.0 + (1.0 + epsilon) * Y;
        double pr = ceil(Y1);
        if(pr > (double)LLONG_MAX) {
            fail("Could not configure stopping rule estimator, as the number of required positives would be too large");
        }
        positivesRequired = (ll)pr;
        numerator = Y1;

        msg("STOPPING_RULE_POSITIVES_REQUIRED ", positivesRequired);
        msg("STOPPING_RULE_NUMERATOR ", numerator);
    }
};

template <int W>
struct BestImprov {
    int improv;
    Bitset<W> newComp1;
    Bitset<W> newComp2;
};

template <int W>
BestImprov<W> computeBestImprov(const Poset<W>& poset, Bitset<W> comp1, Bitset<W> comp2) {
    BestImprov<W> ret = {0, comp1, comp2};

    int bestImprov[2] = {INT_MIN / 2, INT_MIN / 2};
    Bitset<W> opts[2][3];

    auto registerImprov = [&](int p, int v, int improv) {
        if(improv > bestImprov[p]) {
            int idx = bestImprov[p] - improv + 2;
            opts[p][2] = idx >= 0 ? opts[p][idx] : Bitset<W>::empty();
            --idx;
            opts[p][1] = idx >= 0 ? opts[p][idx] : Bitset<W>::empty();
            opts[p][0] = Bitset<W>::empty();
            opts[p][0].add(v);
            bestImprov[p] = improv;
        } else {
            int d = bestImprov[p] - improv;
            if(d < 3) {
                opts[p][d].add(v);
            }
        }
    };

    comp1.iterate([&](int vert1) {
        int improv =
            intersection(unio(poset.pred(vert1), poset.succ(vert1)), comp2).count()
            - intersection(unio(poset.pred(vert1), poset.succ(vert1)), comp1).count();
        registerImprov(0, vert1, improv);
    });

    comp2.iterate([&](int vert2) {
        int improv =
            intersection(unio(poset.pred(vert2), poset.succ(vert2)), comp1).count()
            - intersection(unio(poset.pred(vert2), poset.succ(vert2)), comp2).count();
        registerImprov(1, vert2, improv);
    });

    int baseImprov = bestImprov[0] + bestImprov[1];
    for(int d1 = 0; d1 < 3; ++d1) {
        if(opts[0][d1].isEmpty()) {
            continue;
        }
        for(int d2 = 0; d2 < 3; ++d2) {
            if(opts[1][d2].isEmpty()) {
                continue;
            }
            int improv = baseImprov - d1 - d2;

            int vert1, vert2;
            if(opts[0][d1].iterateWhile([&](int v) {
                Bitset<W> indeps = difference(opts[1][d2], unio(poset.pred(v), poset.succ(v)));
                if(indeps.isEmpty()) {
                    return true;
                }
                vert1 = v;
                vert2 = indeps.min();
                return false;
            })) {
                vert1 = opts[0][d1].min();
                vert2 = opts[1][d2].min();
                improv -= 2;
            }

            if(improv > ret.improv) {
                ret.improv = improv;
                ret.newComp1 = comp1;
                ret.newComp1.del(vert1);
                ret.newComp1.add(vert2);
                ret.newComp2 = comp2;
                ret.newComp2.del(vert2);
                ret.newComp2.add(vert1);
            }
        }
    }

    return ret;
}

template <int W>
void optimizeComps(
    const Poset<W>& poset,
    int k,
    vector<Bitset<W>>& comps,
    int& consCount,
    int& coverConsCount
) {
    vector<int> ord(poset.size());
    for(int i = 0; i < poset.size(); ++i) {
        ord[i] = i;
    }
    shuffle(ord.begin(), ord.end(), rng);

    comps.clear();
    for(int s = 0; s < poset.size(); s += k) {
        Bitset<W> comp = Bitset<W>::empty();
        for(int i = s; i < s + k && i < poset.size(); ++i) {
            int v = ord[i];
            comp.add(v);
        }
        comps.push_back(comp);
    }

    vector<vector<BestImprov<W>>> bestImprovs(comps.size());
    for(vector<BestImprov<W>>& x : bestImprovs) {
        x.resize(comps.size());
    }

    auto bestImprovCmp = [&](pair<int, int> a, pair<int, int> b) {
        int A = bestImprovs[a.first][a.second].improv;
        int B = bestImprovs[b.first][b.second].improv;
        if(A == B) {
            return a < b;
        } else {
            return A > B;
        }
    };
    set<pair<int, int>, decltype(bestImprovCmp)> bestImprovOrd(bestImprovCmp);

    for(int compIdx1 = 0; compIdx1 < (int)comps.size(); ++compIdx1) {
        Bitset<W> comp1 = comps[compIdx1];
        for(int compIdx2 = compIdx1 + 1; compIdx2 < (int)comps.size(); ++compIdx2) {
            Bitset<W> comp2 = comps[compIdx2];
            bestImprovs[compIdx1][compIdx2] = computeBestImprov(poset, comp1, comp2);
            bestImprovOrd.emplace(compIdx1, compIdx2);
        }
    }

    while(!bestImprovOrd.empty()) {
        int compIdx1 = bestImprovOrd.begin()->first;
        int compIdx2 = bestImprovOrd.begin()->second;
        BestImprov<W> bestImprov = bestImprovs[compIdx1][compIdx2];

        if(bestImprov.improv <= 0) {
            break;
        }

        comps[compIdx1] = bestImprov.newComp1;
        comps[compIdx2] = bestImprov.newComp2;

        auto recompute = [&](int a, int b) {
            bestImprovOrd.erase(make_pair(a, b));
            bestImprovs[a][b] = computeBestImprov(poset, comps[a], comps[b]);
            bestImprovOrd.emplace(a, b);
        };

        recompute(compIdx1, compIdx2);
        for(int i = 0; i < compIdx1; ++i) {
            recompute(i, compIdx1);
            recompute(i, compIdx2);
        }
        for(int i = compIdx1 + 1; i < compIdx2; ++i) {
            recompute(compIdx1, i);
            recompute(i, compIdx2);
        }
        for(int i = compIdx2 + 1; i < (int)comps.size(); ++i) {
            recompute(compIdx1, i);
            recompute(compIdx2, i);
        }
    }

    consCount = 0;
    coverConsCount = 0;
    for(Bitset<W> comp : comps) {
        comp.iterate([&](int v) {
            consCount += intersection(poset.succ(v), comp).count();
            coverConsCount += intersection(poset.succCover(v), comp).count();
        });
    }
}

template <int W>
class ARMCEstimator {
public:
    ARMCEstimator(const Poset<W>& poset, int k, ExactCounterGlobalMemoryLimit& memLimit)
        : poset_(poset),
          k_(k)
    {
        assume(k_ >= 1 && k_ <= poset_.size());
        memLimit.checkOutOfMemory();

        msg("ATTEMPT_K ", k);

        msg("FIND_RELAXATION_START");

        atomic<int> triesLeft;
        triesLeft = 50;

        mutex compsMut;
        int consCount = -1;
        int coverConsCount = -1;

        parallelize([&](int) {
            while(true) {
                if(triesLeft.fetch_sub(1) <= 0) {
                    break;
                }
                vector<Bitset<W>> compsCand;
                int consCountCand;
                int coverConsCountCand;
                optimizeComps(poset_, k_, compsCand, consCountCand, coverConsCountCand);

                lock_guard<mutex> lock(compsMut);
                if(consCountCand > consCount) {
                    comps_ = move(compsCand);
                    consCount = consCountCand;
                    coverConsCount = coverConsCountCand;
                }
            }
        });
        assume(consCount != -1);

        msg("FIND_RELAXATION_END");

        msg("RELAXATION_CONSTRAINT_COUNT ", consCount);
        msg("RELAXATION_COVER_CONSTRAINT_COUNT ", coverConsCount);

        compSamplerGens_.resize(comps_.size());
        atomic<int> nextCompIdx;
        nextCompIdx = 0;

        relaxLogCount_ = lgamma((double)poset.size() + 1.0);
        mutex relaxLogCountMut;

        msg("RELAXATION_EXACT_COUNT_START");
        auto exactCountStartTime = chrono::steady_clock::now();
        parallelize([&](int threadIdx) {
            try {
                while(true) {
                    int compIdx = nextCompIdx.fetch_add(1);
                    if(compIdx >= (int)comps_.size()) {
                        break;
                    }
                    Bitset<W> comp = comps_[compIdx];
                    msg("THREAD_COMP_START ", threadIdx, " ", comp.count());

                    Poset<W> compPoset(comp.count());
                    int ai = 0;
                    comp.iterate([&](int a) {
                        int bi = 0;
                        comp.iterate([&](int b) {
                            if(poset.has(a, b)) {
                                compPoset.add(ai, bi);
                            }

                            ++bi;
                        });
                        ++ai;
                    });
                    auto result = createExactLinextSampler(compPoset, memLimit);
                    if(!isfinite(result.first)) {
                        fail("Overflow in exact linear extension counting");
                    }
                    {
                        lock_guard<mutex> lock(relaxLogCountMut);
                        relaxLogCount_ += result.first - lgamma((double)comp.count() + 1.0);
                    }
                    compSamplerGens_[compIdx] = result.second;

                    msg("THREAD_COMP_END ", threadIdx, " ", comp.count());
                }
            } catch(ExactCounterGlobalMemoryLimit::OutOfMemory) {}
        });
        memLimit.checkOutOfMemory();
        exactCountTime_ = 1e-9 * (double)chrono::duration_cast<chrono::nanoseconds>(chrono::steady_clock::now() - exactCountStartTime).count();

        msg("RELAXATION_EXACT_COUNT_END");
        msg("RELAXATION_LINEXT_LOG_COUNT ", relaxLogCount_);

        for(const Bitset<W>& comp : comps_) {
            compSize_.push_back(comp.count());
            vector<int> translation;
            comp.iterate([&](int v) {
                translation.push_back(v);
            });
            compVertTranslation_.push_back(move(translation));
        }

        correctionTermEstimationTimeGuess_ = -1.0;
    }

    int k() {
        return k_;
    }

    double relaxLogCount() {
        return relaxLogCount_;
    }

    double exactCountTime() {
        return exactCountTime_;
    }

    double guessCorrectionTermEstimationTime(StoppingRuleEstimatorConf conf) {
        if(correctionTermEstimationTimeGuess_ >= 0.0) {
            msg("GUESS_CORRECTION_TERM_ESTIMATION_TIME_USE_CACHED");
            return correctionTermEstimationTimeGuess_;
        }

        ll positivesRequired = (conf.positivesRequired + ARMCInvAlpha - 1) / ARMCInvAlpha;
        atomic<ll> positivesSeen;
        positivesSeen = 0;

        double timeLimit = exactCountTime_ / (double)ARMCInvAlpha;

        msg("GUESS_CORRECTION_TERM_ESTIMATION_TIME_START");
        auto startTime = chrono::steady_clock::now();
        parallelize([&](int) {
            vector<LinextSampler> compSamplers(comps_.size());
            while(true) {
                for(int i = 0; i < 64; ++i) {
                    if(drawSample_(compSamplers)) {
                        positivesSeen.fetch_add(1, memory_order_relaxed);
                    }
                }
                if(positivesSeen.load(memory_order_relaxed) >= positivesRequired) {
                    break;
                }
                double timeElapsed = 1e-9 * (double)chrono::duration_cast<chrono::nanoseconds>(chrono::steady_clock::now() - startTime).count();
                if(timeElapsed > timeLimit) {
                    break;
                }
            }
        });
        msg("GUESS_CORRECTION_TERM_ESTIMATION_TIME_END");
        double timeElapsed = 1e-9 * (double)chrono::duration_cast<chrono::nanoseconds>(chrono::steady_clock::now() - startTime).count();
        ll ps = positivesSeen.load();
        double result;
        if(ps == 0) {
            result = Infinity;
        } else {
            result = timeElapsed * (double)conf.positivesRequired / (double)ps;
        }
        correctionTermEstimationTimeGuess_ = result;
        return result;
    }

    double estimateCorrectionTerm(StoppingRuleEstimatorConf conf) {
        atomic<ll> nextSampleIdx;
        nextSampleIdx = 1;

        vector<ll> positiveSampleIdxs;
        mutex positiveSampleIdxsMut;

        auto startTime = chrono::steady_clock::now();
        parallelize([&](int) {
            vector<LinextSampler> compSamplers(comps_.size());
            while(true) {
                ll sampleIdx = nextSampleIdx.fetch_add(1, memory_order_relaxed);
                if(sampleIdx < 0) {
                    break;
                }
                if(drawSample_(compSamplers)) {
                    lock_guard<mutex> lock(positiveSampleIdxsMut);
                    positiveSampleIdxs.push_back(sampleIdx);
                    if((ll)positiveSampleIdxs.size() == conf.positivesRequired) {
                        double timeElapsed = 1e-9 * (double)chrono::duration_cast<chrono::nanoseconds>(chrono::steady_clock::now() - startTime).count();
                        double samplesPerSec = (double)nextSampleIdx.load() / timeElapsed;
                        msg("SAMPLES_PER_SEC_APPROX ", samplesPerSec);
                        nextSampleIdx = LLONG_MIN;
                    }
                }
            }
        });

        sort(positiveSampleIdxs.begin(), positiveSampleIdxs.end());
        return log(conf.numerator / (double)positiveSampleIdxs[conf.positivesRequired - 1]);
    }

private:
    // compSamplers should have size at least comps_.size(), and it does not
    // need to be initialized
    bool drawSample_(vector<LinextSampler>& compSamplers) {
        for(int compIdx = 0; compIdx < (int)comps_.size(); ++compIdx) {
            compSamplerGens_[compIdx](compSamplers[compIdx]);
        }

        int compVertsLeft[Bitset<W>::BitCount];
        copy(compSize_.begin(), compSize_.end(), compVertsLeft);

        Bitset<W> vertsSeen = Bitset<W>::empty();
        Bitset<W> vertsNotSeen = poset_.allVerts();

        for(int i = poset_.size() - 1; i >= 0; --i) {
            int param = UnifInt<int>(0, i)(rng);
            int compIdx = 0;
            while(true) {
                param -= compVertsLeft[compIdx];
                if(param < 0) {
                    break;
                }
                ++compIdx;
            }
            --compVertsLeft[compIdx];
            int v = compVertTranslation_[compIdx][compSamplers[compIdx]()];
            if(!isSubset(poset_.pred(v), vertsSeen) || !isSubset(poset_.succ(v), vertsNotSeen)) {
                return false;
            }
            vertsSeen.add(v);
            vertsNotSeen.del(v);
        }
        return true;
    }

    const Poset<W>& poset_;
    int k_;
    vector<Bitset<W>> comps_;
    vector<int> compSize_;
    vector<vector<int>> compVertTranslation_;
    vector<function<void(LinextSampler&)>> compSamplerGens_;
    double relaxLogCount_;
    double exactCountTime_;
    double correctionTermEstimationTimeGuess_;
};

}

template <int W>
void method_armc(const Poset<W>& poset, double epsilon, double delta) {
    ExactCounterGlobalMemoryLimit memLimit((size_t)30 * 1024 * 1024 * 1024);
    StoppingRuleEstimatorConf stopRuleConf(epsilon, delta);
    
    // Emulating optional<>
    vector<ARMCEstimator<W>> bestARMC;

    int k = ARMCInitK;
    while(true) {
        k = min(k, poset.size());

        try {
            ARMCEstimator<W> armc(poset, k, memLimit);
            if(bestARMC.empty() || armc.relaxLogCount() < bestARMC[0].relaxLogCount()) {
                msg("NEW_BEST_K ", k);
                bestARMC.clear();
                bestARMC.push_back(move(armc));
            }

            double memUsageRatio = memLimit.memoryUsageRatio();
            msg("MEM_USAGE_RATIO ", memUsageRatio);
            if(memUsageRatio > ARMCMemUsageRatioLimit) {
                msg("MEM_USAGE_RATIO_TOO_HIGH");
                break;
            }
            
            double nextExactCountGuess = ARMCBeta * armc.exactCountTime();
            msg("NEXT_K_EXACT_COUNT_TIME_GUESS ", nextExactCountGuess);

            double sampleGuess = bestARMC[0].guessCorrectionTermEstimationTime(stopRuleConf);
            msg("CORRECTION_TERM_ESTIMATION_TIME_GUESS ", sampleGuess);

            if(nextExactCountGuess > sampleGuess) {
                msg("NEXT_K_EXACT_COUNT_TIME_GUESS_TOO_HIGH");
                break;
            }
        } catch(ExactCounterGlobalMemoryLimit::OutOfMemory) {
            msg("RELAXATION_EXACT_COUNT_OUT_OF_MEMORY");
            break;
        }

        if(k == poset.size()) {
            break;
        }

        k += ARMCKappa;
    }

    if(bestARMC.empty()) {
        msg("NO_SUCCESSFUL_ARMC_K_FOUND");
        fail("Could not construct ARMC relaxation sampler for any parameter k");
    }

    ARMCEstimator<W> armc = move(bestARMC[0]);

    msg("USING_ARMC_K ", armc.k());
    msg("RELAXATION_LINEXT_LOG_COUNT ", armc.relaxLogCount());

    msg("CORRECTION_TERM_ESTIMATION_START");
    double correctionTerm = armc.estimateCorrectionTerm(stopRuleConf);
    msg("CORRECTION_TERM_ESTIMATION_END");
    msg("CORRECTION_TERM ", correctionTerm);

    msg("LINEXT_LOG_COUNT ", armc.relaxLogCount() + correctionTerm);
}

#define FOR_EACH_POSET_W_ELEM(W) \
    template void method_armc<W>(const Poset<W>&, double, double);
FOR_EACH_POSET_W
