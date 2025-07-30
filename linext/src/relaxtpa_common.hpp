#include "relaxation.hpp"
#include "pcg32.hpp"

static const int MaxCoord = (int)(((unsigned int)1 << 31) - 1);
static const int InitialIterationCount = 16;

inline int sampleCoord(PCG32& pcg32) {
    return (int)(pcg32() >> 1);
}

class RelaxationSamplingContext {
public:
    template <int W>
    RelaxationSamplingContext(const Poset<W>& poset) {
        n_ = poset.size();

        pool_.resize(n_ * n_ + 1);
        poolPos_ = 0;
        hardPred_.resize(n_);
        hardSucc_.resize(n_);
        softPred_.resize(n_);
        softSucc_.resize(n_);

        Poset<W> relaxation;
        msg("FIND_RELAXATION_START");
        tie(relaxation, relaxationLogCount_) = findRelaxation(poset);
        msg("FIND_RELAXATION_END");
        msg("RELAXATION_LINEXT_LOG_COUNT ", relaxationLogCount_);
        showPosetInfo(relaxation, "RELAXATION");

        int hardConstraintCount = 0;
        int softConstraintCount = 0;
        for(int v = 0; v < n_; ++v) {
            hardPred_[v] = addToPool_(relaxation.predCover(v));
            hardSucc_[v] = addToPool_(relaxation.succCover(v));
            softPred_[v] = addToPool_(difference(poset.predCover(v), relaxation.pred(v)));
            softSucc_[v] = addToPool_(difference(poset.succCover(v), relaxation.succ(v)));
            hardConstraintCount += hardSucc_[v].b - hardSucc_[v].a;
            softConstraintCount += softSucc_[v].b - softSucc_[v].a;
        }
        msg("SAMPLING_HARD_CONSTRAINT_COUNT ", hardConstraintCount);
        msg("SAMPLING_SOFT_CONSTRAINT_COUNT ", softConstraintCount);
    }
    RelaxationSamplingContext(const RelaxationSamplingContext&) = delete;
    RelaxationSamplingContext(RelaxationSamplingContext&&) = delete;
    RelaxationSamplingContext& operator=(const RelaxationSamplingContext&) = delete;
    RelaxationSamplingContext& operator=(RelaxationSamplingContext&&) = delete;
    ~RelaxationSamplingContext() {}

    int vertCount() const {
        return n_;
    }

    Range<const int*> hardPred(int v) const {
        assume(v >= 0 && v < n_);
        return hardPred_[v];
    }
    Range<const int*> hardSucc(int v) const {
        assume(v >= 0 && v < n_);
        return hardSucc_[v];
    }
    Range<const int*> softPred(int v) const {
        assume(v >= 0 && v < n_);
        return softPred_[v];
    }
    Range<const int*> softSucc(int v) const {
        assume(v >= 0 && v < n_);
        return softSucc_[v];
    }
    Range<const int*> neighbors(int v) const {
        assume(v >= 0 && v < n_);
        return Range<const int*>(hardPred_[v].a, softSucc_[v].b);
    }

    double relaxationLogCount() const {
        return relaxationLogCount_;
    }

private:
    template <int W>
    Range<const int*> addToPool_(Bitset<W> verts) {
        Range<const int*> ret;
        ret.a = &pool_[poolPos_];
        verts.iterate([&](int v) {
            assume(poolPos_ < (int)pool_.size());
            pool_[poolPos_++] = v;
        });
        ret.b = &pool_[poolPos_];
        return ret;
    }

    int n_;
    vector<int> pool_;
    int poolPos_;
    vector<Range<const int*>> hardPred_;
    vector<Range<const int*>> hardSucc_;
    vector<Range<const int*>> softPred_;
    vector<Range<const int*>> softSucc_;
    double relaxationLogCount_;
};
