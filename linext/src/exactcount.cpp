#include "exactcount.hpp"

struct ExactCounterGlobalMemoryLimit::AccessKey {};

ExactCounterGlobalMemoryLimit::ExactCounterGlobalMemoryLimit(size_t memLimit) {
    msg("EXACT_COUNTER_GLOBAL_MEMORY_LIMIT ", memLimit);

    memLimit_ = memLimit;
    memInUse_ = 0;
    outOfMemory_ = false;
}

ExactCounterGlobalMemoryLimit::~ExactCounterGlobalMemoryLimit() {
    if(!outOfMemory_.load() && memInUse_.load() != 0) {
        fail("ExactCounterGlobalMemoryLimit destroyed while some memory remains in use");
    }
}

double ExactCounterGlobalMemoryLimit::memoryUsageRatio() const {
    return (double)memInUse_.load() / (double)memLimit_;
}

bool ExactCounterGlobalMemoryLimit::isOutOfMemory() const {
    return outOfMemory_.load();
}
void ExactCounterGlobalMemoryLimit::checkOutOfMemory() const {
    if(outOfMemory_.load(memory_order_relaxed)) {
        throw OutOfMemory();
    }
}

void ExactCounterGlobalMemoryLimit::takeMem_(AccessKey, size_t size) {
    size_t memInUse = memInUse_.fetch_add(size);
    memInUse += size;

    if(memInUse > memLimit_) {
        outOfMemory_ = true;
        throw OutOfMemory();
    }
}
void ExactCounterGlobalMemoryLimit::returnMem_(AccessKey, size_t size) {
    size_t oldMemInUse = memInUse_.fetch_sub(size);
    if(oldMemInUse < size) {
        fail("Tried to return more memory than has been allocated in total");
    }
}

struct LinextSampler::AccessKey {};

void* LinextSampler::accessData_(AccessKey) {
    return (void*)data_;
}
LinextSampler::Func& LinextSampler::accessFunc_(AccessKey) {
    return func_;
}

namespace {

template <int W, bool Sampling>
struct HashTableBucket {
    Bitset<W> key;
    double value;
};
template <int W>
struct HashTableBucket<W, true> {
    Bitset<W> key;
    double value;
    bool disconnected;
    union {
        pair<const HashTableBucket<W, true>*, const HashTableBucket<W, true>*> components;
        Bitset<W> minimals;
    };
};

template <int W, bool Sampling>
class BigHashTable {
public:
    typedef HashTableBucket<W, Sampling> Bucket;

    BigHashTable(ExactCounterGlobalMemoryLimit& memLimit) : BigHashTable(memLimit, 20) {}

    class HashedKey {
    public:
        HashedKey(Bitset<W> key) {
            assume(!key.isEmpty());
            key_ = key;
            hash_ = 0;
            for(int w = 0; w < W; ++w) {
                hash_ ^= key.words[w];
                hash_ = (hash_ ^ (hash_ >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
                hash_ = (hash_ ^ (hash_ >> 27)) * UINT64_C(0x94d049bb133111eb);
            }
            hash_ = hash_ ^ (hash_ >> 31);
        }

        Bitset<W> key() const {
            return key_;
        }

        uint64_t hash() const {
            return hash_;
        }
    
    private:
        Bitset<W> key_;
        uint64_t hash_;
    };

    bool get(HashedKey hkey, double& value) const {
        size_t bucketIdx = (size_t)hkey.hash() & hashMask_;
        size_t increment = 1;
        while(true) {
            const Bucket& bucket = table_[bucketIdx];
            if(bucket.key == hkey.key()) {
                value = bucket.value;
                return true;
            }
            if(bucket.key.isEmpty()) {
                return false;
            }

            bucketIdx += increment;
            ++increment;
            bucketIdx &= hashMask_;
        }
    }
    bool getBucket(HashedKey hkey, const Bucket*& outBucket) const {
        size_t bucketIdx = (size_t)hkey.hash() & hashMask_;
        size_t increment = 1;
        while(true) {
            const Bucket& bucket = table_[bucketIdx];
            if(bucket.key == hkey.key()) {
                outBucket = &bucket;
                return true;
            }
            if(bucket.key.isEmpty()) {
                return false;
            }

            bucketIdx += increment;
            ++increment;
            bucketIdx &= hashMask_;
        }
    }

    // Element with given key must not exist already
    void add(HashedKey hkey, double value) {
        if(elemCount_ >= elemCountLimit_) {
            BigHashTable newHashTable(*memLimit_, hashBits_ + 1);
            for(size_t bucketIdx = 0; bucketIdx <= hashMask_; ++bucketIdx) {
                const Bucket& bucket = table_[bucketIdx];
                if(!bucket.key.isEmpty()) {
                    newHashTable.add(bucket.key, bucket.value);
                }
            }
            *this = move(newHashTable);
        }
        ++elemCount_;

        size_t bucketIdx = (size_t)hkey.hash() & hashMask_;
        size_t increment = 1;
        while(true) {
            Bucket& bucket = table_[bucketIdx];
            assume(bucket.key != hkey.key());
            if(bucket.key.isEmpty()) {
                bucket.key = hkey.key();
                bucket.value = value;
                return;
            }

            bucketIdx += increment;
            ++increment;
            bucketIdx &= hashMask_;
        }
    }

    ExactCounterGlobalMemoryLimit& memLimit() {
        return *memLimit_;
    }

private:
    BigHashTable(ExactCounterGlobalMemoryLimit& memLimit, int hashBits) {
        memLimit_ = &memLimit;

        assume(hashBits >= 0 && hashBits < 64);
        hashBits_ = hashBits;
        size_t bucketCount = (size_t)1 << hashBits_;
        hashMask_ = bucketCount - 1;
        elemCountLimit_ = (2 * bucketCount + 2) / 3;

        size_t bufSize = bucketCount * sizeof(Bucket);
        memLimit.takeMem_(ExactCounterGlobalMemoryLimit::AccessKey(), bufSize);
        Bucket* tableBuf = (Bucket*)calloc(bufSize, 1);
        if(tableBuf == nullptr) {
            fail("Allocating BigHashTable failed");
        }
        table_.reset(tableBuf);
        TableDeleter& deleter = table_.get_deleter();
        deleter.memLimit = &memLimit;
        deleter.bufSize = bufSize;
        elemCount_ = 0;
    }

    struct TableDeleter {
        ExactCounterGlobalMemoryLimit* memLimit;
        size_t bufSize;

        TableDeleter() : memLimit(nullptr) {}

        void operator()(Bucket* table) {
            if(memLimit != nullptr) {
                memLimit->returnMem_(ExactCounterGlobalMemoryLimit::AccessKey(), bufSize);
                free(table);
            }
        }
    };

    ExactCounterGlobalMemoryLimit* memLimit_;
    int hashBits_;
    size_t hashMask_;
    size_t elemCountLimit_;
    unique_ptr<Bucket[], TableDeleter> table_;
    size_t elemCount_;

    template <int W2>
    friend void prepareMemForSampling(const Poset<W2>& poset, BigHashTable<W2, true>& mem);
};

template <int W>
Bitset<W> posetComponent(const Poset<W>& poset, Bitset<W> verts, int v) {
    assume(isSubset(verts, poset.allVerts()));
    assume(verts[v]);

    Bitset<W> seen = Bitset<W>::empty();
    seen.add(v);
    Bitset<W> comp = Bitset<W>::empty();
    while(true) {
        Bitset<W> d = difference(intersection(seen, verts), comp);
        if(d.isEmpty()) {
            break;
        }
        d.iterate([&](int x) {
            comp.add(x);
            seen = unio(seen, unio(poset.pred(x), poset.succ(x)));
        });
    }

    return comp;
}

array<array<double, 128>, 128> initBinomialTable() {
    array<array<double, 128>, 128> ret;
    for(int n = 0; n < 128; ++n) {
        ret[n][0] = 1.0;
        for(int k = 1; k < n; ++k) {
            ret[n][k] = ret[n - 1][k - 1] + ret[n - 1][k];
        }
        ret[n][n] = 1.0;
    }
    return ret;
}
const array<array<double, 128>, 128> binomialTable = initBinomialTable();

double computeBinomial(int n, int k) {
    assume(n >= 0);
    assume(k >= 0 && k <= n);
    if(n < 128) {
        return binomialTable[n][k];
    } else {
        return exp(lgamma((double)n + 1.0) - lgamma((double)k + 1.0) - lgamma((double)(n - k) + 1.0));
    }
}

template <int W, bool Sampling>
double countLinext_(const Poset<W>& poset, BigHashTable<W, Sampling>& mem, Bitset<W> verts);

template <int W, bool Sampling>
double countLinextInComp_(
    const Poset<W>& poset,
    BigHashTable<W, Sampling>& mem,
    Bitset<W> verts,
    bool checkMem
) {
    if(verts.hasAtMost1Ones()) {
        return 1.0;
    }

    typename BigHashTable<W, Sampling>::HashedKey key(verts);

    double count;
    if(checkMem && mem.get(key, count)) {
        return count;
    }

    mem.memLimit().checkOutOfMemory();

    count = 0.0;
    poset.minimals(verts).iterate([&](int v) {
        Bitset<W> subVerts = verts;
        subVerts.del(v);
        count += countLinext_(poset, mem, subVerts);
    });

    mem.add(key, count);
    return count;
}

template <int W, bool Sampling>
double countLinext_(const Poset<W>& poset, BigHashTable<W, Sampling>& mem, Bitset<W> verts) {
    if(verts.hasAtMost1Ones()) {
        return 1.0;
    }

    typename BigHashTable<W, Sampling>::HashedKey key(verts);

    double count;
    if(mem.get(key, count)) {
        return count;
    }

    mem.memLimit().checkOutOfMemory();

    Bitset<W> baseComp = posetComponent(poset, verts, verts.min());
    if(baseComp == verts) {
        return countLinextInComp_(poset, mem, verts, false);
    }    

    count = countLinextInComp_(poset, mem, baseComp, true);
    count *= computeBinomial(verts.count(), baseComp.count());
    count *= countLinext_(poset, mem, difference(verts, baseComp));

    mem.add(key, count);
    return count;
}

template <int W>
void prepareMemForSampling(const Poset<W>& poset, BigHashTable<W, true>& mem) {
    // We need singleton buckets for some component pointers
    for(int v = 0; v < poset.size(); ++v) {
        Bitset<W> singleton = Bitset<W>::empty();
        singleton.add(v);
        mem.add(singleton, 1.0);
    }

    for(size_t bucketIdx = 0; bucketIdx <= mem.hashMask_; ++bucketIdx) {
        HashTableBucket<W, true>& bucket = mem.table_[bucketIdx];
        Bitset<W> verts = bucket.key;
        if(!verts.isEmpty()) {
            Bitset<W> comp = posetComponent(poset, verts, verts.min());
            if(comp == verts) {
                bucket.disconnected = false;
                bucket.minimals = poset.minimals(verts);
            } else {
                bucket.disconnected = true;
                bool res;
                res = mem.getBucket(comp, bucket.components.first);
                assume(res);
                res = mem.getBucket(difference(verts, comp), bucket.components.second);
                assume(res);
                if(bucket.components.first->key.count() < bucket.components.second->key.count()) {
                    // Larger component first so that in sampling bigger
                    // components stay closer to the beginning
                    swap(bucket.components.first, bucket.components.second);
                }
            }
        }
    }
}

template <int W>
struct LinextSamplerImpl {
    const BigHashTable<W, true>* mem;
    int vertsLeft;
    int compCount;
    int compVertCount[Bitset<W>::BitCount];
    const HashTableBucket<W, true>* compPtr[Bitset<W>::BitCount];

    int operator()() {
        assume(vertsLeft > 0);
        int param = UnifInt<int>(0, --vertsLeft)(rng);
        int compIdx = 0;
        while(true) {
            if(param < compVertCount[compIdx]) {
                break;
            }
            param -= compVertCount[compIdx];
            ++compIdx;
        }

        while(compPtr[compIdx]->disconnected) {
            int compIdx2 = compCount++;
            compPtr[compIdx2] = compPtr[compIdx]->components.second;
            compPtr[compIdx] = compPtr[compIdx]->components.first;
            int size2 = compPtr[compIdx2]->key.count();
            compVertCount[compIdx] -= size2;
            compVertCount[compIdx2] = size2;
            if(param < size2) {
                compIdx = compIdx2;
            } else {
                param -= size2;
            }
        }

        const HashTableBucket<W, true>& bucket = *compPtr[compIdx];
        if(bucket.key.hasAtMost1Ones()) {
            --compCount;
            compVertCount[compIdx] = compVertCount[compCount];
            compPtr[compIdx] = compPtr[compCount];
            return bucket.key.min();
        }

        double param2 = UnifReal<double>(0.0, bucket.value)(rng);
        int v = -1;
        const HashTableBucket<W, true>* vCompPtr = nullptr;
        bucket.minimals.iterateWhile([&](int x) {
            v = x;
            Bitset<W> rest = bucket.key;
            rest.del(x);
            bool res = mem->getBucket(rest, vCompPtr);
            assume(res);
            param2 -= vCompPtr->value;
            return param2 >= 0.0;
        });
        assume(v != -1);
        assume(vCompPtr != nullptr);

        compPtr[compIdx] = vCompPtr;
        --compVertCount[compIdx];

        return v;
    }
};

template <int W, bool Sampling>
struct CreateSamplerGen {
    static function<void(LinextSampler&)> run(const Poset<W>& poset, BigHashTable<W, Sampling>& mem, double count) {
        return function<void(LinextSampler&)>();
    }
};
template <int W>
struct CreateSamplerGen<W, true> {
    static function<void(LinextSampler&)> run(const Poset<W>& poset, BigHashTable<W, true>& mem, double count) {
        if(isfinite(count)) {
            prepareMemForSampling(poset, mem);

            int n = poset.size();
            const HashTableBucket<W, true>* root;
            bool res = mem.getBucket(poset.allVerts(), root);
            assume(res);

            shared_ptr<BigHashTable<W, true>> sharedMem = make_shared<BigHashTable<W, true>>(move(mem));

            return [sharedMem, n, root](LinextSampler& res) {
                static_assert(LinextSampler::DataBytes >= sizeof(LinextSamplerImpl<W>), "LinextSampler::DataBytes is set too low");

                LinextSampler::AccessKey key;
                
                LinextSamplerImpl<W>& impl = *(LinextSamplerImpl<W>*)res.accessData_(key);
                impl.mem = sharedMem.get();
                impl.vertsLeft = n;
                impl.compCount = 1;
                impl.compVertCount[0] = n;
                impl.compPtr[0] = root;

                res.accessFunc_(key) = [](void* ptr) mutable -> int {
                    LinextSamplerImpl<W>& impl = *(LinextSamplerImpl<W>*)ptr;
                    return impl();
                };
            };
        } else {
            return [](LinextSampler&) {
                fail("Linear extension sampler generator called even though linear extension counting overflowed");
            };
        }
    }
};

template <int W, bool Sampling>
pair<double, function<void(LinextSampler&)>> countLinext(
    const Poset<W>& poset,
    ExactCounterGlobalMemoryLimit& memLimit
) {
    BigHashTable<W, Sampling> mem(memLimit);
    double count = countLinext_(poset, mem, poset.allVerts());
    function<void(LinextSampler&)> samplerGen = CreateSamplerGen<W, Sampling>::run(poset, mem, count);
    return make_pair(count, samplerGen);
}

template <int W, bool Sampling>
pair<double, function<void(LinextSampler&)>> exactCountImpl(
    const Poset<W>& poset,
    ExactCounterGlobalMemoryLimit& memLimit
) {
    int n = poset.size();
    Bitset<W> allVerts = poset.allVerts();
    bool invert = !Sampling && poset.maximals(allVerts).count() < poset.minimals(allVerts).count();

    if(false) {
#define FOR_EACH_POSET_W_ELEM(W2) \
    } else if(n <= Bitset<W2>::BitCount) { \
        Poset<W2> poset2(n); \
        for(int v = 0; v < n; ++v) { \
            poset.succCover(v).iterate([&](int x) { \
                if(invert) { \
                    poset2.add(x, v); \
                } else { \
                    poset2.add(v, x); \
                } \
            }); \
        } \
        auto result = countLinext<W2, Sampling>(poset2, memLimit); \
        if(isfinite(result.first)) { \
            result.first = log(result.first); \
        } else { \
            result.first = Infinity; \
        } \
        return result;
FOR_EACH_POSET_W
#undef FOR_EACH_POSET_W_ELEM
    } else {
        fail("Poset is too large");
        return make_pair(Infinity, [](LinextSampler&) { });
    }
}

}

template <int W>
double computeExactLinextCount(
    const Poset<W>& poset,
    ExactCounterGlobalMemoryLimit& memLimit
) {
    return exactCountImpl<W, false>(poset, memLimit).first;
}

template <int W>
pair<double, function<void(LinextSampler&)>> createExactLinextSampler(
    const Poset<W>& poset,
    ExactCounterGlobalMemoryLimit& memLimit
) {
    return exactCountImpl<W, true>(poset, memLimit);
}

#define FOR_EACH_POSET_W_ELEM(W) \
    template double computeExactLinextCount<W>( \
        const Poset<W>& poset, \
        ExactCounterGlobalMemoryLimit& memLimit \
    ); \
    template pair<double, function<void(LinextSampler&)>> createExactLinextSampler<W>( \
        const Poset<W>& poset, \
        ExactCounterGlobalMemoryLimit& memLimit \
    );
FOR_EACH_POSET_W
