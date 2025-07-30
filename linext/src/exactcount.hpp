#pragma once

#include "poset.hpp"

class ExactCounterGlobalMemoryLimit {
public:
    struct OutOfMemory {};

    ExactCounterGlobalMemoryLimit(size_t memLimit);
    ~ExactCounterGlobalMemoryLimit();

    ExactCounterGlobalMemoryLimit(const ExactCounterGlobalMemoryLimit&) = delete;
    ExactCounterGlobalMemoryLimit(ExactCounterGlobalMemoryLimit&&) = delete;

    ExactCounterGlobalMemoryLimit& operator=(const ExactCounterGlobalMemoryLimit&) = delete;
    ExactCounterGlobalMemoryLimit& operator=(ExactCounterGlobalMemoryLimit&&) = delete;

    double memoryUsageRatio() const;

    bool isOutOfMemory() const;
    void checkOutOfMemory() const;

    struct AccessKey;
    void takeMem_(AccessKey, size_t size);
    void returnMem_(AccessKey, size_t size);

private:
    size_t memLimit_;
    atomic<bool> outOfMemory_;
    atomic<size_t> memInUse_;
};

// Returns elements of the linear extensions in order when called successively.
// Do not call for more times than there are elements in the poset.
// Calling a default constructed LinextSampler is undefined behavior.
class LinextSampler {
public:
    LinextSampler() {}

    int operator()() {
        return func_((void*)data_);
    }

    struct AccessKey;
    void* accessData_(AccessKey);
    typedef int (*Func)(void*);
    Func& accessFunc_(AccessKey);

    static constexpr size_t DataBytes = 12320;

private:
    alignas(max_align_t) char data_[DataBytes];
    Func func_;
};

// Returns logarithm of number of linear extensions or infinity if there was an overflow
template <int W>
double computeExactLinextCount(
    const Poset<W>& poset,
    ExactCounterGlobalMemoryLimit& memLimit
);


// Returns pair of:
//   - Logarithm of number of linear extensions or infinity if there was an overflow
//   - Thread safe function that initializes a LinextSampler so that it can be used to sample
//     one linear extension uniformly at random
template <int W>
pair<double, function<void(LinextSampler&)>> createExactLinextSampler(
    const Poset<W>& poset,
    ExactCounterGlobalMemoryLimit& memLimit
);
