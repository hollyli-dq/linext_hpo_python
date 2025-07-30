#include "exactcount.hpp"

template <int W>
void method_exact(const Poset<W>& poset, double epsilon, double delta) {
    ExactCounterGlobalMemoryLimit memLimit((size_t)30 * 1024 * 1024 * 1024);
    double result = computeExactLinextCount(poset, memLimit);
    if(isfinite(result)) {
        msg("LINEXT_LOG_COUNT ", result);
    } else {
        msg("EXACT_LINEXT_COUNT_OVERFLOW");
        fail("Overflow in computation");
    }
}

#define FOR_EACH_POSET_W_ELEM(W) \
    template void method_exact<W>(const Poset<W>&, double, double);
FOR_EACH_POSET_W
