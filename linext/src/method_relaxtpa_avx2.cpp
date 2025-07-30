#include "relaxtpa_vectorized.hpp"
#include "avx2vector.hpp"

template <int W>
void method_relaxtpa_avx2(const Poset<W>& poset, double epsilon, double delta) {
    runVectorizedRelaxationTPA<W, AVX2Vector>(poset, epsilon, delta);
}

#define FOR_EACH_POSET_W_ELEM(W) \
    template void method_relaxtpa_avx2<W>(const Poset<W>&, double, double);
FOR_EACH_POSET_W
