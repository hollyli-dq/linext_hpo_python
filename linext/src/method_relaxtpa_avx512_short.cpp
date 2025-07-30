#ifdef __AVX512F__

#include "relaxtpa_vectorized.hpp"
#include "shortavx512vector.hpp"

template <int W>
void method_relaxtpa_avx512_short(const Poset<W>& poset, double epsilon, double delta) {
    runVectorizedRelaxationTPA<W, ShortAVX512Vector>(poset, epsilon, delta);
}

#define FOR_EACH_POSET_W_ELEM(W) \
    template void method_relaxtpa_avx512_short<W>(const Poset<W>&, double, double);
FOR_EACH_POSET_W

#endif
