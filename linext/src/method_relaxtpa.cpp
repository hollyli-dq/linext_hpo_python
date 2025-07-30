#include "relaxtpa_basic.hpp"

template <int W>
void method_relaxtpa(const Poset<W>& poset, double epsilon, double delta) {
    runBasicRelaxationTPA(poset, epsilon, delta);
}

#define FOR_EACH_POSET_W_ELEM(W) \
    template void method_relaxtpa<W>(const Poset<W>&, double, double);
FOR_EACH_POSET_W
