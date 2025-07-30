#pragma once

#include "bit64.hpp"

template <int W>
struct Bitset {
    static_assert(W > 0, "The number of words in a bitset should be positive");
    static const int BitCount = W << 6;
    uint64_t words[W];

    Bitset() {}
    
    static Bitset<W> empty() {
        Bitset<W> ret;
        fill(ret.words, ret.words + W, 0);
        return ret;
    }
    static Bitset<W> range(int count) {
        assume(count >= 0 && count <= BitCount);
        Bitset<W> ret;
        int w = count >> 6;
        fill(ret.words, ret.words + w, (uint64_t)-1);
        int rem = count & 63;
        if(rem) {
            ret.words[w] = ((uint64_t)1 << rem) - (uint64_t)1;
            fill(ret.words + w + 1, ret.words + W, (uint64_t)0);
        } else {
            fill(ret.words + w, ret.words + W, (uint64_t)0);
        }
        return ret;
    }
    static Bitset<W> randomBits() {
        Bitset<W> ret;
        for(int w = 0; w < W; ++w) {
            ret.words[w] = UnifInt<uint64_t>()(rng);
        }
        return ret;
    }

    bool operator[](int i) const {
        assume(i >= 0);
        assume(i < BitCount);
        return (bool)(words[i >> 6] & ((uint64_t)1 << (i & 63)));
    }

    void add(int i) {
        words[i >> 6] |= (uint64_t)1 << (i & 63);
    }
    void del(int i) {
        words[i >> 6] &= ~((uint64_t)1 << (i & 63));
    }
    void toggle(int i) {
        words[i >> 6] ^= (uint64_t)1 << (i & 63);
    }

    bool isEmpty() const {
        for(int w = 0; w < W; ++w) {
            if(words[w]) return false;
        }
        return true;
    }
    bool hasAtMost1Ones() const {
        for(int w = 0; w < W; ++w) {
            if(words[w]) {
                uint64_t b = words[w] & -words[w];
                if(words[w] != b) {
                    return false;
                }
                for(int w2 = w + 1; w2 < W; ++w2) {
                    if(words[w2]) {
                        return false;
                    }
                }
                return true;
            }
        }
        return true;
    }
    int count() const {
        int ret = 0;
        for(int w = 0; w < W; ++w) {
            ret += popcount64(words[w]);
        }
        return ret;
    }

    int min() const {
        int base = 0;
        for(int w = 0; w < W; ++w) {
            if(words[w]) {
                return base + ctz64(words[w]);
            }
            base += 64;
        }
        assume(false);
    }

    template <typename F>
    bool iterateWhile(F f) const {
        int base = 0;
        for(int w = 0; w < W; ++w) {
            uint64_t mask = (uint64_t)-1;
            while(words[w] & mask) {
                int b = ctz64(words[w] & mask);
                mask = (uint64_t)-2 << b;
                int i = base + b;
                if(!f(i)) {
                    return false;
                }
            }
            base += 64;
        }
        return true;
    }
    template <typename F>
    void iterate(F f) const {
        iterateWhile([&](int i) {
            f(i);
            return true;
        });
    }

    int random() const {
        int c = count();
        assume(c > 0);
        int left = UnifInt<int>(0, c - 1)(rng);
        int base = 0;
        for(int w = 0; w < W; ++w) {
            int wc = popcount64(words[w]);
            left -= wc;
            if(left < 0) {
                return base + randomMaskElement64(words[w]);
            }
            base += 64;
        }
        assume(false);
    }

    Bitset<W> randomSubsetOfSize(int s) const {
        assume(s >= 0 && s <= count());
        Bitset<W> A = Bitset<W>::empty();
        Bitset<W> B = *this;
        while(true) {
            Bitset<W> M = intersection(unio(randomBits(), A), B);
            int sM = M.count();
            if(sM == s) {
                return M;
            }
            if(sM < s) {
                A = M;
            } else {
                B = M;
            }
        }
    }
};

template <int W>
const int Bitset<W>::BitCount;

template <int W>
bool operator==(Bitset<W> a, Bitset<W> b) {
    return equal(a.words, a.words + W, b.words);
}
template <int W>
bool operator!=(Bitset<W> a, Bitset<W> b) {
    return !(a == b);
}

namespace std {
template <int W>
struct hash<Bitset<W>> {
    size_t operator()(const Bitset<W>& a) const {
        uint64_t ret = 0;
        for(int w = 0; w < W; ++w) {
            ret = 31 * ret + a.words[w];
        }
        ret ^= (31 * ret) >> 32;
        return (size_t)ret;
    }
};
}

template <int W>
Bitset<W> unio(Bitset<W> a, Bitset<W> b) {
    Bitset<W> ret;
    for(int w = 0; w < W; ++w) {
        ret.words[w] = a.words[w] | b.words[w];
    }
    return ret;
}
template <int W>
Bitset<W> intersection(Bitset<W> a, Bitset<W> b) {
    Bitset<W> ret;
    for(int w = 0; w < W; ++w) {
        ret.words[w] = a.words[w] & b.words[w];
    }
    return ret;
}
template <int W>
Bitset<W> difference(Bitset<W> a, Bitset<W> b) {
    Bitset<W> ret;
    for(int w = 0; w < W; ++w) {
        ret.words[w] = a.words[w] & ~b.words[w];
    }
    return ret;
}

template <int W>
bool isSubset(Bitset<W> a, Bitset<W> b) {
    for(int w = 0; w < W; ++w) {
        if(a.words[w] & ~b.words[w]) {
            return false;
        }
    }
    return true;
}
template <int W>
bool isDisjoint(Bitset<W> a, Bitset<W> b) {
    for(int w = 0; w < W; ++w) {
        if(a.words[w] & b.words[w]) {
            return false;
        }
    }
    return true;
}
