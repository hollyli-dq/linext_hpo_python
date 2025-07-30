#pragma once

#include "bitset.hpp"

template <int W>
class Poset {
public:
    typedef Bitset<W> B;

    Poset() {}
    Poset(int size) {
        assume(size >= 0 && size <= B::BitCount);
        pred_.resize(size, B::empty());
        succ_.resize(size, B::empty());
    }

    int size() const {
        return pred_.size();
    }
    B allVerts() const {
        return B::range(size());
    }

    bool has(int a, int b) const {
        assume(a >= 0 && a < size());
        assume(b >= 0 && b < size());
        return succ_[a][b];
    }
    B pred(int v) const {
        assume(v >= 0 && v < size());
        return pred_[v];
    }
    B succ(int v) const {
        assume(v >= 0 && v < size());
        return succ_[v];
    }
    B predReflex(int v) const {
        assume(v >= 0 && v < size());
        B ret = pred_[v];
        ret.add(v);
        return ret;
    }
    B succReflex(int v) const {
        assume(v >= 0 && v < size());
        B ret = succ_[v];
        ret.add(v);
        return ret;
    }
    B predCover(int v, B verts) const {
        assume(isSubset(verts, allVerts()));
        assume(v >= 0 && v < size());
        B ret = intersection(pred_[v], verts);
        ret.iterate([&](int x) {
            ret = difference(ret, pred_[x]);
        });
        return ret;
    }
    B predCover(int v) const {
        return predCover(v, allVerts());
    }
    B succCover(int v, B verts) const {
        assume(isSubset(verts, allVerts()));
        assume(v >= 0 && v < size());
        B ret = intersection(succ_[v], verts);
        ret.iterate([&](int x) {
            ret = difference(ret, succ_[x]);
        });
        return ret;
    }
    B succCover(int v) const {
        return succCover(v, allVerts());
    }

    template <typename F>
    void components(B verts, F f) const {
        assume(isSubset(verts, allVerts()));

        B todo = verts;
        todo.iterate([&](int v) {
            B seen = B::empty();
            seen.add(v);
            B comp = B::empty();
            while(true) {
                B d = difference(intersection(seen, todo), comp);
                if(d.isEmpty()) {
                    break;
                }
                d.iterate([&](int x) {
                    comp.add(x);
                    seen = unio(seen, unio(pred(x), succ(x)));
                });
            }
            f((const B&)comp);
            todo = difference(todo, comp);
        });
    }

    template <typename F>
    bool topoSortWhile_(int v, B& todo, F& f) const {
        while(true) {
            B deps = intersection(pred(v), todo);
            if(deps.isEmpty()) {
                break;
            }
            if(!topoSortWhile_(deps.min(), todo, f)) {
                return false;
            }
        }
        todo.del(v);
        return (bool)f(v);
    }
    template <typename F>
    bool topoSortWhile(B verts, F f) const {
        assume(isSubset(verts, allVerts()));
        B todo = verts;
        return verts.iterateWhile([&](int v) {
            if(!todo[v]) return true;
            return topoSortWhile_(v, todo, f);
        });
    }

    B minimals(B verts) const {
        assume(isSubset(verts, allVerts()));
        B ret = B::empty();
        verts.iterate([&](int v) {
            if(isDisjoint(pred(v), verts)) {
                ret.add(v);
            }
        });
        return ret;
    }
    B maximals(B verts) const {
        assume(isSubset(verts, allVerts()));
        B ret = B::empty();
        verts.iterate([&](int v) {
            if(isDisjoint(succ(v), verts)) {
                ret.add(v);
            }
        });
        return ret;
    }

    template <B(Poset<W>::*ExtremalFunc)(B) const>
    double countLinext_(
        unordered_map<B, double>& mem,
        B verts,
        ll& expansionsLeft
    ) const {
        int vertCount = verts.count();
        if(vertCount <= 1) {
            return 1.0;
        }
        double result = 1.0;
        double logCoef = lgamma((double)vertCount + 1.0);
        components(verts, [&](B comp) {
            auto it = mem.find(comp);
            if(it == mem.end()) {
                if(!expansionsLeft) {
                    return;
                }
                --expansionsLeft;
                double val = 0.0;
                (this->*ExtremalFunc)(comp).iterate([&](int v) {
                    B subComp = comp;
                    subComp.del(v);
                    val += countLinext_<ExtremalFunc>(mem, subComp, expansionsLeft);
                });
                it = mem.emplace(comp, val).first;
            }
            result *= it->second;
            logCoef -= lgamma((double)comp.count() + 1.0);
        });
        return result * exp(logCoef);
    }

    // Returns -1.0 if maxExpansions is not enough. May overflow: check for infinity
    double countLinext(B verts, ll maxExpansions = LONG_MAX) const {
        assume(isSubset(verts, allVerts()));
        assume(maxExpansions >= 0);

        unordered_map<B, double> mem;
        double ret = 1.0;
        double logCoef = 0.0;
        bool multipleComponents = false;
        ll expansionsLeft = maxExpansions;
        components(verts, [&](B comp) {
            mem.clear();
            if(minimals(comp).count() < maximals(comp).count()) {
                ret *= countLinext_<&Poset::minimals>(mem, comp, expansionsLeft);
            } else {
                ret *= countLinext_<&Poset::maximals>(mem, comp, expansionsLeft);
            }
            if(comp != verts) {
                multipleComponents = true;
                logCoef -= lgamma((double)comp.count() + 1.0);
            }
        });
        if(!expansionsLeft) {
            return -1.0;
        }
        if(multipleComponents) {
            logCoef += lgamma((double)verts.count() + 1.0);
            ret *= exp(logCoef);
        }
        return ret;
    }

    void add(int a, int b) {
        assume(a >= 0 && a < size());
        assume(b >= 0 && b < size());
        assume(a != b);
        assume(!has(b, a));

        if(has(a, b)) {
            return;
        }
        B aPred = predReflex(a);
        B bSucc = succReflex(b);
        aPred.iterate([&](int v) {
            succ_[v] = unio(succ_[v], bSucc);
        });
        bSucc.iterate([&](int v) {
            pred_[v] = unio(pred_[v], aPred);
        });
    }

private:
    vector<B> pred_;
    vector<B> succ_;
};

#define FOR_EACH_POSET_W \
    FOR_EACH_POSET_W_ELEM(1) \
    FOR_EACH_POSET_W_ELEM(2) \
    FOR_EACH_POSET_W_ELEM(4) \
    FOR_EACH_POSET_W_ELEM(8) \
    FOR_EACH_POSET_W_ELEM(16)

template <typename Handler>
void readPoset(istream& in, bool shuffleElements, Handler& handler) {
    string line;
    getline(in, line);
    if(in.fail()) {
        fail("Reading poset file failed");
    }

    int n = 0;

    stringstream liness1(line);
    int x;
    while(liness1 >> x) {
        ++n;
    }

    auto run = [&](auto& poset) {
        vector<int> perm(n);
        for(int i = 0; i < n; ++i) {
            perm[i] = i;
        }
        if(shuffleElements) {
            shuffle(perm.begin(), perm.end(), rng);
        }

        stringstream liness2(line);
        for(int i = 0; i < n; ++i) {
            int x;
            liness2 >> x;
            if(x) {
                if(i == 0) {
                    fail("Poset::read: Invalid poset");
                }
                poset.add(perm[0], perm[i]);
            }
        }

        for(int i = 1; i < n; ++i) {
            for(int j = 0; j < n; ++j) {
                int x;
                if(!(in >> x)) {
                    fail("Poset::read: Not enough values in input");
                }
                if(x) {
                    if(i == j || poset.has(perm[j], perm[i])) {
                        fail("Poset::read: Invalid poset");
                    }
                    poset.add(perm[i], perm[j]);
                }
            }
        }
        handler(poset);
    };

    if(false) {
#define FOR_EACH_POSET_W_ELEM(W) \
    } else if(n <= Bitset<W>::BitCount) { \
        Poset<W> poset(n); \
        run(poset);
FOR_EACH_POSET_W
#undef FOR_EACH_POSET_W_ELEM
    } else {
        fail("Read poset too large");
    }
}

template <int W>
void showPosetInfo(const Poset<W>& poset, const string& name = "POSET") {
    msg(name, "_SIZE ", poset.size());

    int constraintCount = 0;
    int coverConstraintCount = 0;
    for(int v = 0; v < poset.size(); ++v) {
        constraintCount += poset.succ(v).count();
        coverConstraintCount += poset.succCover(v).count();
    }
    msg(name, "_CONSTRAINT_COUNT ", constraintCount);
    msg(name, "_COVER_CONSTRAINT_COUNT ", coverConstraintCount);
}
