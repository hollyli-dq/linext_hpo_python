#pragma once

#include "poset.hpp"

namespace relaxation_ {

inline double logAdd(double a, double b) {
    if(a < b) {
        swap(a, b);
    }
    if(b == -Infinity) {
        return a;
    } else {
        return a + log(1.0 + exp(b - a));
    }
}

template <int W>
inline vector<double> treeSpectrum_(
    const Poset<W>& poset,
    Bitset<W> verts,
    int a,
    Bitset<W>& nogo
) {
    auto logNCR = [](int ni, int ki) {
        double n = (double)ni;
        double k = (double)ki;
        return lgamma(n + 1.0) - (lgamma(k + 1.0) + lgamma(n - k + 1.0));
    };

    int b;
    bool forward;
    Bitset<W> succsLeft = difference(poset.succCover(a, verts), nogo);
    if(!succsLeft.isEmpty()) {
        b = succsLeft.min();
        forward = true;
    } else {
        Bitset<W> predsLeft = difference(poset.predCover(a, verts), nogo);
        if(!predsLeft.isEmpty()) {
            b = predsLeft.min();
            forward = false;
        } else {
            vector<double> ret;
            ret.push_back(0.0);
            return ret;
        }
    }
    
    nogo.add(b);
    vector<double> A = treeSpectrum_(poset, verts, a, nogo);
    nogo.del(b);
    nogo.add(a);
    vector<double> B = treeSpectrum_(poset, verts, b, nogo);
    nogo.del(a);
    
    int As = A.size();
    int Bs = B.size();
    int Rs = As + Bs;
    
    vector<double> R(Rs);
    
    if(forward) {
        for(int i = Bs - 2; i >= 0; --i) {
            B[i] = logAdd(B[i], B[i + 1]);
        }
        
        for(int r = 0; r < Rs; ++r) {
            double val = -Infinity;
            for(int i = max(0, r - Bs + 1); i < As && i < r + 1; ++i) {
                val = logAdd(val, A[i] + logNCR(r, i) + logNCR(Rs - r - 1, As - i - 1) + B[r - i]);
            }
            R[r] = val;
        }
    } else {
        for(int i = 1; i < Bs; ++i) {
            B[i] = logAdd(B[i], B[i - 1]);
        }
        
        for(int r = 0; r < Rs; ++r) {
            double val = -Infinity;
            for(int i = max(0, r - Bs); i < As && i < r; ++i) {
                val = logAdd(val, A[i] + logNCR(r, i) + logNCR(Rs - r - 1, As - i - 1) + B[r - i - 1]);
            }
            R[r] = val;
        }
    }
    
    return R;
}

template <int W>
double treeLinextLogCount(const Poset<W>& poset, Bitset<W> verts) {
    if(verts.count() <= 1) {
        return 0.0;
    }
    Bitset<W> nogo = Bitset<W>::empty();
    vector<double> spectrum = treeSpectrum_(poset, verts, verts.min(), nogo);
    double logCount = -Infinity;
    for(double x : spectrum) {
        logCount = logAdd(logCount, x);
    }
    return logCount;
}

template <int W>
bool isComponentTreeDFS_(const Poset<W>& poset, Bitset<W> comp, int v, int p, Bitset<W>& seen) {
    if(seen[v]) {
        return false;
    }
    seen.add(v);
    return unio(
        poset.predCover(v, comp),
        poset.succCover(v, comp)
    ).iterateWhile([&](int x) {
        if(x == p) {
            return true;
        }
        return isComponentTreeDFS_(poset, comp, x, v, seen);
    });
}
template <int W>
bool isComponentTree(const Poset<W>& poset, Bitset<W> comp) {
    if(comp.isEmpty()) {
        return false;
    }
    Bitset<W> seen = Bitset<W>::empty();
    return isComponentTreeDFS_(poset, comp, comp.min(), -1, seen);
}

struct UnionFind {
    UnionFind(int n) : parent(n) {
        for(int i = 0; i < n; ++i) {
            parent[i] = i;
        }
    }
    
    int find(int x) {
        if(parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }
    
    void merge(int x, int y) {
        x = find(x);
        y = find(y);
        parent[x] = y;
    }
    
    vector<int> parent;
};

template <int W>
double findPosetRelaxation(const Poset<W>& poset, Poset<W>& relaxation, Bitset<W> verts);

template <int W>
double findComponentRelaxation(const Poset<W>& poset, Poset<W>& relaxation, Bitset<W> comp) {
    typedef Bitset<W> B;

    int compSize = comp.count();
    if(compSize <= 1) {
        return 0.0;
    }

    // Perfectly ordered components
    {
        B left = B::empty();
        B right = comp;
        if(!poset.topoSortWhile(comp, [&](int v) {
            left.add(v);
            right = intersection(right, poset.succ(v));
            return right.isEmpty() || unio(left, right) != comp;
        })) {
            double leftLogCount = findPosetRelaxation(poset, relaxation, left);
            double rightLogCount = findPosetRelaxation(poset, relaxation, right);
            left.iterate([&](int a) {
                right.iterate([&](int b) {
                    relaxation.add(a, b);
                });
            });
            return leftLogCount + rightLogCount;
        }
    }

    // Trees
    if(isComponentTree(poset, comp)) {
        double logCount = treeLinextLogCount(poset, comp);
        comp.iterate([&](int a) {
            intersection(poset.succ(a), comp).iterate([&](int b) {
                relaxation.add(a, b);
            });
        });
        return logCount;
    }

    // Linear extensions of the poset easily countable using exact DP
    {
        double count = poset.countLinext(comp, 10000);
        if(isfinite(count) && count != -1.0) {
            comp.iterate([&](int a) {
                intersection(poset.succ(a), comp).iterate([&](int b) {
                    relaxation.add(a, b);
                });
            });
            return log(count);
        }
    }

    double logCount;
    Poset<W> baseRelaxation = relaxation;

    // Heuristic partition into three sets (left, right and other) such that
    // left and right are completely ordered
    {
        auto computeHeuristic = [&](B left, B right) {
            double a = (double)left.count();
            double b = (double)right.count();
            return lgamma(a + b + 1.0) - lgamma(a + 1.0) - lgamma(b + 1.0);
        };
        double bestHeuristic = -1.0;
        B bestLeft = B::empty();
        B bestRight = B::empty();
        comp.iterate([&](int start) {
            B left = B::empty();
            left.add(start);
            B right = intersection(poset.succ(start), comp);
            double heuristic = computeHeuristic(left, right);
            while(true) {
                double bestNewHeuristic = -1.0;
                int bestAdd = -1;
                difference(comp, left).iterate([&](int v) {
                    B newLeft = left;
                    newLeft.add(v);
                    B newRight = intersection(right, poset.succ(v));
                    double newHeuristic = computeHeuristic(newLeft, newRight);
                    if(newHeuristic > bestNewHeuristic) {
                        bestNewHeuristic = newHeuristic;
                        bestAdd = v;
                    }
                });
                if(bestAdd != -1 && bestNewHeuristic > heuristic) {
                    left.add(bestAdd);
                    right = intersection(right, poset.succ(bestAdd));
                    heuristic = bestNewHeuristic;
                } else {
                    break;
                }
            }
            if(heuristic > bestHeuristic) {
                bestHeuristic = heuristic;
                bestLeft = left;
                bestRight = right;
            }
        });
        assume(!bestLeft.isEmpty() && !bestRight.isEmpty());

        B left = bestLeft;
        B right = bestRight;
        B leftRight = unio(left, right);
        B other = difference(comp, leftRight);
        int otherSize = other.count();

        double leftLogCount = findPosetRelaxation(poset, relaxation, left);
        double rightLogCount = findPosetRelaxation(poset, relaxation, right);
        double otherLogCount = findPosetRelaxation(poset, relaxation, other);
        logCount =
            lgamma((double)compSize + 1.0)
            - lgamma((double)otherSize + 1.0)
            - lgamma((double)(compSize - otherSize) + 1.0)
            + leftLogCount
            + rightLogCount
            + otherLogCount;
        
        left.iterate([&](int a) {
            right.iterate([&](int b) {
                relaxation.add(a, b);
            });
        });
    }

    // Spanning tree
    {
        vector<pair<int, int>> allEdges;
        comp.iterate([&](int v) {
            poset.succCover(v, comp).iterate([&](int x) {
                allEdges.emplace_back(v, x);
            });
        });
        for(int tryIdx = 0; tryIdx < 6; ++tryIdx) {
            shuffle(allEdges.begin(), allEdges.end(), rng);
            Poset<W> relaxationCand(poset.size());
            UnionFind uf(poset.size());
            for(auto edge : allEdges) {
                if(uf.find(edge.first) != uf.find(edge.second)) {
                    relaxationCand.add(edge.first, edge.second);
                    uf.merge(edge.first, edge.second);
                }
            }
            double logCountCand = treeLinextLogCount(relaxationCand, comp);
            if(logCountCand < logCount) {
                logCount = logCountCand;
                relaxation = baseRelaxation;
                comp.iterate([&](int v) {
                    relaxationCand.succCover(v, comp).iterate([&](int x) {
                        relaxation.add(v, x);
                    });
                });
            }
        }
    }

    return logCount;
}

template <int W>
double findPosetRelaxation(const Poset<W>& poset, Poset<W>& relaxation, Bitset<W> verts) {
    double logCount = lgamma((double)verts.count() + 1.0);
    poset.components(verts, [&](Bitset<W> comp) {
        logCount -= lgamma((double)comp.count() + 1.0);
        logCount += relaxation_::findComponentRelaxation(poset, relaxation, comp);
    });
    return logCount;
}

}

template <int W>
pair<Poset<W>, double> findRelaxation(const Poset<W>& poset) {
    Poset<W> relaxation(poset.size());
    double logCount = relaxation_::findPosetRelaxation(poset, relaxation, poset.allVerts());
    return make_pair(relaxation, logCount);
}
