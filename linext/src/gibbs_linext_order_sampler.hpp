#pragma once

#include "poset.hpp"
#include "pcg32.hpp"

template <int W>
class GibbsLinextOrderSampler {
public:
    static constexpr ll InitialIterationCount = 16;

    GibbsLinextOrderSampler(const Poset<W>& po, Bitset<W> verts, int vert1, int vert2)
        : po(&po)
    {
        assume(isSubset(verts, po.allVerts()));
        assume(po.allVerts()[vert1]);
        assume(po.allVerts()[vert2]);
        assume(vert1 != vert2);

        int vertIdx[Bitset<W>::BitCount];

        k = 0;
        verts.iterate([&](int v) {
            vertIdx[v] = k++;
        });

        fill(pred, pred + k, Bitset<W>::empty());
        fill(succ, succ + k, Bitset<W>::empty());

        verts.iterate([&](int v) {
            Bitset<W> seen;
            po.topoSortWhile(intersection(verts, po.succ(v)), [&](int x) {
                if(!seen[x]) {
                    pred[vertIdx[x]].add(vertIdx[v]);
                    succ[vertIdx[v]].add(vertIdx[x]);
                }
                return true;
            });
        });

        vertIdx1 = vertIdx[vert1];
        vertIdx2 = vertIdx[vert2];

        initialBackStep = pcg32.backwardStep(2 * InitialIterationCount);
    }
    
    // Retuns true iff vert1 is before vert2
    bool sample() {
        PCG32Step backStep = initialBackStep;
        ll iterCount = InitialIterationCount;
        while(true) {
            pcg32.skip(backStep);
            fill(left, left + k, 0);
            fill(right, right + k, (uint32_t)-1);

            for(ll iterIdx = 0; iterIdx < iterCount; ++iterIdx) {
                int j = pcg32() % k;
                uint32_t u = pcg32();

                uint32_t aLeft = 0;
                uint32_t bLeft = (uint32_t)-1;
                uint32_t aRight = 0;
                uint32_t bRight = (uint32_t)-1;

                pred[j].iterate([&](int x) {
                    if(left[x] > aLeft) aLeft = left[x];
                    if(right[x] > aRight) aRight = right[x];
                });
                succ[j].iterate([&](int x) {
                    if(left[x] < bLeft) bLeft = left[x];
                    if(right[x] < bRight) bRight = right[x];
                });

                left[j] = aLeft + (uint32_t)(((uint64_t)u * (uint64_t)(bLeft - aLeft)) >> 32);
                right[j] = aRight + (uint32_t)(((uint64_t)u * (uint64_t)(bRight - aRight)) >> 32);
            }

            if(right[vertIdx1] < left[vertIdx2] || right[vertIdx2] < left[vertIdx1]) {
                break;
            }

            iterCount += iterCount;
            backStep = backStep + backStep;
        }

        pcg32.skip(backStep);
        return right[vertIdx1] < left[vertIdx2];
    }
    
private:
    const Poset<W>* po;
    int vertIdx1;
    int vertIdx2;
    int k;
    int subVerts[Bitset<W>::BitCount];
    Bitset<W> pred[Bitset<W>::BitCount];
    Bitset<W> succ[Bitset<W>::BitCount];
    uint32_t left[Bitset<W>::BitCount];
    uint32_t right[Bitset<W>::BitCount];
    PCG32 pcg32;
    PCG32Step initialBackStep;
};
