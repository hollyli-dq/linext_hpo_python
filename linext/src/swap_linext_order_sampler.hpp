#pragma once

#include "poset.hpp"
#include "pcg32.hpp"

template <int W>
class SwapLinextOrderSampler {
public:
    static constexpr ll InitialIterationCount = 16;

    SwapLinextOrderSampler(const Poset<W>& po, Bitset<W> verts, int vert1, int vert2)
        : po(&po),
          vert1(vert1),
          vert2(vert2)
    {
        assume(isSubset(verts, po.allVerts()));
        assume(po.allVerts()[vert1]);
        assume(po.allVerts()[vert2]);
        assume(vert1 != vert2);
        
        int order[Bitset<W>::BitCount];
        int descendantCount[Bitset<W>::BitCount];
        
        k = 0;
        verts.iterate([&](int v) {
            order[k] = v;
            descendantCount[v] = intersection(po.succ(v), verts).count() + 1;
            ++k;
        });
        assume(k >= 2);
        
        sort(order, order + k, [&](int a, int b) {
            return descendantCount[a] > descendantCount[b];
        });
        
        int p = 0;
        for(int i = 0; i < k - 1; ++i) {
            if(descendantCount[order[p]] >= k - i) {
                initialState[i] = order[p++];
            } else {
                initialState[i] = -1;
            }
        }
        initialState[k - 1] = order[p++];
        
        queueSize = 0;
        for(; p < k; ++p) {
            queue[queueSize++] = order[p];
        }

        initialBackStep = pcg32.backwardStep(InitialIterationCount);

        int idxBits = k == 2 ? 0 : 32 - __builtin_clz(k - 2);
        idxMask = ((uint32_t)1 << idxBits) - (uint32_t)1;
        uint64_t activeBound64 = ((uint64_t)1 << (31 + idxBits)) / (k - 1);
        activeBound64 = min(activeBound64, ((uint64_t)1 << 32) - (uint64_t)1);
        activeBound = activeBound64;
    }
    
    // Retuns true iff vert1 is before vert2
    bool sample() {
        PCG32Step backStep = initialBackStep;
        ll iterCount = InitialIterationCount;
        ll finalizeIterCount = 0;
        PCG32 startPCG32(0, 0);
        while(true) {
            pcg32.skip(backStep);
            startPCG32 = pcg32;

            copy(initialState, initialState + k, state);
            int queuePos = 0;
            
            for(ll iter = 0; iter < iterCount; ++iter) {
                uint32_t randVal = pcg32();
                bool d = randVal < activeBound;
                int j = randVal & idxMask;
                
                if(d && j < k - 1) {
                    if(state[j] == -1 || state[j + 1] == -1 || !po->has(state[j], state[j + 1])) {
                        swap(state[j], state[j + 1]);
                        if(state[k - 1] == -1) {
                            state[k - 1] = queue[queuePos++];
                        }
                    }
                }
            }
            
            if(queuePos == queueSize) {
                break;
            }

            backStep = backStep + backStep;
            finalizeIterCount += iterCount;
            iterCount += iterCount;
            pcg32 = startPCG32;
        }

        for(ll iter = 0; iter < finalizeIterCount; ++iter) {
            uint32_t randVal = pcg32();
            bool d = randVal < activeBound;
            int j = randVal & idxMask;

            if(d && j < k - 1) {
                if(!po->has(state[j], state[j + 1])) {
                    swap(state[j], state[j + 1]);
                }
            }
        }

        pcg32 = startPCG32;

        int i = 0;
        while(true) {
            if(state[i] == vert1) {
                return true;
            }
            if(state[i] == vert2) {
                return false;
            }
            ++i;
        }
    }
    
private:
    const Poset<W>* po;
    int vert1;
    int vert2;
    int k;
    int state[Bitset<W>::BitCount];
    int initialState[Bitset<W>::BitCount];
    int queue[Bitset<W>::BitCount];
    int queueSize;
    PCG32 pcg32;
    PCG32Step initialBackStep;
    uint32_t idxMask;
    uint32_t activeBound;
};
