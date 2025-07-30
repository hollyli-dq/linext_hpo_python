#ifdef LINEXT_USE_CUDA

#include "relaxtpa_common.hpp"
#include "relaxtpa_gpu.hpp"
#include "tpa.hpp"

namespace {

vector<vector<uint32_t>> createWarpOps(
    const vector<vector<int>>& blocks,
    const RelaxationSamplingContext& relax
) {
    if(blocks.empty()) {
        vector<vector<uint32_t>> warpOps(WarpCount);
        for(int warpIdx = 0; warpIdx < WarpCount; ++warpIdx) {
            warpOps[warpIdx].push_back(0);
            warpOps[warpIdx].push_back(1);
            warpOps[warpIdx].push_back(0);
            warpOps[warpIdx].push_back(2);
        }
        return warpOps;
    }

    vector<int> stepBit(WarpCount, 0);
    vector<vector<uint32_t>> warpOps(WarpCount);

    auto addBlock = [&](vector<int> block) {
        assume(!block.empty());
        shuffle(block.begin(), block.end(), rng);

        for(int warpIdx = 0; warpIdx < WarpCount; ++warpIdx) {
            vector<int> ops;
            for(int i = warpIdx; i < (int)block.size(); i += WarpCount) {
                int v = block[i];
                const int* hardPredPtr = relax.hardPred(v).a;
                const int* softPredPtr = relax.softPred(v).a;
                const int* hardSuccPtr = relax.hardSucc(v).a;
                const int* softSuccPtr = relax.softSucc(v).a;
                while(true) {
                    size_t prevSize = ops.size();
                    if(hardPredPtr != relax.hardPred(v).b) {
                        ops.push_back(*hardPredPtr++ | 0x5000);
                    } else if(softPredPtr != relax.softPred(v).b) {
                        ops.push_back(*softPredPtr++ | 0x7000);
                    }
                    if(hardSuccPtr != relax.hardSucc(v).b) {
                        ops.push_back(*hardSuccPtr++ | 0x6000);
                    } else if(softSuccPtr != relax.softSucc(v).b) {
                        ops.push_back(*softSuccPtr++ | 0x4000);
                    }
                    if(ops.size() == prevSize) {
                        break;
                    }
                }
                ops.push_back(v | 0x8000 | stepBit[warpIdx]);
                stepBit[warpIdx] ^= 0x1000;
            }
            while(ops.size() & 3) {
                ops.push_back(0);
            }
            for(int i = 0; i < (int)ops.size(); i += 2) {
                warpOps[warpIdx].push_back((uint32_t)ops[i] | ((uint32_t)ops[i + 1] << 16));
            }
            warpOps[warpIdx].push_back(0);
            warpOps[warpIdx].push_back(0);
        }
    };

    for(int i = 0; i < (int)blocks.size(); ++i) {
        addBlock(blocks[i]);
    }
    for(int warpIdx = 0; warpIdx < WarpCount; ++warpIdx) {
        warpOps[warpIdx].pop_back();
        warpOps[warpIdx].push_back(1);
    }

    for(int i = (int)blocks.size() - 1; i >= 0; --i) {
        addBlock(blocks[i]);
    }
    for(int warpIdx = 0; warpIdx < WarpCount; ++warpIdx) {
        warpOps[warpIdx].pop_back();
        warpOps[warpIdx].push_back(2);
    }

    return warpOps;
}

void method_relaxtpa_gpu(
    const RelaxationSamplingContext& relax,
    const vector<vector<int>>& blocks,
    double epsilon,
    double delta
) {
    vector<vector<uint32_t>> warpOps = createWarpOps(blocks, relax);

    epsilon = log(1.0 + epsilon);
    msg("PARALLEL_TPA_EPSILON ", epsilon);
    msg("PARALLEL_TPA_DELTA ", delta);

    ll prelimWalkCount = tpaPreliminaryWalkCount(delta);
    msg("PARALLEL_TPA_PRELIM_WALK_COUNT ", prelimWalkCount);

    msg("PARALLEL_TPA_FINAL_WALK_COUNT_TABLE_COMPUTE_START");
    vector<ll> finalWalkCountTable;
    while((double)finalWalkCountTable.size() - 1.0 < max(relax.relaxationLogCount(), 1.0)) {
        ll walkCount = tpaFinalWalkCount(
            epsilon, delta,
            (double)finalWalkCountTable.size()
        );
        finalWalkCountTable.push_back(walkCount);
    }
    msg("PARALLEL_TPA_FINAL_WALK_COUNT_TABLE_COMPUTE_END");
    msg("PARALLEL_TPA_FINAL_WALK_COUNT_TABLE_SIZE ", finalWalkCountTable.size());

    runRelaxTpaGPU(
        relax.vertCount(),
        relax.relaxationLogCount(),
        finalWalkCountTable,
        warpOps,
        prelimWalkCount
    );
}

template <int W>
vector<vector<int>> createBlocks(const RelaxationSamplingContext& relax) {
    typedef Bitset<W> B;
    int n = relax.vertCount();
    if(n == 0) {
        return vector<vector<int>>();
    }

    vector<B> neigh(n, B::empty());
    for(int v = 0; v < n; ++v) {
        for(const int* x : relax.neighbors(v)) {
            neigh[v].add(*x);
        }
    }

    set<pair<int, int>> vertsByDeg;
    for(int v = 0; v < n; ++v) {
        vertsByDeg.emplace(neigh[v].count(), v);
    }
    
    vector<int> order;
    while(!vertsByDeg.empty()) {
        auto it = vertsByDeg.begin();
        int v = it->second;
        vertsByDeg.erase(it);

        neigh[v].iterate([&](int x) {
            int d = neigh[x].count();
            bool erased = (bool)vertsByDeg.erase(make_pair(d, x));
            assume(erased);
            neigh[x].del(v);
            vertsByDeg.emplace(d - 1, x);
        });

        order.push_back(v);
    }
    assume((int)order.size() == n);
    
    reverse(order.begin(), order.end());
    vector<int> vertColor(n, -1);
    int colorCount = 0;
    for(int v : order) {
        B neighColors = B::empty();
        neigh[v].iterate([&](int x) {
            assume(vertColor[x] >= 0 && vertColor[x] < n - 1);
            neighColors.add(vertColor[x]);
        });
        int color = difference(B::range(n), neighColors).min();
        if(color >= colorCount) {
            colorCount = color + 1;
        }
        vertColor[v] = color;
    }

    vector<vector<int>> blocks(colorCount);
    for(int v = 0; v < n; ++v) {
        assume(vertColor[v] >= 0 && vertColor[v] < colorCount);
        blocks[vertColor[v]].push_back(v);
    }

    return blocks;
}

}

template <int W>
void method_relaxtpa_gpu(const Poset<W>& poset, double epsilon, double delta) {
    RelaxationSamplingContext relax(poset);

    msg("INDEP_BLOCKS_CREATE_START");
    vector<vector<int>> blocks = createBlocks<W>(relax);
    msg("INDEP_BLOCKS_CREATE_END");

    msg("INDEP_BLOCKS_COUNT ", blocks.size());
    stringstream sizess;
    for(const vector<int>& block : blocks) {
        sizess << ' ' << block.size();
    }
    msg("INDEP_BLOCKS_SIZES", sizess.str());
    
    method_relaxtpa_gpu(relax, blocks, epsilon, delta);
}

#define FOR_EACH_POSET_W_ELEM(W) \
    template void method_relaxtpa_gpu<W>(const Poset<W>&, double, double);
FOR_EACH_POSET_W

#endif
