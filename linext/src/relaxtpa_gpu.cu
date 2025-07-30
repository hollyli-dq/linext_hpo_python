#include "relaxtpa_gpu.hpp"

#include "pcg32.hpp"

namespace {

const int MaxCoord = (int)(((unsigned int)1 << 31) - 1);
const ll InitialIterationCount = 16;
const ll initialFinalWalkCount = (ll)1 << 62;

__device__ ll signedAtomicAdd(ll& var, ll val) {
    return (ll)atomicAdd((ull*)&var, (ull)val);
}

struct TPAState {
    ll prelimWalkCount;
    const ll* finalWalkCountTable;
    ll finalWalkCountTableSize;
    ll finalWalkCountTarget;
    ll finalWalkCount;
    ll prelimResults;
    ll nextWalkIdx;
    ll finalHitCount;
    ll totalIterCount;

    __device__ bool popWalkJob(bool& prelim) {
        ll walkIdx = signedAtomicAdd(nextWalkIdx, -1);
        if(walkIdx >= 0) {
            prelim = walkIdx >= initialFinalWalkCount;
            return true;
        } else {
            return false;
        }
    }

    __device__ void pushWalkResult(bool prelim, ll hitCount) {
        if(prelim) {
            // prelimResults stores number of results in top 16 bits
            ll increment = ((ll)1 << 48) | hitCount;
            ll results = signedAtomicAdd(prelimResults, increment);
            results += increment;

            ll prelimResultCount = results >> 48;
            if(prelimResultCount == prelimWalkCount) {
                ll prelimHitCount = results & (((ll)1 << 48) - (ll)1);
                ll prelimEstimateCeil = (prelimHitCount + prelimWalkCount - 1) / prelimWalkCount;
                prelimEstimateCeil = min(prelimEstimateCeil, finalWalkCountTableSize - 1);

                finalWalkCountTarget = finalWalkCountTable[prelimEstimateCeil];
                finalWalkCount = finalWalkCountTarget;

                ll decrement = initialFinalWalkCount - finalWalkCount;
                ll idx = signedAtomicAdd(nextWalkIdx, -decrement);
                idx -= decrement;

                if(idx < -1) {
                    finalWalkCount += -1 - idx;
                }
            }
        } else {
            signedAtomicAdd(finalHitCount, hitCount);
        }
    }

    double prelimEstimate() const {
        ll prelimHitCount = prelimResults & (((ll)1 << 48) - (ll)1);
        return (double)prelimHitCount / (double)prelimWalkCount;
    }

    double finalEstimate() const {
        return (double)finalHitCount / (double)finalWalkCount;
    }
};

#include "relaxtpa_gpu_kernel.cuh"
#define VERT_POS_USE_SHARED_MEM
#include "relaxtpa_gpu_kernel.cuh"
#undef VERT_POS_USE_SHARED_MEM

template <typename T>
struct GPUDeleter {
    void operator()(T* ptr) {
        CUDACHECK(cudaFree(ptr));
    }
};

template <typename T>
using GPUArray = unique_ptr<T[], GPUDeleter<T>>;

template <typename T>
GPUArray<T> createUninitializedGPUArray(int elemCount) {
    T* ptrDev;
    size_t size = sizeof(T) * elemCount;
    CUDACHECK(cudaMalloc(&ptrDev, size));
    return GPUArray<T>(ptrDev);
}

template <typename T>
GPUArray<T> vecToGPU(const vector<T>& vec) {
    T* ptrDev;
    size_t size = sizeof(T) * vec.size();
    CUDACHECK(cudaMalloc(&ptrDev, size));
    CUDACHECK(cudaMemcpy(ptrDev, vec.data(), size, cudaMemcpyHostToDevice));
    return GPUArray<T>(ptrDev);
}

template <typename T>
using GPUStruct = unique_ptr<T, GPUDeleter<T>>;

template <typename T>
GPUStruct<T> structToGPU(const T& src) {
    T* ptrDev;
    CUDACHECK(cudaMalloc(&ptrDev, sizeof(T)));
    CUDACHECK(cudaMemcpy(ptrDev, &src, sizeof(T), cudaMemcpyHostToDevice));
    return GPUStruct<T>(ptrDev);
}

template <typename T>
void structFromGPU(const GPUStruct<T>& gpuStruct, T& dest) {
    CUDACHECK(cudaMemcpy(&dest, gpuStruct.get(), sizeof(T), cudaMemcpyDeviceToHost));
}

cudaDeviceProp getGPUProp() {
    int device;
    CUDACHECK(cudaGetDevice(&device));
    cudaDeviceProp gpuProp;
    CUDACHECK(cudaGetDeviceProperties(&gpuProp, device));
    return gpuProp;
}

}

void runRelaxTpaGPU(
    int n,
    double relaxationLogCount,
    const vector<ll>& finalWalkCountTable,
    const vector<vector<uint32_t>>& warpOps,
    ll prelimWalkCount
) {
    cudaDeviceProp gpuProp = getGPUProp();
    msg("GPU_SM_COUNT ", gpuProp.multiProcessorCount);
    msg("GPU_SHARED_MEM_SIZE ", gpuProp.sharedMemPerMultiprocessor);
    int blockCount = gpuProp.multiProcessorCount;

    const int vertPosSizePerBlock = 32 * max(n, 1);

    msg("GPU_VERT_POS_BYTES_PER_BLOCK ", vertPosSizePerBlock * sizeof(int));
    bool useSharedMem;
    if(vertPosSizePerBlock * sizeof(int) <= gpuProp.sharedMemPerMultiprocessor) {
        msg("GPU_VERT_POS_USE_SHARED_MEM");
        useSharedMem = true;
    } else {
        msg("GPU_VERT_POS_USE_GLOBAL_MEM");
        useSharedMem = false;
    }

    GPUArray<ll> finalWalkCountTableDev = vecToGPU(finalWalkCountTable);

    vector<uint32_t> warpOpData;
    vector<int> warpOpOffsets(WarpCount);
    for(int warpIdx = 0; warpIdx < WarpCount; ++warpIdx) {
        warpOpOffsets[warpIdx] = warpOpData.size();
        for(uint32_t op : warpOps[warpIdx]) {
            warpOpData.push_back(op);
        }
        while(warpOpData.size() & 31) {
            warpOpData.push_back(0);
        }
    }
    for(int i = 0; i < 32; ++i) {
        warpOpData.push_back(0);
    }

    GPUArray<uint32_t> warpOpDataDev = vecToGPU(warpOpData);

    vector<const uint32_t*> warpOpStarts;
    for(int offset : warpOpOffsets) {
        warpOpStarts.push_back(warpOpDataDev.get() + offset);
    }

    GPUArray<const uint32_t*> warpOpStartsDev = vecToGPU(warpOpStarts);

    vector<int> warpRandsPerIteration(WarpCount);
    for(int warpIdx = 0; warpIdx < WarpCount; ++warpIdx) {
        int count = 0;
        for(uint32_t op : warpOps[warpIdx]) {
            if((op & 0xF000) == 0x8000) {
                ++count;
            }
            if(((op >> 16) & 0xF000) == 0x8000) {
                ++count;
            }
        }
        warpRandsPerIteration[warpIdx] = count;
    }

    GPUArray<int> warpRandsPerIterationDev = vecToGPU(warpRandsPerIteration);

    TPAState tpaState;
    tpaState.prelimWalkCount = prelimWalkCount;
    tpaState.finalWalkCountTable = finalWalkCountTableDev.get();
    tpaState.finalWalkCountTableSize = finalWalkCountTable.size();
    tpaState.finalWalkCountTarget = -1;
    tpaState.finalWalkCount = -1;
    tpaState.prelimResults = 0;
    tpaState.nextWalkIdx = initialFinalWalkCount + prelimWalkCount - 1;
    tpaState.finalHitCount = 0;
    tpaState.totalIterCount = 0;
    
    GPUStruct<TPAState> tpaStateDev = structToGPU(tpaState);

    GPUArray<int> vertPosBufDev;
    if(useSharedMem) {
        CUDACHECK(cudaFuncSetAttribute(
            kernel_shared,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            vertPosSizePerBlock * sizeof(int)
        ));
    } else {
        CUDACHECK(cudaFuncSetAttribute(
            kernel_global,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxL1
        ));

        vertPosBufDev = createUninitializedGPUArray<int>(vertPosSizePerBlock * blockCount);
    }

    uint64_t seed = UnifInt<uint64_t>()(rng);

    msg("PARALLEL_TPA_START");

    msg("GPU_SAMPLE_KERNEL_BLOCK_COUNT ", blockCount);
    msg("GPU_SAMPLE_KERNEL_BLOCK_SIZE ", 32 * WarpCount);
    msg("GPU_SAMPLE_KERNEL_START");

    if(useSharedMem) {
        kernel_shared<<<blockCount, 32 * WarpCount, vertPosSizePerBlock * sizeof(int)>>>(
            n,
            seed,
            warpOpStartsDev.get(),
            warpRandsPerIterationDev.get(),
            tpaStateDev.get()
        );
    } else {
        kernel_global<<<blockCount, 32 * WarpCount>>>(
            n,
            seed,
            warpOpStartsDev.get(),
            warpRandsPerIterationDev.get(),
            tpaStateDev.get(),
            vertPosBufDev.get()
        );
    }

    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());
    msg("GPU_SAMPLE_KERNEL_END");

    structFromGPU(tpaStateDev, tpaState);

    msg("PARALLEL_TPA_END");
    msg("PARALLEL_TPA_PRELIM_ESTIMATE ", tpaState.prelimEstimate());
    msg("PARALLEL_TPA_FINAL_WALK_COUNT_TARGET ", tpaState.finalWalkCountTarget);
    msg("PARALLEL_TPA_FINAL_WALK_COUNT ", tpaState.finalWalkCount);
    msg("PARALLEL_TPA_FINAL_ESTIMATE ", tpaState.finalEstimate());
    msg("TOTAL_ITER_COUNT ", tpaState.totalIterCount);
    msg("LINEXT_LOG_COUNT ", relaxationLogCount - tpaState.finalEstimate());
}
