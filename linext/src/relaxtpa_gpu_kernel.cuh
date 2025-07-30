__global__ void __launch_bounds__(32 * WarpCount, 1)
#ifdef VERT_POS_USE_SHARED_MEM
    kernel_shared
#else
    kernel_global
#endif
(
    int n,
    uint64_t seed,
    const uint32_t** warpOpStarts,
    const int* warpRandsPerIteration,
    TPAState* tpaState
#ifndef VERT_POS_USE_SHARED_MEM
    , int* globalVertPosBuf
#endif
) {
    // INITIALIZATION

    const int warpIdx = threadIdx.x >> 5;
    const int laneIdx = threadIdx.x & 31;
    const bool leftSide = !(threadIdx.x & 1);
    const bool primary = leftSide && !warpIdx;

    const uint32_t* const ops = warpOpStarts[warpIdx];

    PCG32 pcg32(seed, (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x);
    const PCG32Step initialBackStep =
        pcg32.backwardStep(InitialIterationCount * warpRandsPerIteration[warpIdx]);
    pcg32.skip(initialBackStep);

#ifdef VERT_POS_USE_SHARED_MEM
    extern __shared__ int sharedVertPosBuf[];
    int* const blockVertPosBuf = sharedVertPosBuf;
#else
    int* const blockVertPosBuf = globalVertPosBuf + 32 * max(n, 1) * blockIdx.x;
#endif
    int* const vertPosBuf = blockVertPosBuf + laneIdx;
    int* const pairVertPosBuf = blockVertPosBuf + (laneIdx ^ 1);

    const int initCoord = leftSide ? 0 : MaxCoord;

    int beta = MaxCoord;

    // FUNCTIONS

    auto vertPos = [&](int v) -> int& {
        return vertPosBuf[v << 5];
    };
    auto pairVertPos = [&](int v) -> int& {
        return pairVertPosBuf[v << 5];
    };

    auto initVertPosCoopDirty = [&]() {
        for(int v = warpIdx; v < n; v += WarpCount) {
            vertPos(v) = initCoord;
        }
    };

    auto isCoupledCoop = [&]() -> bool {
        int orig;
        if(!warpIdx) {
            orig = vertPos(0);
        }

        bool isCoupled1 = true;
        bool isCoupled2 = true;
        int v1 = warpIdx;
        int v2 = warpIdx + WarpCount;
        while(v2 < n) {
            int a1 = vertPos(v1);
            int b1 = pairVertPos(v1);
            int a2 = vertPos(v2);
            int b2 = pairVertPos(v2);

            isCoupled1 = isCoupled1 && a1 == b1;
            isCoupled2 = isCoupled2 && a2 == b2;

            v1 += 2 * WarpCount;
            v2 += 2 * WarpCount;
        }
        if(v1 < n) {
            isCoupled1 = isCoupled1 && vertPos(v1) == pairVertPos(v1);
        }
        bool isCoupled = isCoupled1 && isCoupled2;
        __syncthreads();

        if(!warpIdx) {
            vertPos(0) = 1;
        }
        __syncthreads();

        if(!isCoupled) {
            // Should be OK as all writing threads write the same value
            vertPos(0) = 0;
        }
        __syncthreads();

        isCoupled = (bool)vertPos(0);
        __syncthreads();

        if(!warpIdx) {
            vertPos(warpIdx) = orig;
        }
        __syncthreads();

        return isCoupled;
    };

    auto updateBetaCoopDestructive = [&](bool sel) {
        const uint32_t* opPtr = ops;

        uint32_t myOp = __ldg(&opPtr[laneIdx]);
        opPtr += 32;
        int opIdx = 0;

        uint32_t op[2];
        auto getOps = [&]() {
            op[0] = __shfl_sync((unsigned)-1, myOp, opIdx++);
            op[1] = __shfl_sync((unsigned)-1, myOp, opIdx++);
            if(opIdx == 32) {
                myOp = __ldg(&opPtr[laneIdx]);
                opIdx = 0;
                opPtr += 32;
            }
        };

        int newBeta = 0;
        int maxSoftPred = 0;

        while(true) {
            getOps();
            if(!op[0] && op[1]) {
                break;
            } else {
                int read1[2];
                int read2[2];

                for(int r = 0; r < 2; ++r) {
                    if((op[r] & 0x0000'F000u) >= 0x0000'7000u) {
                        read1[r] = *(int*)((char*)vertPosBuf + ((op[r] << 7) & 0x7FF80u));
                    }
                    if(op[r] >= 0x7000'0000u) {
                        read2[r] = *(int*)((char*)vertPosBuf + ((op[r] >> 9) & 0x7FF80u));
                    }
                }

                for(int r = 0; r < 2; ++r) {
                    if(op[r] & 0x0000'8000u) {
                        newBeta = max(newBeta, maxSoftPred - read1[r]);
                        maxSoftPred = 0;
                    }
                    if((op[r] & 0x0000'F000u) == 0x0000'7000u) {
                        maxSoftPred = max(maxSoftPred, read1[r]);
                    }
                    if(op[r] & 0x8000'0000u) {
                        newBeta = max(newBeta, maxSoftPred - read2[r]);
                        maxSoftPred = 0;
                    }
                    if((op[r] & 0xF000'0000u) == 0x7000'0000u) {
                        maxSoftPred = max(maxSoftPred, read2[r]);
                    }
                }
            }
        }
        __syncthreads();

        if(sel && !warpIdx) {
            vertPos(0) = newBeta;
        }
        __syncthreads();

        if(sel && warpIdx) {
            atomicMax(&vertPos(0), newBeta);
        }
        __syncthreads();

        if(sel) {
            beta = vertPos(0);
        }
        __syncthreads();
    };

    auto iterationCoop = [&]() {
        const uint32_t* opPtr = ops;

        uint32_t myOp = __ldg(&opPtr[laneIdx]);
        opPtr += 32;
        int opIdx = 0;

        uint32_t op[2];
        auto getOps = [&]() {
            op[0] = __shfl_sync((unsigned)-1, myOp, opIdx++);
            op[1] = __shfl_sync((unsigned)-1, myOp, opIdx++);
            if(opIdx == 32) {
                myOp = __ldg(&opPtr[laneIdx]);
                opIdx = 0;
                opPtr += 32;
            }
        };
        
        int myRand;
        int maxHardPred = 0;
        int pos;
        while(true) {
            int minSoftSucc = MaxCoord;
            while(true) {
                getOps();
                if(!op[0]) {
                    if(op[1] == 2) {
                        goto done;
                    }
                    break;
                }

                int* vertPos1[2];
                int* vertPos2[2];

                int read1[2];
                int read2[2];

                for(int r = 0; r < 2; ++r) {
                    vertPos1[r] = (int*)((char*)vertPosBuf + ((op[r] << 7) & 0x7FF80u));
                    vertPos2[r] = (int*)((char*)vertPosBuf + ((op[r] >> 9) & 0x7FF80u));
                    
                    if(op[r] & 0x0000'4000u) {
                        read1[r] = *vertPos1[r];
                    }
                    if(op[r] & 0x4000'0000u) {
                        read2[r] = *vertPos2[r];
                    }
                }

                for(int r = 0; r < 2; ++r) {
                    if(op[r] & 0x0000'2000u) {
                        read1[r] -= beta;
                    }
                    if(op[r] & 0x2000'0000u) {
                        read2[r] -= beta;
                    }

                    if(op[r] & 0x0000'8000u) {
                        if(op[r] & 0x0000'1000u) {
                            pos = __shfl_up_sync((unsigned)-1, myRand, 1, 2);
                        } else {
                            myRand = (int)(pcg32.genGPU() >> 1);
                            pos = __shfl_down_sync((unsigned)-1, myRand, 1, 2);
                        }

                        if(
                            maxHardPred < pos &&
                            pos - beta < minSoftSucc
                        ) {
                            *vertPos1[r] = pos;
                        }
                        maxHardPred = 0;
                        minSoftSucc = MaxCoord;
                    } else {
                        // Either (op[r] & 0x0000'4000u) or we're past-the-end
                        // and we set minSoftSucc to garbage
                        if(op[r] & 0x0000'1000u) {
                            maxHardPred = max(maxHardPred, read1[r]);
                        } else {
                            minSoftSucc = min(minSoftSucc, read1[r]);
                        }
                    }

                    if(op[r] & 0x8000'0000u) {
                        if(op[r] & 0x1000'0000u) {
                            pos = __shfl_up_sync((unsigned)-1, myRand, 1, 2);
                        } else {
                            myRand = (int)(pcg32.genGPU() >> 1);
                            pos = __shfl_down_sync((unsigned)-1, myRand, 1, 2);
                        }

                        if(
                            maxHardPred < pos &&
                            pos - beta < minSoftSucc
                        ) {
                            *vertPos2[r] = pos;
                        }
                        maxHardPred = 0;
                        minSoftSucc = MaxCoord;
                    } else {
                        // Either (op[r] & 0x4000'0000u) or we're past-the-end
                        // and we set minSoftSucc to garbage
                        if(op[r] & 0x1000'0000u) {
                            maxHardPred = max(maxHardPred, read2[r]);
                        } else {
                            minSoftSucc = min(minSoftSucc, read2[r]);
                        }
                    }
                }
            }
            __syncthreads();
        }
        done: __syncthreads();
    };

    // MAIN KERNEL
    
    bool prelim;
    bool active;

    if(primary) {
        active = tpaState->popWalkJob(prelim);
    }

    int hitCount = 0;
    PCG32Step backStep = initialBackStep;
    int iterCount = InitialIterationCount;
    int itersLeft = InitialIterationCount;
    pcg32.skip(backStep);

    initVertPosCoopDirty();
    __syncthreads();

    ll totalIterCount = 0;

    while(true) {
        int itersToRun = itersLeft;
        itersToRun = min(itersToRun, __shfl_xor_sync((unsigned)-1, itersToRun, 16));
        itersToRun = min(itersToRun, __shfl_xor_sync((unsigned)-1, itersToRun, 8));
        itersToRun = min(itersToRun, __shfl_xor_sync((unsigned)-1, itersToRun, 4));
        itersToRun = min(itersToRun, __shfl_xor_sync((unsigned)-1, itersToRun, 2));

        for(int iterIdx = 0; iterIdx < itersToRun; ++iterIdx) {
            iterationCoop();
        }
        itersLeft -= itersToRun;
        totalIterCount += itersToRun;

        bool isCoupled = isCoupledCoop();
        isCoupled = isCoupled && !itersLeft;

        if(__any_sync((unsigned)-1, isCoupled)) {
            updateBetaCoopDestructive(isCoupled);
        }

        if(!itersLeft) {
            if(isCoupled) {
                pcg32.skip(backStep);

                if(beta) {
                    ++hitCount;
                } else {
                    if(primary && active) {
                        tpaState->pushWalkResult(prelim, hitCount);
                        active = tpaState->popWalkJob(prelim);
                    }
                    
                    beta = MaxCoord;
                    hitCount = 0;
                }

                backStep = initialBackStep;
                iterCount = InitialIterationCount;
            } else {
                backStep = backStep + backStep;
                iterCount += iterCount;
            }

            pcg32.skip(backStep);
        }
        if(!itersLeft) {
            itersLeft = iterCount;
            initVertPosCoopDirty();
        }
        if(!warpIdx && __all_sync((unsigned)-1, !active || !leftSide)) {
            // No data race, because vertPos(0) belongs to this warp in initVertPosCoopDirty
            vertPos(0) = -1;
        }
        __syncthreads();
        if(vertPos(0) == -1) {
            break;
        }
        __syncthreads();
    }

    if(primary) {
        signedAtomicAdd(tpaState->totalIterCount, totalIterCount);
    }
}
