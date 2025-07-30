#pragma once

#include "common.hpp"

const int WarpCount = 32;

void runRelaxTpaGPU(
    int n,
    double relaxationLogCount,
    const vector<ll>& finalWalkCountTable,
    const vector<vector<uint32_t>>& warpOps,
    ll prelimWalkCount
);
