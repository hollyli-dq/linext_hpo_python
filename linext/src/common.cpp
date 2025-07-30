#include "common.hpp"

thread_local mt19937 rng(random_device{}());

const clock_t StartCPUTime = clock();
const chrono::steady_clock::time_point StartWallTime = chrono::steady_clock::now();

static int initHardwareThreadCount() {
    int ret = thread::hardware_concurrency();
    if(ret <= 0 || ret > 65536) {
        fail("Determining the number of hardware threads failed");
    }
    return ret;
}
const int HardwareThreadCount = initHardwareThreadCount();

mutex outputMutex;
