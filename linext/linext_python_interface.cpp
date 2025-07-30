/*
 * Python interface for linext C++ library
 * Provides C-style functions that can be called via ctypes
 */

#include "src/exactcount.hpp"
#include "src/poset.hpp"
#include <vector>
#include <cmath>

// Forward declarations for linext methods
template <int W>
void method_armc(const Poset<W>& poset, double epsilon, double delta);

template <int W>
void method_relaxtpa(const Poset<W>& poset, double epsilon, double delta);

template <int W>
void method_relaxtpa_loose1(const Poset<W>& poset, double epsilon, double delta);

template <int W>
void method_telescope_basic_gibbs(const Poset<W>& poset, double epsilon, double delta);

extern "C" {
    
// C interface function for Python ctypes
long count_linear_extensions(int** matrix, int size) {
    try {
        // Handle edge cases
        if (size <= 0) return 1;
        if (size == 1) return 1;
        if (size > 64) return -2;  // Too large for W=1
        
        // Create poset with template parameter W=1 (supports up to 64 nodes)
        Poset<1> poset(size);
        
        // Add edges from adjacency matrix
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (matrix[i][j] == 1) {
                    poset.add(i, j);  // Add edge from i to j
                }
            }
        }
        
        // Create memory limit (8GB for larger problems)
        ExactCounterGlobalMemoryLimit memLimit(8ULL * 1024 * 1024 * 1024);
        
        // Count linear extensions using exact algorithm
        double logResult = computeExactLinextCount<1>(poset, memLimit);
        
        // Convert from log to actual count
        if (std::isinf(logResult)) {
            return -3;  // Overflow
        }
        
        long result = static_cast<long>(std::exp(logResult) + 0.5);
        return result;
        
    } catch (...) {
        return -1;  // Error indicator
    }
}

// Alternative interface with flattened matrix (easier for Python)
long count_linear_extensions_flat(int* matrix_flat, int size) {
    try {
        // Handle edge cases
        if (size <= 0) return 1;
        if (size == 1) return 1;
        if (size > 64) return -2;  // Too large for W=1
        
        // Create poset with template parameter W=1 (supports up to 64 nodes)
        Poset<1> poset(size);
        
        // Add edges from flattened adjacency matrix
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (matrix_flat[i * size + j] == 1) {
                    poset.add(i, j);  // Add edge from i to j
                }
            }
        }
        
        // Create memory limit (8GB for larger problems)
        ExactCounterGlobalMemoryLimit memLimit(8ULL * 1024 * 1024 * 1024);
        
        // Count linear extensions using exact algorithm
        double logResult = computeExactLinextCount<1>(poset, memLimit);
        
        // Convert from log to actual count
        if (std::isinf(logResult)) {
            return -3;  // Overflow
        }
        
        long result = static_cast<long>(std::exp(logResult) + 0.5);
        return result;
        
    } catch (...) {
        return -1;  // Error
    }
}

// ARMC method with approximation guarantees
long count_linear_extensions_armc(int* matrix_flat, int size, double epsilon, double delta) {
    try {
        // Handle edge cases
        if (size <= 0) return 1;
        if (size == 1) return 1;
        if (size > 64) return -2;  // Too large for W=1
        
        // Create poset with template parameter W=1
        Poset<1> poset(size);
        
        // Add edges from flattened adjacency matrix
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (matrix_flat[i * size + j] == 1) {
                    poset.add(i, j);
                }
            }
        }
        
        // Create memory limit (8GB)
        ExactCounterGlobalMemoryLimit memLimit(8ULL * 1024 * 1024 * 1024);
        
        // Call ARMC method
        method_armc<1>(poset, epsilon, delta);
        
        // Note: ARMC outputs via msg() to stderr, not return value
        // For Python interface, we need to capture stderr output
        // This is a simplified version - in practice, you'd need to redirect stderr
        return -4;  // Indicate ARMC method needs stderr parsing
        
    } catch (...) {
        return -1;  // Error
    }
}

// Relaxation-based methods
long count_linear_extensions_relaxtpa(int* matrix_flat, int size, double epsilon, double delta) {
    try {
        if (size <= 0) return 1;
        if (size == 1) return 1;
        if (size > 64) return -2;
        
        Poset<1> poset(size);
        
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (matrix_flat[i * size + j] == 1) {
                    poset.add(i, j);
                }
            }
        }
        
        // Call relaxtpa method - note this also outputs via msg()
        method_relaxtpa<1>(poset, epsilon, delta);
        return -4;  // Indicate needs stderr parsing
        
    } catch (...) {
        return -1;
    }
}

long count_linear_extensions_relaxtpa_loose1(int* matrix_flat, int size, double epsilon, double delta) {
    try {
        if (size <= 0) return 1;
        if (size == 1) return 1;
        if (size > 64) return -2;
        
        Poset<1> poset(size);
        
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (matrix_flat[i * size + j] == 1) {
                    poset.add(i, j);
                }
            }
        }
        
        method_relaxtpa_loose1<1>(poset, epsilon, delta);
        return -4;  // Indicate needs stderr parsing
        
    } catch (...) {
        return -1;
    }
}

// Telescope sampling methods  
long count_linear_extensions_telescope_gibbs(int* matrix_flat, int size, double epsilon, double delta) {
    try {
        if (size <= 0) return 1;
        if (size == 1) return 1;
        if (size > 64) return -2;
        
        Poset<1> poset(size);
        
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (matrix_flat[i * size + j] == 1) {
                    poset.add(i, j);
                }
            }
        }
        
        method_telescope_basic_gibbs<1>(poset, epsilon, delta);
        return -4;  // Indicate needs stderr parsing
        
    } catch (...) {
        return -1;
    }
}

} // extern "C" 