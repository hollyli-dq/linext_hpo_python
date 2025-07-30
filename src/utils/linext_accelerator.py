"""
Comprehensive LinextAccelerator with exact and approximation methods.
Provides ultra-fast NLE computation using multiple algorithms.
"""

import ctypes
import numpy as np
import os
import tempfile
import subprocess
import time
from typing import Optional, Union, Dict, Any
from enum import Enum

class LinextMethod(Enum):
    """Available linext algorithms."""
    EXACT = "exact"                           # Exact counting (slow but precise)
    ARMC = "armc"                            # Adaptive Random Monte Carlo (fast approximation) 
    RELAXTPA = "relaxtpa"                    # Relaxed Two-Phase Algorithm (balanced)
    RELAXTPA_LOOSE1 = "relaxtpa_loose1"      # Loose variant 1 (faster)
    RELAXTPA_LOOSE2 = "relaxtpa_loose2"      # Loose variant 2 (faster)
    RELAXTPA_AVX2 = "relaxtpa_avx2"          # AVX2 vectorized (fastest on compatible CPUs)
    TELESCOPE_SWAP = "telescope_basic_swap"   # Telescope with swap
    TELESCOPE_GIBBS = "telescope_basic_gibbs" # Telescope with Gibbs
    AUTO = "auto"                            # Automatic method selection

class LinextAccelerator:
    """
    High-performance C++ LinExt interface supporting exact and approximation methods.
    Provides 10-1000x speedup over Python implementations using C++ acceleration.
    Requires the LinExt C++ library to be compiled.
    """
    
    def __init__(self, linext_path: Optional[str] = None, default_method: LinextMethod = LinextMethod.AUTO):
        """
        Initialize LinextAccelerator with method selection.
        
        Args:
            linext_path: Path to linext binary
            default_method: Default algorithm to use
        """
        self.linext_path = self._find_linext_binary(linext_path)
        self.default_method = default_method
        self.available = self.linext_path is not None
        
        # Performance cache for method selection
        self._method_cache: Dict[int, LinextMethod] = {}
        
        if self.available:
            print(f"✅ LinextAccelerator initialized: {self.linext_path}")
            print(f"🚀 Default method: {default_method.value}")
            print(f"⚡ C++ acceleration: ENABLED")
        else:
            print("❌ LinextAccelerator binary not found!")
            print("🔧 Please build the C++ library: cd linext && make -f Makefile.shared")
    
    def _find_linext_binary(self, linext_path: Optional[str]) -> Optional[str]:
        """Find linext binary in possible locations."""
        if linext_path and os.path.exists(linext_path):
            return linext_path
            
        possible_paths = [
            './linext/linext',
            '../linext/linext', 
            '../../linext/linext',
            'linext/linext',
            'linext',
            '/usr/local/bin/linext'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None
    
    def nle(self, 
            adj_matrix: np.ndarray, 
            method: Optional[LinextMethod] = None,
            epsilon: float = 0.01,
            delta: float = 0.01) -> int:
        """
        Compute NLE using specified method.
        
        Args:
            adj_matrix: Adjacency matrix of partial order
            method: Algorithm to use (None = use default/auto)
            epsilon: Approximation accuracy parameter (smaller = more accurate)
            delta: Confidence parameter (smaller = higher confidence)
            
        Returns:
            Number of linear extensions
        """
        if not self.available:
            raise RuntimeError("C++ LinExt library not available. Please build the C++ library first.")
            
        if adj_matrix.size == 0 or len(adj_matrix.shape) != 2:
            return 1
            
        n = adj_matrix.shape[0]
        if n <= 1:
            return 1
            
        # Select optimal method
        selected_method = self._select_method(method, n)
        
        # Use linext binary for all matrix sizes and methods
        return self._call_linext_binary(adj_matrix, selected_method, epsilon, delta)
    
    def nle_exact(self, adj_matrix: np.ndarray) -> int:
        """Compute exact NLE (guaranteed correctness)."""
        return self.nle(adj_matrix, LinextMethod.EXACT)
    
    def nle_fast(self, adj_matrix: np.ndarray, epsilon: float = 0.1, delta: float = 0.1) -> int:
        """
        Compute fast approximate NLE (good for large matrices).
        
        Args:
            adj_matrix: Adjacency matrix of the DAG
            epsilon: Relative error bound (0.1 = 10% error)
            delta: Confidence parameter (0.1 = 90% confidence)
        """
        return self.nle(adj_matrix, LinextMethod.RELAXTPA_LOOSE1, epsilon=epsilon, delta=delta)
    
    def nle_balanced(self, adj_matrix: np.ndarray, epsilon: float = 0.01, delta: float = 0.05) -> int:
        """
        Compute balanced NLE (good speed/accuracy tradeoff).
        
        Args:
            adj_matrix: Adjacency matrix of the DAG
            epsilon: Relative error bound (0.01 = 1% error)
            delta: Confidence parameter (0.05 = 95% confidence)
        """
        return self.nle(adj_matrix, LinextMethod.RELAXTPA, epsilon=epsilon, delta=delta)
    
    def nle_ultra_fast(self, adj_matrix: np.ndarray, epsilon: float = 0.2, delta: float = 0.1) -> int:
        """
        Compute ultra-fast approximate NLE (fastest possible).
        
        Args:
            adj_matrix: Adjacency matrix of the DAG
            epsilon: Relative error bound (0.2 = 20% error)
            delta: Confidence parameter (0.1 = 90% confidence)
        """
        return self.nle(adj_matrix, LinextMethod.ARMC, epsilon=epsilon, delta=delta)
    
    def _select_method(self, method: Optional[LinextMethod], matrix_size: int) -> LinextMethod:
        """Select optimal method based on matrix size and requirements."""
        if method and method != LinextMethod.AUTO:
            return method
            
        if method is None:
            method = self.default_method
            
        if method != LinextMethod.AUTO:
            return method
            
        # Cache-based selection for repeated similar sizes
        if matrix_size in self._method_cache:
            return self._method_cache[matrix_size]
            
        # Auto-selection based on matrix size (all methods use C++ acceleration)
        if matrix_size <= 8:
            selected = LinextMethod.EXACT  # C++ exact method for small DAGs
        elif matrix_size <= 15:
            selected = LinextMethod.RELAXTPA  # Balanced approximation
        elif matrix_size <= 25:
            selected = LinextMethod.RELAXTPA_LOOSE1  # Fast approximation
        else:
            selected = LinextMethod.ARMC  # Ultra-fast approximation
            
        self._method_cache[matrix_size] = selected
        return selected
    
    def _call_linext_binary(self, 
                           adj_matrix: np.ndarray, 
                           method: LinextMethod, 
                           epsilon: float, 
                           delta: float) -> int:
        """Call linext binary with specified method."""
        # Write matrix to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for row in adj_matrix:
                f.write(' '.join(map(str, row.astype(int))) + '\n')
            temp_file = f.name
        
        try:
            # Convert method name (Python enum uses underscores, linext uses dashes)
            method_name = method.value.replace('_', '-')
            
            # Call linext binary
            result = subprocess.run([
                self.linext_path,
                temp_file,
                method_name,
                str(epsilon),
                str(delta)
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                # Parse LINEXT_LOG_COUNT from stderr
                for line in result.stderr.split('\n'):
                    if 'LINEXT_LOG_COUNT' in line:
                        try:
                            log_count = float(line.split()[-1])
                            nle_count = int(round(np.exp(log_count)))
                            return max(1, nle_count)
                        except (ValueError, OverflowError):
                            continue
                            
            # If parsing failed, fall back
            raise RuntimeError(f"Linext parsing failed: {result.stderr}")
            
        finally:
            try:
                os.unlink(temp_file)
            except OSError:
                pass
    
    def benchmark_methods(self, adj_matrix: np.ndarray, methods: Optional[list] = None) -> Dict[str, Any]:
        """
        Benchmark different methods on the given matrix.
        
        Returns:
            Dictionary with timing and accuracy results
        """
        if methods is None:
            methods = [LinextMethod.EXACT, LinextMethod.RELAXTPA, LinextMethod.ARMC]
            
        results = {}
        n = adj_matrix.shape[0]
        
        print(f"🧪 Benchmarking methods on {n}x{n} matrix...")
        
        for method in methods:
            try:
                start_time = time.time()
                result = self.nle(adj_matrix, method)
                elapsed = time.time() - start_time
                
                results[method.value] = {
                    'result': result,
                    'time_ms': elapsed * 1000,
                    'success': True
                }
                
                print(f"  {method.value:20s}: {result:8d} ({elapsed*1000:6.2f}ms)")
                
            except Exception as e:
                results[method.value] = {
                    'result': None,
                    'time_ms': None,
                    'success': False,
                    'error': str(e)
                }
                print(f"  {method.value:20s}: FAILED ({e})")
        
        return results

# Global instance
_linext_accelerator = None

def get_linext_accelerator(method: LinextMethod = LinextMethod.AUTO) -> LinextAccelerator:
    """Get singleton LinextAccelerator instance."""
    global _linext_accelerator
    if _linext_accelerator is None:
        _linext_accelerator = LinextAccelerator(default_method=method)
    return _linext_accelerator 