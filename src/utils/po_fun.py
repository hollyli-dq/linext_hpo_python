import numpy as np
import math
import sys
from scipy.stats import multivariate_normal,norm
from scipy.special import betaln, gammaln
from tabulate import tabulate
from typing import List, Dict, Tuple, Union

import networkx as nx
import random
import seaborn as sns
import pandas as pd  
from collections import Counter
import itertools
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any
from collections import defaultdict
from math import log, inf
from scipy.stats import beta,expon
from scipy.stats import gamma 


_COV_CACHE: Dict[Tuple[int, float, float], Tuple[np.ndarray, float, float]] = {}


class ConversionUtils:
    """
    Utility class for converting sequences and orders to different representations.
    """

    @staticmethod
    def seq2dag(seq: List[int], n: int) -> np.ndarray:
        """
        Converts a sequence to a directed acyclic graph (DAG) represented as an adjacency matrix.

        Parameters:
        - seq: A sequence (list) of integers representing a total order.
        - n: Total number of elements.

        Returns:
        - adj_matrix: An n x n numpy array representing the adjacency matrix of the DAG.
        """
        adj_matrix = np.zeros((n, n), dtype=int)
        for i in range(len(seq)):
            u = seq[i] - 1  # Convert to 0-based index
            for j in range(i + 1, len(seq)):
                v = seq[j] - 1  # Convert to 0-based index
                adj_matrix[u, v] = 1
        return adj_matrix

    @staticmethod
    def order2partial(v: List[List[int]], n: Optional[int] = None) -> np.ndarray:
        """
        Computes the intersection of the transitive closures of a list of total orders.

        Parameters:
        - v: List of sequences, where each sequence is a list of integers representing a total order.
        - n: Total number of elements (optional).

        Returns:
        - result_matrix: An n x n numpy array representing the adjacency matrix of the partial order.
        """
        if n is None:
            n = max(max(seq) for seq in v)
        z = np.zeros((n, n), dtype=int)
        for seq in v:
            dag_matrix = ConversionUtils.seq2dag(seq, n)
            closure_matrix = BasicUtils.transitive_closure(dag_matrix)
            z += closure_matrix
        result_matrix = (z == len(v)).astype(int)
        return result_matrix



    
class GenerationUtils:
    """
    Utility class for generating latent positions, partial orders, random partial orders, 
    linear extensions, total orders, and subsets.
    """
    @staticmethod
    def generate_choice_sets_for_assessors(
        M_a_dict: Dict[int, List[int]],
        min_tasks: int = 1,
        min_size: int = 2
    ) -> Dict[int, List[List[int]]]:
        """
        Generate a dictionary of choice sets O_{a,i} for each assessor.
        
        Each assessor a is assigned a random number of tasks (choice sets) between
        min_tasks and max_tasks. For each task, a random subset of items (of size at least
        min_size and at most the total number of items in M_a) is selected from the assessor's M_a.
        
        Parameters:
            M_a_dict : Dict[int, List[int]]
                Dictionary mapping assessor IDs to their overall list of item IDs.
            min_tasks : int, optional
                Minimum number of tasks per assessor (default is 1).
            max_tasks : int, optional
                Maximum number of tasks per assessor (default is 3).
            min_size : int, optional
                Minimum number of items in a choice set (default is 2).
        
        Returns:
            Dict[int, List[List[int]]]: Dictionary where each key is an assessor ID and each value is a list
                                        of choice sets (each choice set is a list of item IDs).
        """
        O_a_i_dict = {}
        for assessor, items in M_a_dict.items():
            num_items = len(items)
            max_tasks = 10*num_items
            
            # FIXED: Ensure max_tasks >= min_tasks to avoid ValueError
            if max_tasks < min_tasks:
                max_tasks = min_tasks  # Set max_tasks to at least min_tasks
                
            # Determine the number of tasks for this assessor.
            num_tasks = random.randint(min_tasks, max_tasks)
            tasks = []
            for _ in range(num_tasks):
                # Choose a task size: at least min_size, at most all available items.
                task_size = random.randint(min_size, num_items)
                task = sorted(random.sample(items, task_size))
                tasks.append(task)
            O_a_i_dict[assessor] = tasks
        return O_a_i_dict

    @staticmethod
    def generate_latent_positions(n: int, K: int, rho: float) -> np.ndarray:
        """
        Generates latent positions Z for n items in K dimensions with correlation rho.

        Parameters:
        - n: Number of items.
        - K: Number of dimensions.
        - rho: Correlation coefficient between dimensions.

        Returns:
        - Z: An n x K numpy array of latent positions.
        """
        Sigma = BasicUtils.build_Sigma_rho(K,rho)
        mu = np.zeros(K)
        rv = multivariate_normal(mean=mu, cov=Sigma)
        Z = rv.rvs(size=n)
        if K == 1:
            Z = Z.reshape(n, 1)
        return Z

    @staticmethod
    def generate_random_PO(n: int) -> nx.DiGraph:
        """
        Generates a random partial order (directed acyclic graph) with `n` nodes.
        Ensures there are no cycles in the generated graph.

        Parameters:
        - n: Number of nodes in the partial order.

        Returns:
        - h: A NetworkX DiGraph representing the partial order.
        """
        h = nx.DiGraph()
        h.add_nodes_from(range(n))
        possible_edges = list(itertools.combinations(range(n), 2))
        random.shuffle(possible_edges)
        for u, v in possible_edges:
            if random.choice([True, False]):
                h.add_edge(u, v)
                if not nx.is_directed_acyclic_graph(h):
                    h.remove_edge(u, v)
        return h
    @staticmethod
    def generate_U(n: int, K: int, rho_val: float) -> np.ndarray:
        """
        Generate a latent variable matrix U of size n x K from a multivariate normal distribution
        with zero mean and a covariance matrix based on the given correlation rho_val.

        Parameters:
        - n: Number of observations.
        - K: Number of features.
        - rho_val: Correlation value for constructing the covariance matrix.

        Returns:
        - U: An n x K numpy array of latent positions.
        """
        K=int(K)
        cov = BasicUtils.build_Sigma_rho(K, rho_val)
        mean = np.zeros(K)
        U = np.random.multivariate_normal(mean, cov, size=n)
        return U
    @staticmethod
    def unifLE(tc: np.ndarray, elements: List[int], le: Optional[List[int]] = None) -> List[int]:
        """
        Sample a linear extension uniformly at random from the given partial order matrix `tc`.

        Parameters:
        - tc: Transitive closure matrix representing the partial order (numpy 2D array).
        - elements: List of elements corresponding to the current `tc` matrix.
        - le: List to build the linear extension (default: None).

        Returns:
        - le: A linear extension (list of elements in the original subset).
        """
        if le is None:
            le = []

        if len(elements) == 0:
            return le

        # Find the set of minimal elements (no incoming edges)
        indegrees = np.sum(tc, axis=0)
        minimal_elements_indices = np.where(indegrees == 0)[0]

        if len(minimal_elements_indices) == 0:
            raise ValueError("No minimal elements found. The partial order might contain cycles.")

        # Randomly select one of the minimal elements
        idx_in_tc = random.choice(minimal_elements_indices)
        element = elements[idx_in_tc]
        le.append(element)

        # Remove the selected element from the matrix and elements list
        tc_new = np.delete(np.delete(tc, idx_in_tc, axis=0), idx_in_tc, axis=1)
        elements_new = [e for i, e in enumerate(elements) if i != idx_in_tc]

        # Recursive call
        return GenerationUtils.unifLE(tc_new, elements_new, le)

    @staticmethod
    def sample_total_order(h: np.ndarray, subset: List[int]) -> List[int]:
        """
        Sample a total order (linear extension) for a restricted partial order.

        Parameters:
        - h: The original partial order adjacency matrix.
        - subset: List of node indices to sample a linear extension for.

        Returns:
        - sampled_order: A list representing the sampled linear extension.
        """
        # Restrict the matrix to the given subset
        restricted_matrix = BasicUtils.restrict_partial_order(h, subset)

        # Initialize elements as the elements in the subset
        elements = subset.copy()
        restricted_matrix_tc = BasicUtils.transitive_closure(restricted_matrix)

        # Sample one linear extension using the `unifLE` function
        sampled_order = GenerationUtils.unifLE(restricted_matrix_tc, elements)

        return sampled_order
    @staticmethod
    def topological_sort(adj_matrix: np.ndarray):
        """
        Returns one valid topological ordering of nodes in a DAG
        represented by an adjacency matrix.

        Parameters:
        - adj_matrix: n x n adjacency matrix (0/1),
                    where edge i->j means adj_matrix[i, j] == 1.

        Returns:
        - ordering: A list of node indices in topological order.
        
        Raises:
        - ValueError if the graph has a cycle or is not a DAG.
        """
        n = adj_matrix.shape[0]
        # in_degree[i] = number of incoming edges for node i
        in_degree = np.sum(adj_matrix, axis=0)

        # start with nodes that have no incoming edges
        queue = [i for i in range(n) if in_degree[i] == 0]
        ordering = []

        while queue:
            node = queue.pop()
            ordering.append(node)

            # "Remove" node from the graph => 
            # For each edge node->v, reduce in_degree[v] by 1
            for v in range(n):
                if adj_matrix[node, v] == 1:
                    in_degree[v] -= 1
                    # If v becomes a node with no incoming edges => add to queue
                    if in_degree[v] == 0:
                        queue.append(v)

        if len(ordering) != n:
            # A cycle must exist, or something prevented us from ordering all nodes
            raise ValueError("The adjacency matrix contains a cycle (not a DAG).")

        return ordering

    @staticmethod
    def generate_subsets(N: int, n: int) -> List[List[int]]:
        """
        Generate N subsets O1, O2, ..., ON where:
        - N is the number of subsets.
        - n is the size of the universal set {0, 1, ..., n-1}.
        
        Each subset Oi is created by:
        - Determining the subset size ni by uniformly sampling from [2, n].
        - Randomly selecting ni distinct elements from the set {0, 1, ..., n-1}.

        Parameters:
        - N: Number of subsets to generate.
        - n: Size of the universal set.

        Returns:
        - subsets: A list of subsets, each subset is a list of distinct integers.
        """
        subsets = []
        universal_set = list(range(n))  # Universal set from 0 to n-1

        for _ in range(N):
            # Randomly sample the subset size ni from [2, n]
            ni = random.randint(2, n)
            # Randomly select ni distinct elements from the universal set
            subset = random.sample(universal_set, ni)
            subset =sorted(subset)
            subsets.append(subset)

        return subsets
    @staticmethod
    def generate_total_orders_for_assessor(
        h_dict: Dict[int, np.ndarray],
        M_a_dict: Dict[int, List[int]],
        O_a_i_dict: Dict[int, List[List[int]]],
        prob_noise: float
    ) -> Dict[int, List[List[int]]]:
        """
        For each assessor, generate total orders (linear extensions) from their local partial order.
        
        Parameters:
        h_dict: Dictionary mapping assessor IDs to local partial order matrices (each of shape (|Mₐ|,|Mₐ|)).
        M_a_dict: Dictionary mapping assessor IDs to their ordered list of global item IDs.
                The order corresponds to the rows/columns in the local partial order matrix.
        O_a_i_dict: Dictionary mapping assessor IDs to a list of choice sets.
                    Each choice set is a list of global item IDs.
        prob_noise: The noise (jump) probability.
        
        Returns:
        Dict[int, List[List[int]]]: Mapping from assessor IDs to a list of total orders.
                                    Each total order is expressed as a list of global item IDs.
        """
        total_orders_dict = {}
        
        for a, choice_sets in O_a_i_dict.items():
            # Retrieve local partial order matrix.
            h_local = h_dict.get(a)
            if h_local is None:
                print(f"Warning: No partial order matrix found for assessor {a}. Skipping.")
                continue
            # Retrieve assessor's ordered global items.
            M_a = M_a_dict.get(a)
            if M_a is None:
                print(f"Warning: No item set found for assessor {a}. Skipping.")
                continue
            
            assessor_orders = []
            for subset in choice_sets:
                # Generate total order for this choice set.
                total_order = StatisticalUtils.generate_total_order_for_choice_set_with_queue_jump(subset, M_a, h_local, prob_noise)
                if total_order:
                    assessor_orders.append(total_order)
            total_orders_dict[a] = assessor_orders
        
        return total_orders_dict


class BasicUtils:
    """
    Utility class for basic operations on partial orders.
    """    
    @staticmethod
    def apply_transitive_reduction_hpo(h_U: dict) -> None:
        """
        For each key in h_U, if the value is a NumPy array, replace it with its transitive closure.
        If the value is a dictionary (e.g. assessor-level partial orders by task), then apply the
        operation to each matrix in that dictionary.
        
        This function modifies h_U in place.
        OPTIMIZED: Uses fast transitive reduction algorithm.
        """
        for key, value in h_U.items():
            if isinstance(value, dict):
                # If value is a dictionary, iterate over its keys
                for subkey, subval in value.items():
                    if isinstance(subval, np.ndarray):
                        value[subkey] = BasicUtils.transitive_reduction_optimized(subval)
            elif isinstance(value, np.ndarray):
                h_U[key] = BasicUtils.transitive_reduction_optimized(value)

    @staticmethod
    def build_Sigma_rho(K, rho_val: float) -> np.ndarray:
        mat = np.full((K, K), rho_val, dtype=float)
        np.fill_diagonal(mat, 1.0)
        return mat
    @staticmethod
    def generate_partial_order(Z):
        """
        Vectorised partial‑order generator.
        h[i,j] = 1  ⇔  Z[i] is strictly greater than Z[j] in **all** K dimensions.
        """
        # Z has shape (n, K)
        # Expand to (n, 1, K) and (1, n, K) for broadcasting
        greater = (Z[:, None, :] > Z[None, :, :])        # shape (n, n, K)
        h = np.all(greater, axis=2).astype(np.int8)       # collapse K‑axis
        np.fill_diagonal(h, 0)                            # ensure h[i,i] = 0
        return h
    
    @staticmethod
    def is_total_order(adj_matrix: np.ndarray) -> bool:

        n = adj_matrix.shape[0]
        # Compute transitive closure
        closure = BasicUtils.transitive_closure(adj_matrix)

        # For every pair (i,j), i != j, check comparability
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Must have either i->j or j->i
                    if not (closure[i, j] or closure[j, i]):
                        return False
    
        return True

    @staticmethod
    def restrict_partial_order(h: np.ndarray, subset: List[int]) -> np.ndarray:
        """
        Restrict the partial order matrix `h` to the given `subset`.

        Parameters:
        - h: The original partial order adjacency matrix.
        - subset: List of node indices to restrict to.

        Returns:
        - restricted_matrix: The adjacency matrix restricted to the subset.
        """
        subset_indices = subset  # Elements are already 0-based indices
        restricted_matrix = h[np.ix_(subset_indices, subset_indices)]
        return restricted_matrix

    @staticmethod
    def transitive_reduction(adj_matrix: np.ndarray) -> np.ndarray:
        """
        Computes the transitive reduction of a partial order represented by an transitive closure matrix.
        transitive reduction need to be computed based on the transitive closure matrix.
        Parameters:
        - adj_matrix: An n x n numpy array representing the adjacency matrix of the partial order.

        Returns:
        - tr: An n x n numpy array representing the adjacency matrix of the transitive reduction.
        """
        n = adj_matrix.shape[0]
        tr = adj_matrix.copy()
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if tr[i, k] and tr[k, j]:
                        tr[i, j] = 0
        return tr


    # ------------------------------------------------------------------ #
    @staticmethod
    def transitive_reduction_optimized(C):
        C = C.astype(bool, copy=False)
        n  = C.shape[0]
        if n <= 1:
            return C.astype(np.int8)

        # remove diagonal so a vertex cannot be its own intermediate
        B = C.copy()
        np.fill_diagonal(B, False)

        # Boolean matrix product — path of length ≥2
        P = (B.astype(np.int8) @ B.astype(np.int8)) > 0

        # keep only indispensable edges
        red = C & ~P
        np.fill_diagonal(red, False)
        return red.astype(np.int8)



    @staticmethod
    def transitive_closure(adj_matrix: np.ndarray) -> np.ndarray:
        """
        Computes the transitive closure of a relation represented by an adjacency matrix.

        Parameters:
        - adj_matrix: An n x n numpy array representing the adjacency matrix of the relation.

        Returns:
        - closure: An n x n numpy array representing the adjacency matrix of the transitive closure.
        """
        n = adj_matrix.shape[0]
        closure = adj_matrix.copy()
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    closure[i, j] = closure[i, j] or (closure[i, k] and closure[k, j])
        return closure

    @staticmethod
    def nle(tr: np.ndarray) -> int:
        """
        Counts the number of linear extensions of the partial order with transitive reduction `tr`.

        Parameters:
        - tr: An n x n numpy array representing the adjacency matrix of the transitive reduction.

        Returns:
        - count: An integer representing the number of linear extensions.
        """
        if tr.size == 0 or len(tr) == 1:
            return 1

        n = tr.shape[0]
        cs = np.sum(tr, axis=0)
        csi = (cs == 0)
        bs = np.sum(tr, axis=1)
        bsi = (bs == 0)
        free = np.where(bsi & csi)[0]
        k = len(free)

        if k == n:
            return math.factorial(n)

        if k > 0:
            # Delete free rows and columns
            tr = np.delete(np.delete(tr, free, axis=0), free, axis=1)
            fac = math.factorial(n) // math.factorial(n - k)
        else:
            fac = 1

        # Recompute cs and csi based on the updated tr
        cs = np.sum(tr, axis=0)
        csi = (cs == 0)
        bs = np.sum(tr, axis=1)
        bsi = (bs == 0)
        tops = np.where(csi)[0]
        bots = np.where(bsi)[0]

        # Special case: if n - k == 2, return fac
        if (n - k) == 2:
            return fac

        # Check for a unique top and bottom
        if len(tops) == 1 and len(bots) == 1:
            i = tops[0]
            j = bots[0]
            if i < tr.shape[0] and j < tr.shape[1]:
                trr = np.delete(np.delete(tr, [i, j], axis=0), [i, j], axis=1)
                return fac * BasicUtils.nle(trr)
            else:
                return 0  # Or handle appropriately

        # Iterate over all top elements
        count = 0
        for i in tops:
            if i >= tr.shape[0]:
                continue
            trr = np.delete(np.delete(tr, i, axis=0), i, axis=1)
            count += BasicUtils.nle(trr)

        return fac * count
    
    @staticmethod
    def compute_missing_relationships(
        h_true: np.ndarray,
        h_final: np.ndarray,
        index_to_item: Dict[int, int]
    ) -> List[Tuple[int, int]]:
        """
        Compute the missing relationships present in the true partial order but absent in the inferred one.

        Parameters
        ----------
        h_true : np.ndarray
            Adjacency matrix representing the true partial order.
        h_final : np.ndarray
            Adjacency matrix representing the inferred partial order.
        index_to_item : Dict[int, int]
            Mapping from matrix indices to items.

        Returns
        -------
        List[Tuple[int, int]]
            List of tuples (i, j) indicating missing relationships.
        """

        missing = []
        n = h_true.shape[0]
        h_true_reduced = BasicUtils.transitive_reduction_optimized(h_true)
        h_final_reduced = BasicUtils.transitive_reduction_optimized(h_final)
        for i in range(n):
            for j in range(n):
                if h_true_reduced[i, j] == 1 and h_final_reduced[i, j] == 0:
                    missing.append((index_to_item[i], index_to_item[j]))
        return missing



    @staticmethod
    def compute_redundant_relationships(
        h_true: np.ndarray,
        h_final: np.ndarray,
        index_to_item: Dict[int, int]
    ) -> List[Tuple[int, int]]:
        """
        Compute the redundant relationships present in the inferred partial order but absent in the true one.

        Parameters
        ----------
        h_true : np.ndarray
            Adjacency matrix representing the true partial order.
        h_final : np.ndarray
            Adjacency matrix representing the inferred partial order.
        index_to_item : Dict[int, int]
            Mapping from matrix indices to items.

        Returns
        -------
        List[Tuple[int, int]]
            List of tuples (i, j) indicating redundant relationships.
        """
        redundant = []
        n = h_true.shape[0]
        for i in range(n):
            for j in range(n):
                if h_true[i, j] == 0 and h_final[i, j] == 1:
                    redundant.append((index_to_item[i], index_to_item[j]))
        return redundant



    @staticmethod
    def find_tops(tr):
        """
        Identify all top elements (nodes with no incoming edges) in the partial order.

        Parameters:
        - tr: Adjacency matrix of the partial order (numpy.ndarray).

        Returns:
        - List of indices representing top elements.
        """
        incoming = np.sum(tr, axis=0)
        tops = [i for i, count in enumerate(incoming) if count == 0]
        return tops
    @staticmethod
    def num_extensions_with_first(tr, first_item_idx):
        """
        Compute how many linear extensions of the partial order `tr` start with the item `first_item_idx`.

        Parameters:
        - tr: Adjacency matrix of the partial order (numpy.ndarray).
        - first_item_idx: The index of the item we want to be the first in the linear extension.

        Returns:
        - int: The number of linear extensions of `tr` that start with `first_item_idx`.
        """
        # Identify top elements of the current poset
        tops = BasicUtils.find_tops(tr)
        
        # If first_item_idx is not a top element, no linear extension can start with it.
        if first_item_idx not in tops:
            return 0

        # If it is top, remove it from tr and count the nle of the reduced poset.
        tr_reduced = np.delete(np.delete(tr, first_item_idx, axis=0), first_item_idx, axis=1)
        return BasicUtils.nle(tr_reduced)

    @staticmethod
    def is_consistent(h: np.ndarray, observed_orders: List[List[int]]) -> bool:
        """
        Check if all observed orders are consistent with the partial order h.

        Parameters:
        - h: The partial order matrix (NumPy array).
        - observed_orders: List of observed total orders (each is a list of item indices).

        Returns:
        - True if all observed orders are consistent with h, False otherwise.
        """
        # Create a directed graph from the partial order matrix h
        G_PO = nx.DiGraph(h)
        # Compute the transitive closure to capture all implied precedence relations
        tc_PO = BasicUtils.transitive_closure(h)

        # Iterate over each observed order
        for idx, order in enumerate(observed_orders):
            # Create a mapping from item to its position in the observed order
            position = {item: pos for pos, item in enumerate(order)}

            # Check all edges in the transitive closure
            for u, v in zip(*np.where(tc_PO == 1)):
                # Check if both u and v are in the observed order
                if u in position and v in position:
                    # If u comes after v in the observed order, it's a conflict
                    if position[u] > position[v]:
                        return False  # Inconsistency found

        return True



    @staticmethod
    def generate_all_linear_extensions(h: np.ndarray, items: Optional[List[Any]] = None) -> List[List[Any]]:
        """
        Generate all linear extensions (i.e. valid total orders) of a partial order
        represented by the adjacency matrix h. Here, h is an n x n matrix where h[i, j] == 1
        means that index i must precede index j. The items are by default the indices [0,1,...,n-1],
        but if a list 'items' is provided, it will be used to map indices to actual items.

        Parameters:
            h: n x n numpy array representing the partial order.
            items: Optional list of items corresponding to the indices of h.
                If None, items are assumed to be [0, 1, ..., n-1].

        Returns:
            A list of linear extensions, each represented as a list of items (or indices if items is None).
        """
        n = h.shape[0]
        if items is None:
            items = list(range(n))        
        def _recursive_extensions(h_sub: np.ndarray, remaining: List[int]) -> List[List[int]]:
            # Base case: if no elements remain, return an empty extension.
            if not remaining:
                return [[]]
            
            m = len(remaining)
            # Compute in-degrees for the current submatrix.
            in_degree = [0] * m
            for i in range(m):
                for j in range(m):
                    if h_sub[i, j]:
                        in_degree[j] += 1
            
            # Minimal elements are those with in-degree zero.
            minimal_indices = [i for i, d in enumerate(in_degree) if d == 0]
            
            extensions = []
            for idx in minimal_indices:
                # 'current' is the actual index from the original set.
                current = remaining[idx]
                # Remove the minimal element from the remaining list.
                new_remaining = remaining[:idx] + remaining[idx+1:]
                # Remove the corresponding row and column from the matrix.
                new_h = np.delete(np.delete(h_sub, idx, axis=0), idx, axis=1)
                # Recursively generate extensions for the reduced poset.
                for ext in _recursive_extensions(new_h, new_remaining):
                    extensions.append([current] + ext)
            return extensions

        # Start the recursion with all indices [0, 1, ..., n-1].
        index_extensions = _recursive_extensions(h, list(range(n)))
        # Map the indices to actual items if provided.
        extensions = [[items[i] for i in extension] for extension in index_extensions]
        return extensions

    @staticmethod
    def intersection_of_extensions(items, order_list):
        """
        Given a list of linear orders (each order is a permutation of [0, 1, ..., n-1]),
        return the intersection adjacency representation.
        
        For any distinct pair x and y, add an edge x -> y if and only if x precedes y in every order.
        
            order_list: A list of linear orders (each order is a tuple or list of integers).
        
        Returns:
            A dictionary where each key is an item and the value is a set of items that follow it
            in every linear extension.
        """
        # Define the items and a mapping (though items are already indices here)
 
        # Create a position map for each order:
        # For each order, pos_map[x] gives the position of x in that order.
        pos_maps = [{x: pos for pos, x in enumerate(order)} for order in order_list]
        
        # Initialize the intersection adjacency dictionary.
        intersection_adj = {x: set() for x in items}
        
        # For every distinct pair (x, y), check if x precedes y in every order.
        for x in items:
            for y in items:
                if x != y and all(pos_map[x] < pos_map[y] for pos_map in pos_maps):
                    intersection_adj[x].add(y)
        
        return intersection_adj


class StatisticalUtils:
    """
    Utility class for statistical computations related to partial orders.
    """

    @staticmethod
    def count_unique_partial_orders(h_trace):
        """
        Count the frequency of each unique partial order in h_trace.
        
        Parameters:
        - h_trace: List of NumPy arrays representing partial orders.
        
        Returns:
        - Dictionary with partial order representations as keys and their counts as values.
        """
        unique_orders = defaultdict(int)
        
        for h_Z in h_trace:
            # Convert the matrix to a tuple of tuples for immutability
            h_tuple = tuple(map(tuple, h_Z))
            unique_orders[h_tuple] += 1
    

        sorted_unique_orders = sorted(unique_orders.items(), key=lambda x: x[1], reverse=True)
        
        # Convert the sorted tuples back to NumPy arrays for readability
        sorted_unique_orders = [(np.array(order), count) for order, count in sorted_unique_orders]
        return sorted_unique_orders
    @staticmethod
    def log_U_prior(Z: np.ndarray, rho: float, K: int, debug: bool = False) -> float:
        """
        Compute the log prior probability of Z.

        Parameters:
        - Z: Current latent variable matrix (numpy.ndarray).
        - rho: Step size for proposal (used here to scale covariance).
        - K: Number of dimensions.
        - debug: If True, prints the covariance matrix.

        Returns:
        - log_prior: Scalar log prior probability.
        """
        # Covariance matrix is scaled identity matrix
        Sigma =BasicUtils.build_Sigma_rho(K,rho)

        if debug:
            print(f"Covariance matrix Sigma:\n{Sigma}")

        # Compute log prior for each row in Z assuming independent MVN
        try:
            mvn = multivariate_normal(mean=np.zeros(K), cov=Sigma)
            log_prob = mvn.logpdf(Z)
            log_prior = np.sum(log_prob)
        except np.linalg.LinAlgError as e:
            print(f"LinAlgError in log_prior: {e}")
            print(f"Covariance matrix Sigma:\n{Sigma}")
            raise e

        return log_prior
    @staticmethod
    def log_U_prior_optimized(Z: np.ndarray, rho: float, K: int, debug: bool = False) -> float:
        """
        OPTIMIZED: Vectorized log prior computation for U0.
        
        Up to 20x faster than original by eliminating scipy object creation
        and using pure numpy operations.
        """
        # Build covariance matrix
        Sigma = BasicUtils.build_Sigma_rho(K, rho)
        
        # Use pure numpy operations instead of scipy
        try:
            # Pre-compute inverse and log determinant
            Sigma_inv = np.linalg.inv(Sigma)
            sign, log_det = np.linalg.slogdet(Sigma)
            
            if sign <= 0:
                raise np.linalg.LinAlgError("Non-positive determinant")
            
            # Vectorized computation for all rows at once
            # Z has shape (n_items, K), mean is zeros(K)
            n_items = Z.shape[0]
            
            # Vectorized quadratic form: Z @ Sigma_inv @ Z.T
            quad_forms = np.sum((Z @ Sigma_inv) * Z, axis=1)  # Shape: (n_items,)
            
            # Vectorized log probability
            normalization = -0.5 * (K * np.log(2 * np.pi) + log_det)
            log_probs = normalization - 0.5 * quad_forms
            
            log_prior = np.sum(log_probs)
            
                
            return log_prior
            
        except np.linalg.LinAlgError as e:
            if debug:
                print(f"LinAlgError in optimized log_prior, falling back: {e}")
            # Fallback to original method
            return StatisticalUtils.log_U_prior(Z, rho, K, debug)
    @staticmethod
    def transform_U_to_eta(U: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """
        Transform latent positions U to eta using Gumbel link function.
        
        Parameters:
        -----------
        U : np.ndarray
            Matrix of latent positions (n_global × K)
        alpha : np.ndarray
            Vector of assessor/item effects (n_global × 1)
            
        Returns:
        --------
        np.ndarray
            Matrix of transformed positions (n_global × K)
        """
        n_global, K = U.shape
        
        # Initialize output matrix
        eta = np.zeros((n_global, K))
        
        # Gumbel inverse link function
        def gumbel_inv(p):
            return -np.log(-np.log(p))
        
        # Transform each row
        for j in range(n_global):
            # Step 1: Convert to probabilities using normal CDF
            p_vec = norm.cdf(U[j, :])
            
            # Step 2: Apply Gumbel inverse link function
            gumbel_vec = np.array([gumbel_inv(px) for px in p_vec])
            
            # Step 3: Add assessor/item effect
            eta[j, :] = gumbel_vec + alpha[j]
        
        return eta
    @staticmethod
    def description_partial_order(h: np.ndarray) -> Dict[str, Any]:
        """
        Provides a detailed description of the partial order represented by the adjacency matrix h.

        Parameters:
        - h: An n x n numpy array representing the adjacency matrix of the partial order.

        Returns:
        - description: A dictionary containing descriptive statistics of the partial order.
        """
        G = nx.DiGraph(h)
        n = h.shape[0]
        node_num= G.number_of_nodes()

        # Number of relationships (edges)
        num_relationships = G.number_of_edges()

        # Number of alone nodes (no incoming or outgoing edges)
        alone_nodes = [node for node in G.nodes() if G.in_degree(node) == 0 and G.out_degree(node) == 0]
        num_alone_nodes = len(alone_nodes)

        # Maximum number of relationships a node can have with other nodes
        # Considering both in-degree and out-degree
        in_degrees = dict(G.in_degree())
        out_degrees = dict(G.out_degree())
        max_in_degree = max(in_degrees.values()) if in_degrees else 0
        max_out_degree = max(out_degrees.values()) if out_degrees else 0
        max_relationships = max(max_in_degree, max_out_degree)

        # Number of linear extensions
        tc=BasicUtils.transitive_closure(h)
        tr = BasicUtils.transitive_reduction_optimized(tc)
        num_linear_extensions = BasicUtils._r(tr)

        # Depth of the partial order (length of the longest chain)
        try:
            depth = nx.dag_longest_path_length(G)
        except nx.NetworkXUnfeasible:
            depth = None  # If the graph is not a DAG

        description = {
            "Number of Nodes": node_num,
            "Number of Relationships": num_relationships,
            "Number of Alone Nodes": num_alone_nodes,
            "Alone Nodes": alone_nodes,
            "Maximum In-Degree": max_in_degree,
            "Maximum Out-Degree": max_out_degree,
            "Maximum Relationships per Node": max_relationships,
            "Number of Linear Extensions": num_linear_extensions,
            "Depth of Partial Order": depth
        }

        # Print the description
        print("\n--- Partial Order Description ---")
        for key, value in description.items():
            print(f"{key}: {value}")
        print("---------------------------------")



    @staticmethod
    def sample_conditional_z(Z, rZ, cZ, rho):
        K = Z.shape[1]

        # Build correlation matrix
        Sigma = np.full((K, K), rho)
        np.fill_diagonal(Sigma, 1.0)

        dependent_ind = cZ
        given_inds = [i for i in range(K) if i != cZ]

        Sigma_dd = Sigma[dependent_ind, dependent_ind]  # scalar
        Sigma_dg = Sigma[dependent_ind, given_inds]     # shape (K-1,)  <-- FIXED HERE
        Sigma_gg = Sigma[np.ix_(given_inds, given_inds)]

        # X_g is also shape (K-1,)
        X_g = Z[rZ, given_inds]

        # Means are 0
        mu_d = 0.0
        mu_g = 0.0

        # Invert Sigma_gg
        try:
            Sigma_gg_inv = np.linalg.inv(Sigma_gg)
        except np.linalg.LinAlgError:
            Sigma_gg += np.eye(Sigma_gg.shape[0]) * 1e-8
            Sigma_gg_inv = np.linalg.inv(Sigma_gg)

        # Conditional mean
        mu_cond = mu_d + Sigma_dg @ Sigma_gg_inv @ (X_g - mu_g)
        # Conditional variance
        var_cond = Sigma_dd - Sigma_dg @ Sigma_gg_inv @ Sigma_dg

        var_cond = max(var_cond, 1e-8)
        Z_new = np.random.normal(loc=mu_cond, scale=np.sqrt(var_cond))

        return Z_new

#############################################Hyperparameter prior for HPO#############################################

#   ### rho 
    @staticmethod
    def rRprior(fac=1/6, tol=1e-4):
        """
        Draw a sample for ρ from a Beta(1, fac) distribution, but reject any sample
        for which 1 - ρ < tol, to avoid numerical instability when ρ is extremely close to 1.
        
        Parameters:
        fac: Second parameter of the Beta distribution (default 1/6).
        tol: Tolerance such that we require 1 - ρ >= tol (default 1e-4).
        
        Returns:
        A single float value for ρ.
        """
        while True:
            rho = beta.rvs(1, fac)
            if 1 - rho >= tol:
                return rho
    @staticmethod
    def dRprior(rho: float, fac=1/6, tol=1e-4) -> float:
        """
        Compute the log prior for ρ from a Beta(1, fac) distribution, with truncation
        at 1 - tol. If ρ > 1 - tol, return -Inf. Otherwise, adjust the log density
        by subtracting the log cumulative probability at 1-tol.
        
        Parameters:
        rho: the value of ρ.
        fac: the Beta distribution second parameter (default 1/6).
        tol: tolerance for the upper bound (default 1e-4).
        
        Returns:
        The log density (a float).
        """
        if rho > 1 - tol:
            return -np.inf
        # Compute the log PDF at rho.
        log_pdf = beta.logpdf(rho, 1, fac)
        # Subtract the log of the cumulative probability up to 1-tol, effectively renormalizing.
        log_cdf_trunc = beta.logcdf(1 - tol, 1, fac)
        return log_pdf - log_cdf_trunc
####Prob 

    @staticmethod
    def rPprior(noise_beta_prior):
        return beta.rvs(1, noise_beta_prior)
    
    @staticmethod
    def dPprior(p, beta_param):
        """
        Log-prior for p ~ Beta(1, beta_param).

        Returns -inf if p is out of (0,1).
        Otherwise, logpdf of Beta(1, beta_param).
        """
        if p <= 0.0 or p >= 1.0:
            return -math.inf
        
        return beta.logpdf(p, 1.0, beta_param)

### Tau 
    @staticmethod
    def rTauprior(tol: float = 1e-4):
        """Sample tau ~ Uniform(0, 1 - tol] to avoid singular covariance when tau is too close to 1."""
        return random.uniform(0.0, 1.0 - tol)


    @staticmethod
    def dTauprior(tau: float, tol: float = 1e-4):
        """Log-density of the (truncated) Uniform(0, 1-tol] prior for tau."""
        if tau <= 0.0 or tau >= 1.0 - tol:
            return -math.inf
        # Density of Uniform(0, 1-tol] is 1/(1-tol)
        return -math.log(1.0 - tol)
####Theta  

    @staticmethod
    def rTprior(mallow_ua):
        return gamma.rvs(a=1, scale=1.0/mallow_ua)
    @staticmethod
    def dTprior(mallow_theta, ua):
        """
        Log-prior for mallow_theta under an Exponential(ua) distribution.
        i.e. p(mallow_theta) = Exponential(ua) with pdf:
            p(mallow_theta) = ua * exp(-ua * mallow_theta), for mallow_theta > 0.
        """
        if mallow_theta <= 0:
            return -np.inf

        return expon.logpdf(mallow_theta, scale=1/ua)
    
####K  
    @staticmethod
    def dKprior(k: int, lam: float) -> float:
        """Log PMF of Poisson(λ) truncated at k ≥ 1."""
        if k < 1:
            return -np.inf
        # log(k!) using gammaln(k+1)
        log_k_fact = math.lgamma(k+1)
        # normalizing constant for truncation
        norm_const = -np.log(1 - np.exp(-lam))
        val = -lam + k * np.log(lam) - log_k_fact + norm_const
        return val
    
    @staticmethod
    def rKprior(lam: float = 3.0) -> int:
        """
        Sample a new K from a Poisson(λ) distribution truncated to k ≥ 1.
        
        This function repeatedly draws from a Poisson(λ) until a value ≥ 1 is obtained.
        """
        candidate = np.random.poisson(lam)
        while candidate < 1:
            candidate = np.random.poisson(lam)
        return candidate


    
    @staticmethod
    def log_U_hierarchical_prior(
        U0: np.ndarray,                  # shape (|M0|, K)
        U_a_list: list,                  # length A, each shape (|M_a|, K)
        M_a_dict: list,                  # length A, each is a list of global object indices
        tau: float,
        Sigma_rho: np.ndarray       # shape (K,K)
                    # function log_mvnorm(x, mean, cov) -> float
    ) -> float:

        logp = 0.0

        # 1) log for each U^(0)[j,:] ~ N(0, Sigma_rho)
        n_global = U0.shape[0]
        for j in range(n_global):
            x_j = U0[j,:]                # a 1D vector of length K
            zero_vec = np.zeros_like(x_j)
            logp += np.log(multivariate_normal(x_j, zero_vec, Sigma_rho))

        # 2) for each assessor a, for each j in M_a
        A = len(U_a_list)
        for a_idx in range(A):
            Ua = U_a_list[a_idx]        # shape (|M_a|, K)
            Ma = M_a_dict.get(a_idx,[])            # list of global indices
            for row_loc, j_global in enumerate(Ma):
                # row in U^(a) => U_a_list[a_idx][row_loc,:]
                x_aj = Ua[row_loc,:]
                # mean is tau * U0[j_global,:]
                mean_aj = tau * U0[j_global,:]
                # cov is (1 - tau^2)*Sigma_rho                
                cov_aj = (1.0 - tau**2) * Sigma_rho


                logp += np.log(multivariate_normal(x_aj, mean_aj, cov_aj))

        return logp
    @staticmethod
    def sample_conditional_column(Z, rho):
        """
        Z is shape (n, K). For each row i, we want the bridging col
        of shape (n,) that respects the correlation among columns.
        
        We assume an equicorrelation or some covariance Sigma_full 
        of shape (K+1, K+1).
        """
        n, K = Z.shape
        Kplus1 = K + 1

        # Build the (K+1)x(K+1) covariance:
        Sigma_full = BasicUtils.build_Sigma_rho(Kplus1, rho)
        # Partition Sigma_full:
        # Sigma_gg = Sigma_full[0:K,0:K]
        # Sigma_dg = Sigma_full[K,   0:K]
        # Sigma_dd = Sigma_full[K,   K]
        
        Sigma_gg = Sigma_full[:K, :K]
        Sigma_dg = Sigma_full[K, :K]       # shape (K,)
        Sigma_dd = Sigma_full[K, K]        # scalar

        # Invert Sigma_gg once for all
        Sigma_gg_inv = np.linalg.inv(Sigma_gg)

        bridging_col = np.zeros(n)
        for i in range(n):
            x_i = Z[i,:]  # existing coords
            # conditional mean
            mu_cond = Sigma_dg @ Sigma_gg_inv @ x_i
            # conditional var
            var_cond = Sigma_dd - Sigma_dg @ Sigma_gg_inv @ Sigma_dg
            # sample
            bridging_col[i] = np.random.normal(mu_cond, np.sqrt(var_cond))

        return bridging_col
        
    @staticmethod
    def sample_conditional_column_child(Ua, U0_subset, tau, rho, rng):
        """
        Ua        : (n_a , K)   – current assessor matrix (all K old columns)
        U0_subset : (n_a , K)   – corresponding rows (same items) from U0
        returns   : vector length n_a  – new column c
        """
        n_a, K = Ua.shape
        Sigma_num = (1 - tau**2)*(1 - rho)*(1 + (K-1)*rho)
        Sigma_den = 1 + (K-2)*rho
        var_c     = Sigma_num / Sigma_den       # scalar σ²_{j,c}

        # pre-compute the normalising factor
        w = rho / (1 + (K-2)*rho)

        new_col = np.empty(n_a)
        for j in range(n_a):
            diff = Ua[j] - tau * U0_subset[j]   # vector (K,)
            mu   = tau * U0_subset[j, 0] + w * diff.sum()  # formula above
            new_col[j] = rng.normal(mu, np.sqrt(var_c))
        return new_col

    def U0_conditional_update(
        j_global,        # index of the row in U0 we want to update
        U0,              # current U0, shape (n_global, K)
        U_a_dict,        # dictionary of child latents {a: U^a}, each shape (len(M_a), K)
        M_a_dict,        # {a: list_of_indices_in_M_a}, tells which global indices belong to assessor a
        tau,             # correlation parameter
        Sigma_rho,       # K x K covariance for the base distribution
        rng              # random number generator, e.g., np.random.default_rng()
    ):
        """
        Perform a direct Gibbs draw for row j_global of U0 given all child rows U^(a).
        """
        # 1) Gather all child-latent vectors that correspond to the same "global" item j_global
        #    For each a in U_a_dict, find the local index i_loc where j_global appears in M_a_dict[a].
        #    If j_global is not in M_a_dict[a], skip it. Otherwise get U_a[i_loc].
        child_vectors = []
        for a, U_a in U_a_dict.items():
            if j_global in M_a_dict[a]:
                i_loc = M_a_dict[a].index(j_global)
                child_vectors.append(U_a[i_loc, :])
        
        A_j = len(child_vectors)  # how many assessors actually have j_global in their list

        # 2) If no child has j_global, posterior = prior => Normal(0, Sigma_rho)
        if A_j == 0:
            post_mean = np.zeros_like(U0[j_global, :])
            post_cov = Sigma_rho
        else:
            # 3) Compute the posterior mean & covariance for that row
            sum_child = np.sum(child_vectors, axis=0)  # sum_{a=1..A_j} u_j^(a)

            denom = (1 - tau**2) + A_j * (tau**2)
            # Posterior mean
            post_mean = (tau / denom) * sum_child

            # Posterior covariance
            shrink_factor = (1 - tau**2) / denom
            post_cov = shrink_factor * Sigma_rho
        
        # 4) Draw a new sample from N(post_mean, post_cov)
        new_row = rng.multivariate_normal(post_mean, post_cov)
    

        return new_row 

####################Below are separate coding for hpo####
    @staticmethod
    def gumbel_inv_cdf(p: float, eps: float = 1e-15) -> float:
        # Clip p so it lies in [eps, 1 - eps] to avoid log(0)
        p_clipped = np.clip(p, eps, 1 - eps)
        return -np.log(-np.log(p_clipped))
    
    @staticmethod
    def log_U_a_prior(U_a_dict: Dict[int, np.ndarray], tau: float, rho: float, K: int, M_a_dict: Dict[int, List[int]], U0: np.ndarray) -> float:
        """
        Compute the log prior probability for assessor-level latent variables.
        
        Each assessor a has latent variables U_a ~ N(tau * U0[j], (1 - tau^2)*Sigma_rho)
        for each global item j in M_a_dict[a].

        Parameters:
        U_a_dict: Dictionary with keys as assessor IDs and values as latent matrices (shape: (|M_a|, K)).
        tau: The branching parameter.
        rho: The correlation parameter.
        K: Dimensionality of the latent space.
        M_a_dict: Dictionary with keys as assessor IDs and values as lists of global item indices for that assessor.
        U0: Global latent matrix (shape: (|M0|, K)).

        Returns:
        log_prior_total: The sum of log prior probabilities over all assessor-level latent variables.
        """
        Sigma_rho =BasicUtils.build_Sigma_rho(K,rho)
        log_prior_total = 0.0

        for a, U_a in U_a_dict.items():
            # Get the list of global items for assessor a.
            Ma = M_a_dict.get(a, [])
            log_prior = 0.0
            for i, j_global in enumerate(Ma):

                mean_vec = tau * U0[j_global, :]

                cov_mat= (1.0 - tau**2) * Sigma_rho
                log_prob = multivariate_normal.logpdf(
                            U_a[i, :],
                            mean=mean_vec,
                            cov=cov_mat,
                            allow_singular=True
                        )

                log_prior += log_prob
            log_prior_total += log_prior

        return log_prior_total


    @staticmethod


    def log_U_a_prior_fast(
        U_a_dict: Dict[int, np.ndarray],
        tau: float,
        rho: float,
        K: int,
        M_a_dict: Dict[int, List[int]],
        U0: np.ndarray,
        *,                      # force keywords after here
        regularise: float = 1e-8
    ) -> float:
        """
        Ultra‑fast vectorised prior with *cached* Σ⁻¹ and log|Σ|.
        """
        key = (K, rho, tau)
        if key in _COV_CACHE:
            cov_inv, log_det, norm_const = _COV_CACHE[key]
        else:
            # ---------- build Σ -------------------------------------------------
            Sigma = BasicUtils.build_Sigma_rho(K, rho)
            cov   = (1.0 - tau*tau) * Sigma

            # ---------- invert & log‑det (with fallback) ------------------------
            try:
                cov_inv = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                cov += regularise * np.eye(K)
                cov_inv = np.linalg.inv(cov)

            log_det = np.linalg.slogdet(cov)[1]
            norm_const = -0.5 * (K * np.log(2*np.pi) + log_det)

            _COV_CACHE[key] = (cov_inv, log_det, norm_const)   # ✨ cache

        logp = 0.0
        for a, U_a in U_a_dict.items():
            Ma = M_a_dict.get(a, [])
            if not Ma:
                continue
            mu = tau * U0[Ma, :]                 # (m,K)
            res = U_a - mu                       # (m,K)
            quad = np.sum((res @ cov_inv) * res, axis=1)
            logp += np.sum(norm_const - 0.5*quad)

        return logp

    @staticmethod
    ### The objective of this function is buidling a hierarchical partial order of H(U) given M0, Ma, Oa_list and U_alist 
    def build_hierarchical_partial_orders(
        M0,
        assessors,
        M_a_dict,
        U0,           # shape: (|M0|, K)
        U_a_dict,
        alpha,       
        link_inv=None
    ):
        if link_inv is None:
            # Default to Gumbel quantile
            link_inv = StatisticalUtils.gumbel_inv_cdf

        n_global, K = U0.shape
        eta0 = np.zeros_like(U0)
        for j_global in range(n_global):
            p_vec = norm.cdf(U0[j_global, :])  # coordinate-wise
            gumbel_vec = np.array([link_inv(px) for px in p_vec])
            eta0[j_global, :] = gumbel_vec + alpha[j_global]

        h0 = BasicUtils.generate_partial_order(eta0)


        h_U = {}
        h_U={0:h0}

        # Loop over assessors
        for idx_a, a in enumerate(assessors):
            # 1) Build the *full partial order* on M_a
            Ma = M_a_dict.get(a,[])               # e.g. [0,2,4]
            Ua = U_a_dict.get(a,[])               # shape (|M_a|, K)
            # (a) Compute eta^(a) for each item j in M_a
            #     eqn (21): eta_j^{(a)} = G^-1( Phi(U_j^{(a)}) ) + alpha_j
            # We do it row by ro
            eta_a = np.zeros_like(Ua)
            for i_loc, j_global in enumerate(Ma):
                p_vec = norm.cdf(Ua[i_loc, :])
                gumbel_vec = np.array([link_inv(px) for px in p_vec])
                eta_a[i_loc, :] = gumbel_vec + alpha[j_global]

            # adjacency for M_a
            h_a = BasicUtils.generate_partial_order(eta_a)
            # store in dictionary
            h_U[a] = h_a

        return h_U
    
    @staticmethod
    def dict_array_equal(d1, d2):
        """Recursively compare two dictionaries where values may be NumPy arrays."""
        if d1.keys() != d2.keys():
            return False
        for key in d1:
            v1, v2 = d1[key], d2[key]
            if isinstance(v1, dict) and isinstance(v2, dict):
                if not StatisticalUtils.dict_array_equal(v1, v2):
                    return False
            elif isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
                if not np.array_equal(v1, v2):
                    return False
            else:
                if v1 != v2:
                    return False
        return True
    @staticmethod
    def generate_total_order_for_choice_set_with_queue_jump(
        subset: List[int],
        M_a: List[int],
        h_local: np.ndarray,
        prob_noise: float
    ) -> List[int]:
        """
        Given:
        - 'subset': A list of *global* item IDs we want to order.
        - 'M_a': The assessor's entire set of global item IDs (size = |M_a|).
        - 'h_local': A local partial-order matrix of shape (|M_a|, |M_a|),
                    where h_local[i,j]=1 => "item M_a[i] < item M_a[j]" in the assessor's order.
        - 'prob_noise': Probability of a 'jump' (i.e., random pick) in the queue-jump.

        Returns:
        A total order of items in 'subset', as a list of global item IDs.
        """

        # 1) Build a map from *global* item => local index in M_a
        #    so we can slice h_local properly.
        global2local = { g: i for i, g in enumerate(M_a) }

        # 2) Identify which global items in 'subset' are also in M_a,
        #    and convert them to local indices
        local_subset_idx = []
        local_subset_global = []  # store the same items, but parallel to local indices
        for g in subset:
            if g in global2local:          # only items that exist in M_a
                local_idx = global2local[g]
                local_subset_idx.append(local_idx)
                local_subset_global.append(g)

        # If no overlap, return empty
        if not local_subset_idx:
            return []

        # 3) Extract the local submatrix for these items
        #    shape = (len(local_subset_idx), len(local_subset_idx))
        h_matrix_subset = h_local[np.ix_(local_subset_idx, local_subset_idx)]

        # 4) We'll do the queue-jump logic in local SUBSET indices = [0..(n_sub-1)]
        n_sub = len(local_subset_idx)
        # So we make a direct mapping from "subset index" => "global item ID"
        # e.g. subset_idx2global[i] = local_subset_global[i]
        # And we'll keep 'remaining' as [0..n_sub-1].
        subset_idx2global = { i: local_subset_global[i] for i in range(n_sub) }

        remaining = list(range(n_sub))  # local indices in [0..n_sub-1]
        total_order_local = []


        while remaining:
            m = len(remaining)
            if m == 1:
                total_order_local.append(remaining[0])
                break

            # Build sub-submatrix for 'remaining'
            # shape => (m, m)
            h_rem = h_matrix_subset[np.ix_(remaining, remaining)]

            # Transitive reduction of that sub-submatrix
            tr_rem = BasicUtils.transitive_reduction_optimized(h_rem)

            # Count total # of linear extensions
            N_total = BasicUtils.nle(tr_rem)

            # Compute candidate probabilities for each local_idx in [0..m-1]
            candidate_probs = []
            for local_idx in range(m):
                # Number of linear extensions that start with 'local_idx'
                # This uses BasicUtils.num_extensions_with_first
                # but that function expects the partial order submatrix + top elements, etc.
                # So local_idx is an index in [0..m-1].
                # We pass 'tr_rem' and local_idx to BasicUtils.num_extensions_with_first
                N_first = BasicUtils.num_extensions_with_first(tr_rem, local_idx)
                p_no_jump = (1 - prob_noise) * (N_first / N_total)
                candidate_probs.append(p_no_jump)

            # Probability of 'jump' => prob_noise, distributed uniformly among m candidates
            p_jump = prob_noise * (1.0 / m)
            candidate_probs = [p + p_jump for p in candidate_probs]

            # normalize
            total_p = sum(candidate_probs)
            candidate_probs = [p / total_p for p in candidate_probs]

            # Sample an index from 'remaining' with these weights
            chosen_subindex = random.choices(range(m), weights=candidate_probs, k=1)[0]
            chosen_local = remaining[chosen_subindex]

            total_order_local.append(chosen_local)
            remaining.remove(chosen_local)

        # 5) Convert 'total_order_local' (which are indices in [0..n_sub-1])
        #    back to *global* item IDs
        global_order = [subset_idx2global[i] for i in total_order_local]

        return global_order

    @staticmethod
    def truncated_poisson_pdf(x, mu):
        """
        Calculate the PDF of a truncated Poisson distribution.
        
        Parameters:
        - x: Value to evaluate the PDF at
        - mu: Mean parameter of the Poisson distribution
        
        Returns:
        - PDF value at x
        """
        # Use dKprior to calculate the log probability
        log_prob = StatisticalUtils.dKprior(x, mu)
        # Convert from log probability to probability
        return np.exp(log_prob)
    
    @staticmethod
    def truncated_poisson_cdf(x, mu):

        if x < 1:
            return 0
        
        # Calculate the CDF by summing the PDF values from 1 to x
        cdf = 0
        for k in range(1, int(x) + 1):
            cdf += StatisticalUtils.truncated_poisson_pdf(k, mu)
        
        return cdf
    
    @staticmethod
    def truncated_poisson_mean(mu):
        """
        Calculate the mean of a truncated Poisson distribution.
        
        Parameters:
        - mu: Mean parameter of the Poisson distribution
        
        Returns:
        - Mean of the truncated distribution
        """
        from scipy.stats import poisson
        # Mean of truncated Poisson
        norm_const = 1 - poisson.pmf(0, mu)
        return mu / norm_const
    
    @staticmethod
    def truncated_poisson_var(mu):
        """
        Calculate the variance of a truncated Poisson distribution.
        
        Parameters:
        - mu: Mean parameter of the Poisson distribution
        
        Returns:
        - Variance of the truncated distribution
        """
        from scipy.stats import poisson
        # Variance of truncated Poisson
        norm_const = 1 - poisson.pmf(0, mu)
        return mu * (1 - mu * poisson.pmf(0, mu) / norm_const) / norm_const
    
    @staticmethod
    def TruncatedPoisson(mu):
        """
        Create a truncated Poisson distribution object.
        
        Parameters:
        - mu: Mean parameter of the Poisson distribution
        
        Returns:
        - A distribution object with pdf, cdf, mean, and var methods
        """
        class TruncatedPoissonDist:
            def __init__(self, mu):
                self.mu = mu
                self.name = "TruncatedPoisson"
                
            def pdf(self, x):
                return StatisticalUtils.truncated_poisson_pdf(x, self.mu)
                
            def cdf(self, x):
                return StatisticalUtils.truncated_poisson_cdf(x, self.mu)
                
            def mean(self):
                return StatisticalUtils.truncated_poisson_mean(self.mu)
                
            def var(self):
                return StatisticalUtils.truncated_poisson_var(self.mu)
        
        return TruncatedPoissonDist(mu)


    @staticmethod
    def dBetaprior(beta: np.ndarray, sigma_beta: Union[float, np.ndarray]) -> float:
        """
        Log-pdf of a multivariate Normal(0, Sigma) at point 'beta', where Sigma is a diagonal matrix.
        
        Parameters:
        -----------
        beta: shape (p,)
            Vector of coefficients
        sigma_beta: float or np.ndarray of shape (p,)
            The prior standard deviation(s) for each coefficient.
            Can be either a scalar (same std dev for all coefficients) or an array (different std dev per coefficient)
        
        Returns:
        --------
        float
            The log-density value
            
        Notes:
        ------
        When sigma_beta is a scalar, formula is:
          - (p/2) * log(2*pi) 
          - p*log(sigma_beta)
          - (1 / (2*sigma_beta^2)) * sum(beta^2)
          
        When sigma_beta is an array, formula is:
          - (p/2) * log(2*pi) 
          - sum(log(sigma_beta))  # sum of logs instead of p times log of one value
          - sum(beta^2 / (2*sigma_beta^2))  # element-wise division by the variances
        """
        p = len(beta)
        
        if np.isscalar(sigma_beta):
            # Original implementation for scalar sigma_beta
            log_det_part = -0.5 * p * math.log(2.0 * math.pi) - p * math.log(sigma_beta)
            quad_part = -0.5 * np.sum(beta**2) / (sigma_beta**2)
        else:
            # Handle array case
            if len(sigma_beta) != p:
                raise ValueError(f"sigma_beta must be a scalar or have length {p} to match beta")
            
            log_det_part = -0.5 * p * math.log(2.0 * math.pi) - np.sum(np.log(sigma_beta))
            quad_part = -0.5 * np.sum(beta**2 / (sigma_beta**2))
            
        return log_det_part + quad_part

    @staticmethod
    def rBetaPrior(sigma_beta: Union[float, np.ndarray], p: int) -> np.ndarray:
        """
        Sample a new beta from a Normal(0, Sigma) distribution, where Sigma is a diagonal matrix.
        
        Parameters:
        -----------
        sigma_beta: float or np.ndarray
            If scalar: the same prior std dev for each coefficient (diagonal elements will be sigma_beta^2)
            If array: different prior std dev for each coefficient (must have length p)
        p: integer
            Dimension of beta.
        
        Returns:
        --------
        np.ndarray
            Sampled beta vector of shape (p,)
        """
        if np.isscalar(sigma_beta):
            return np.random.normal(loc=0.0, scale=sigma_beta, size=(p,))
        else:
            if len(sigma_beta) != p:
                raise ValueError(f"If sigma_beta is an array, it must have length {p}")
            return np.random.normal(loc=0.0, scale=sigma_beta, size=(p,))

