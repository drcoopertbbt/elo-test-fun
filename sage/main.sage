#!/usr/bin/env sage
"""
Main SageMath zELO Implementation
Single source of truth for all SageMath calculations
Can be run standalone for tests or called as a service
"""

from sage.all import *
import sys
import os
import json
import time

# ============================================================================
# CORE ZELO ALGORITHM - ZBENCH'S EXACT IMPLEMENTATION
# ============================================================================

def calculate_elos_zbench_exact(n, w, epsilon=1e-4, max_iters=1000, verbose=False):
    """
    ZBench's EXACT mathematical algorithm in pure SageMath
    Uses Robbins-Monro learning rate: (1 + iter)^(-0.125)
    
    Args:
        n: Number of documents
        w: Preference matrix (n x n)
        epsilon: Convergence threshold
        max_iters: Maximum iterations
        verbose: Print iteration details
    
    Returns:
        elos: Vector of Elo scores
        losses: List of loss values
        converged: Boolean convergence status
    """
    # Initialize with zeros
    current_elos = vector(RR, [0.0] * n)
    
    losses = []
    
    for iteration in range(max_iters):
        # Create pairwise difference matrix: D[i,j] = elos[j] - elos[i]
        D = matrix(RR, n, n)
        for i in range(n):
            for j in range(n):
                D[i, j] = current_elos[j] - current_elos[i]
        
        # Sigmoid matrix: S[i,j] = 1/(1 + exp(-D[i,j]))
        S = matrix(RR, n, n)
        for i in range(n):
            for j in range(n):
                S[i, j] = 1 / (1 + exp(-D[i, j]))
        
        # Calculate update terms
        numerator = vector(RR, n)
        denominator = vector(RR, n)
        
        for i in range(n):
            numerator[i] = sum(w[i, j] * S[i, j] for j in range(n))
            denominator[i] = sum(w[j, i] * S[j, i] for j in range(n))
        
        # Robbins-Monro learning rate
        learning_rate = (1 + iteration) ** RR(-0.125)
        
        # Update Elos
        for i in range(n):
            if numerator[i] > 0 and denominator[i] > 0:
                current_elos[i] += (log(numerator[i]) - log(denominator[i])) * learning_rate
        
        # Enforce sum-to-zero constraint
        elos_mean = sum(current_elos) / n
        current_elos = vector(RR, [e - elos_mean for e in current_elos])
        
        # Calculate loss using log-sum-exp trick for stability
        loss = 0.0
        for i in range(n):
            for j in range(n):
                if w[i, j] > 0:
                    max_elo = max(current_elos[i], current_elos[j])
                    log_sum = max_elo + log(exp(current_elos[i] - max_elo) + 
                                           exp(current_elos[j] - max_elo))
                    loss -= w[i, j] * (current_elos[i] - log_sum)
        
        losses.append(float(loss))
        
        if verbose and iteration % 50 == 0:
            print(f"Iter {iteration}: Loss = {loss:.6f}")
        
        # Check convergence
        if len(losses) > 1 and abs(losses[-2] - losses[-1]) < epsilon:
            if verbose:
                print(f"Converged at iteration {iteration}")
            break
    
    converged = len(losses) < max_iters
    
    if not converged and verbose:
        print(f"Warning: Did not converge after {max_iters} iterations")
    
    return current_elos, losses, converged

def build_preference_matrix_from_comparisons(n, comparisons):
    """
    Build n x n preference matrix from sparse comparisons
    
    Args:
        n: Number of documents
        comparisons: List of dicts with 'i', 'j', 'weight' keys
    
    Returns:
        w: Preference matrix
    """
    w = matrix(RR, n, n, sparse=True)
    
    # Initialize diagonal (documents tied with themselves)
    for i in range(n):
        w[i, i] = 0.5
    
    for comp in comparisons:
        i = comp['i']
        j = comp['j']
        weight = comp['weight']
        w[i, j] = weight
        # Add complementary loss
        w[j, i] = 1.0 - weight
    
    return w

# ============================================================================
# SERVICE MODE - FOR EXTERNAL CALLS
# ============================================================================

def process_elo_request(input_data):
    """
    Process an Elo calculation request
    
    Args:
        input_data: Dict with 'document_ids', 'comparisons', 'config'
    
    Returns:
        output_data: Dict with 'scores' and 'status'
    """
    # Extract data
    document_ids = input_data['document_ids']
    comparisons = input_data['comparisons']
    config = input_data.get('config', {})
    
    n = len(document_ids)
    model = config.get('model', 'bradley-terry')
    epsilon = config.get('epsilon', 1e-4)
    max_iters = config.get('max_iters', 1000)
    verbose = config.get('verbose', False)
    
    # Build preference matrix
    w = build_preference_matrix_from_comparisons(n, comparisons)
    
    # Calculate Elos
    start_time = time.time()
    
    if model == 'bradley-terry':
        elos, losses, converged = calculate_elos_zbench_exact(n, w, epsilon, max_iters, verbose)
    else:
        # Could add Thurstone model here if needed
        if verbose:
            print(f"Model '{model}' not yet implemented, using bradley-terry")
        elos, losses, converged = calculate_elos_zbench_exact(n, w, epsilon, max_iters, verbose)
    
    elapsed_time = time.time() - start_time
    
    # Create output
    scores_dict = {}
    for i, doc_id in enumerate(document_ids):
        scores_dict[doc_id] = float(elos[i])
    
    output_data = {
        'scores': scores_dict,
        'status': {
            'converged': converged,
            'iterations': len(losses),
            'final_loss': losses[-1] if losses else 0.0,
            'time_seconds': elapsed_time,
            'model': model,
            'n_documents': n,
            'n_comparisons': len(comparisons)
        }
    }
    
    # Add convergence warning if needed
    if not converged:
        output_data['status']['warning'] = f"Did not converge after {max_iters} iterations"
    
    return output_data

# ============================================================================
# TEST MODE - COMPREHENSIVE TEST SUITE
# ============================================================================

def run_test_suite():
    """Run comprehensive test suite"""
    print("=" * 80)
    print("zELO ALGORITHM TEST SUITE")
    print("=" * 80)
    
    # Test 1: Basic functionality
    print("\nTEST 1: Basic Bradley-Terry")
    print("-" * 40)
    
    # Create simple preference matrix
    n = 4
    w = matrix(RR, n, n)
    w[0, 0] = 0.5; w[0, 1] = 1.0; w[0, 2] = 1.0; w[0, 3] = 1.0
    w[1, 0] = 0.0; w[1, 1] = 0.5; w[1, 2] = 1.0; w[1, 3] = 1.0
    w[2, 0] = 0.0; w[2, 1] = 0.0; w[2, 2] = 0.5; w[2, 3] = 1.0
    w[3, 0] = 0.0; w[3, 1] = 0.0; w[3, 2] = 0.0; w[3, 3] = 0.5
    
    elos, losses, converged = calculate_elos_zbench_exact(n, w, verbose=True)
    
    print(f"\nElo scores: {[float(e) for e in elos]}")
    print(f"Converged: {converged}")
    print(f"Final loss: {losses[-1] if losses else 'N/A'}")
    
    # Test 2: Service mode
    print("\nTEST 2: Service Mode")
    print("-" * 40)
    
    test_input = {
        "document_ids": ["doc_A", "doc_B", "doc_C"],
        "comparisons": [
            {"i": 0, "j": 1, "weight": 1.0},
            {"i": 0, "j": 2, "weight": 0.8},
            {"i": 1, "j": 2, "weight": 0.6}
        ],
        "config": {
            "model": "bradley-terry",
            "epsilon": 1e-4,
            "max_iters": 1000,
            "verbose": False
        }
    }
    
    output = process_elo_request(test_input)
    print("Service output:")
    print(json.dumps(output, indent=2))
    
    print("\n" + "=" * 80)
    print("TEST SUITE COMPLETE")
    print("=" * 80)

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for standalone execution"""
    if len(sys.argv) == 1:
        # No arguments - run test suite
        run_test_suite()
    elif len(sys.argv) == 3:
        # Service mode: input.json output.json
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        
        # Load input
        try:
            with open(input_file, 'r') as f:
                input_data = json.load(f)
        except Exception as e:
            print(f"Error loading input file: {e}")
            sys.exit(1)
        
        # Process request
        output_data = process_elo_request(input_data)
        
        # Save output
        try:
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
        except Exception as e:
            print(f"Error writing output file: {e}")
            sys.exit(1)
        
        print(f"Successfully calculated Elos for {output_data['status']['n_documents']} documents")
        print(f"Results saved to {output_file}")
    else:
        print("Usage:")
        print("  sage main.sage                # Run test suite")
        print("  sage main.sage input.json output.json  # Service mode")
        sys.exit(1)

if __name__ == "__main__":
    main()