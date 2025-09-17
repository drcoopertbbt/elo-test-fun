"""
ZBench Integration with SageMath High-Precision Engine
Clean interface module that replaces calculate_elos with SageMath backend
"""

import sys
import os
from pathlib import Path

# Add sage directory to path for imports
sage_dir = Path(__file__).parent.parent.parent / "sage"
sys.path.insert(0, str(sage_dir))

try:
    from main import calculate_elos_sage
    SAGEMATH_AVAILABLE = True
except ImportError:
    SAGEMATH_AVAILABLE = False
    print("Warning: SageMath interface not available, falling back to internal implementation")

def calculate_elos_with_sage_backend(w, document_ids=None, fallback=True):
    """
    High-precision Elo calculation using SageMath backend
    
    Args:
        w: Preference matrix (numpy array or list of lists)
        document_ids: Optional document identifiers
        fallback: Whether to use internal implementation if SageMath fails
    
    Returns:
        elos: Elo scores array
        status: Status information (optional second return value)
    """
    if not SAGEMATH_AVAILABLE:
        if fallback:
            print("  Using internal calculate_elos (SageMath unavailable)")
            from zbench.utils import calculate_elos
            return calculate_elos(w)
        else:
            raise RuntimeError("SageMath not available and fallback disabled")
    
    try:
        # Call SageMath engine
        print("  Using SageMath high-precision engine...")
        elos, status = calculate_elos_sage(w, document_ids)
        
        # Print status
        converged = status.get('converged', False)
        iterations = status.get('iterations', 0)
        print(f"  ✓ SageMath: Converged={converged}, Iterations={iterations}")
        
        return elos, status
        
    except Exception as e:
        if fallback:
            print(f"  Warning: SageMath failed ({e}), using fallback")
            from zbench.utils import calculate_elos
            return calculate_elos(w)
        else:
            raise