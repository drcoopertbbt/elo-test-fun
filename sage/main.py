#!/usr/bin/env python3
"""
Python interface to main.sage
Simple, clean bridge between ZBench and SageMath
Enhanced with comprehensive mathematical logging
"""

import subprocess
import json
import tempfile
import os
import time
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

class SageMathInterface:
    """Clean Python interface to SageMath Elo calculation engine with comprehensive logging"""
    
    def __init__(self, sage_script_path: Optional[str] = None, enable_logging: bool = True):
        """Initialize interface
        
        Args:
            sage_script_path: Path to main.sage (auto-detected if None)
            enable_logging: Enable comprehensive mathematical logging
        """
        if sage_script_path is None:
            # Auto-detect main.sage in same directory
            sage_script_path = Path(__file__).parent / "main.sage"
        
        self.sage_script = Path(sage_script_path)
        self.enable_logging = enable_logging
        self.logs_dir = Path(__file__).parent / "logs"
        
        # Create logs directory if it doesn't exist
        self.logs_dir.mkdir(exist_ok=True)
        
        if not self.sage_script.exists():
            raise FileNotFoundError(f"SageMath script not found: {self.sage_script}")
    
    def _create_log_entry(self, input_data: Dict[str, Any], output_data: Dict[str, Any], 
                         execution_time: float, sage_stdout: str, sage_stderr: str) -> Dict[str, Any]:
        """Create detailed log entry for SageMath execution"""
        timestamp = datetime.datetime.now().isoformat()
        
        # Extract mathematical details
        n_docs = len(input_data["document_ids"])
        n_comparisons = len(input_data["comparisons"])
        config = input_data["config"]
        
        # Analyze preference matrix structure
        preference_matrix = [[0.0 for _ in range(n_docs)] for _ in range(n_docs)]
        for comp in input_data["comparisons"]:
            i, j, weight = comp["i"], comp["j"], comp["weight"]
            preference_matrix[i][j] = weight
        
        # Calculate matrix properties
        matrix_density = n_comparisons / (n_docs * n_docs) if n_docs > 0 else 0
        diagonal_sum = sum(preference_matrix[i][i] for i in range(n_docs))
        off_diagonal_sum = sum(preference_matrix[i][j] for i in range(n_docs) for j in range(n_docs) if i != j)
        
        log_entry = {
            "timestamp": timestamp,
            "execution_metadata": {
                "sage_script": str(self.sage_script),
                "execution_time_seconds": execution_time,
                "success": True,
                "sage_stdout_length": len(sage_stdout),
                "sage_stderr_length": len(sage_stderr)
            },
            "mathematical_input": {
                "algorithm": config.get("model", "bradley-terry"),
                "convergence_parameters": {
                    "epsilon": config.get("epsilon", 1e-4),
                    "max_iterations": config.get("max_iters", 1000),
                    "learning_rate_type": "robbins_monro"
                },
                "problem_dimension": {
                    "n_documents": n_docs,
                    "n_comparisons": n_comparisons,
                    "matrix_density": matrix_density,
                    "preference_matrix_properties": {
                        "diagonal_sum": diagonal_sum,
                        "off_diagonal_sum": off_diagonal_sum,
                        "matrix_trace": diagonal_sum,
                        "matrix_symmetry_measure": self._calculate_symmetry_measure(preference_matrix)
                    }
                },
                "document_ids": input_data["document_ids"],
                "preference_matrix": preference_matrix,
                "sparse_comparisons": input_data["comparisons"]
            },
            "mathematical_output": {
                "elo_scores": output_data.get("scores", {}),
                "convergence_status": output_data.get("status", {}),
                "score_statistics": self._calculate_score_statistics(output_data.get("scores", {})),
                "mathematical_properties": self._analyze_elo_properties(output_data.get("scores", {}))
            },
            "detailed_analysis": {
                "algorithm_trace": self._extract_algorithm_trace(sage_stdout),
                "convergence_analysis": self._analyze_convergence(output_data.get("status", {})),
                "numerical_stability": self._assess_numerical_stability(preference_matrix, output_data.get("scores", {}))
            },
            "sage_output": {
                "stdout": sage_stdout,
                "stderr": sage_stderr
            }
        }
        
        return log_entry
    
    def _calculate_symmetry_measure(self, matrix: List[List[float]]) -> float:
        """Calculate how symmetric the preference matrix is"""
        n = len(matrix)
        if n == 0:
            return 1.0
        
        total_diff = 0.0
        total_pairs = 0
        
        for i in range(n):
            for j in range(i+1, n):
                diff = abs(matrix[i][j] - (1.0 - matrix[j][i]))
                total_diff += diff
                total_pairs += 1
        
        if total_pairs == 0:
            return 1.0
        
        return 1.0 - (total_diff / total_pairs)  # 1.0 = perfectly symmetric
    
    def _calculate_score_statistics(self, scores: Dict[str, float]) -> Dict[str, Any]:
        """Calculate comprehensive statistics on Elo scores"""
        if not scores:
            return {}
        
        values = list(scores.values())
        n = len(values)
        
        if n == 0:
            return {}
        
        mean_score = sum(values) / n
        variance = sum((x - mean_score) ** 2 for x in values) / n if n > 0 else 0
        std_dev = variance ** 0.5
        
        sorted_values = sorted(values)
        
        return {
            "count": n,
            "mean": mean_score,
            "std_dev": std_dev,
            "variance": variance,
            "min": min(values),
            "max": max(values),
            "range": max(values) - min(values),
            "median": sorted_values[n//2] if n > 0 else 0,
            "sum_constraint_satisfied": abs(sum(values)) < 1e-10,  # Should be ~0
            "score_distribution": {
                "q1": sorted_values[n//4] if n > 3 else sorted_values[0],
                "q3": sorted_values[3*n//4] if n > 3 else sorted_values[-1]
            }
        }
    
    def _analyze_elo_properties(self, scores: Dict[str, float]) -> Dict[str, Any]:
        """Analyze mathematical properties of the Elo scores"""
        if not scores:
            return {}
        
        values = list(scores.values())
        n = len(values)
        
        # Check for ties
        unique_values = set(round(v, 6) for v in values)  # Round to avoid floating point issues
        n_ties = n - len(unique_values)
        
        # Analyze ranking structure
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranking_gaps = []
        for i in range(len(sorted_items) - 1):
            gap = sorted_items[i][1] - sorted_items[i+1][1]
            ranking_gaps.append(gap)
        
        return {
            "unique_scores": len(unique_values),
            "tied_documents": n_ties,
            "perfect_ranking": len(unique_values) == n,  # No ties
            "ranking_gaps": ranking_gaps,
            "max_gap": max(ranking_gaps) if ranking_gaps else 0,
            "min_gap": min(ranking_gaps) if ranking_gaps else 0,
            "ranking_structure": [(item[0], item[1]) for item in sorted_items]
        }
    
    def _extract_algorithm_trace(self, sage_stdout: str) -> Dict[str, Any]:
        """Extract algorithmic details from SageMath output"""
        lines = sage_stdout.split('\n')
        
        trace = {
            "iteration_count": 0,
            "convergence_messages": [],
            "test_results": [],
            "mathematical_operations": []
        }
        
        for line in lines:
            if "Iter" in line and "Loss" in line:
                trace["iteration_count"] += 1
                trace["mathematical_operations"].append(line.strip())
            elif "Converged" in line:
                trace["convergence_messages"].append(line.strip())
            elif "TEST" in line.upper():
                trace["test_results"].append(line.strip())
        
        return trace
    
    def _analyze_convergence(self, status: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze convergence behavior and quality"""
        if not status:
            return {}
        
        converged = status.get("converged", False)
        iterations = status.get("iterations", 0)
        final_loss = status.get("final_loss", float('inf'))
        
        analysis = {
            "converged": converged,
            "iterations_used": iterations,
            "final_loss": final_loss,
            "convergence_quality": "excellent" if converged and iterations < 100 else
                                 "good" if converged and iterations < 500 else
                                 "acceptable" if converged else
                                 "poor",
            "efficiency_score": (1000 - iterations) / 1000 if iterations <= 1000 else 0,
            "numerical_precision": "high" if final_loss < 1e-8 else
                                 "medium" if final_loss < 1e-4 else
                                 "low"
        }
        
        return analysis
    
    def _assess_numerical_stability(self, preference_matrix: List[List[float]], 
                                  scores: Dict[str, float]) -> Dict[str, Any]:
        """Assess numerical stability of the computation"""
        n = len(preference_matrix)
        
        if not scores or n == 0:
            return {"status": "insufficient_data"}
        
        # Check matrix conditioning
        matrix_well_conditioned = True
        extreme_values = 0
        
        for i in range(n):
            for j in range(n):
                val = preference_matrix[i][j]
                if val < 1e-10 or val > 1 - 1e-10:
                    extreme_values += 1
        
        # Check score stability  
        score_values = list(scores.values())
        max_score = max(score_values) if score_values else 0
        min_score = min(score_values) if score_values else 0
        score_range = max_score - min_score
        
        return {
            "matrix_conditioning": "good" if extreme_values < n else "poor",
            "extreme_preference_values": extreme_values,
            "score_range": score_range,
            "score_magnitude": max(abs(max_score), abs(min_score)),
            "numerical_warnings": self._generate_numerical_warnings(preference_matrix, scores)
        }
    
    def _generate_numerical_warnings(self, preference_matrix: List[List[float]], 
                                   scores: Dict[str, float]) -> List[str]:
        """Generate warnings about potential numerical issues"""
        warnings = []
        
        n = len(preference_matrix)
        if n == 0:
            return warnings
        
        # Check for extreme preference values
        extreme_count = 0
        for i in range(n):
            for j in range(n):
                if preference_matrix[i][j] < 1e-6 or preference_matrix[i][j] > 1 - 1e-6:
                    extreme_count += 1
        
        if extreme_count > n // 2:
            warnings.append("High number of extreme preference values detected")
        
        # Check score magnitudes
        if scores:
            score_values = list(scores.values())
            max_magnitude = max(abs(s) for s in score_values)
            if max_magnitude > 50:
                warnings.append(f"Very large Elo score magnitude detected: {max_magnitude:.2f}")
        
        return warnings
    
    def _log_execution(self, log_entry: Dict[str, Any]) -> None:
        """Write log entry to both JSON and text log files"""
        if not self.enable_logging:
            return
        
        timestamp = datetime.datetime.now()
        date_str = timestamp.strftime("%Y%m%d")
        time_str = timestamp.strftime("%H%M%S")
        
        # JSON log
        json_log_path = self.logs_dir / f"sage_{date_str}.json"
        
        # Load existing logs or create new list
        if json_log_path.exists():
            with open(json_log_path, 'r') as f:
                existing_logs = json.load(f)
        else:
            existing_logs = {"executions": []}
        
        existing_logs["executions"].append(log_entry)
        
        with open(json_log_path, 'w') as f:
            json.dump(existing_logs, f, indent=2)
        
        # Text log
        text_log_path = self.logs_dir / f"sage_{date_str}.log"
        
        with open(text_log_path, 'a') as f:
            self._write_text_log_entry(f, log_entry)
    
    def _write_text_log_entry(self, f, log_entry: Dict[str, Any]) -> None:
        """Write detailed text log entry"""
        f.write("=" * 100 + "\n")
        f.write(f"SAGEMATH ELO CALCULATION LOG - {log_entry['timestamp']}\n")
        f.write("=" * 100 + "\n\n")
        
        # Execution metadata
        exec_meta = log_entry['execution_metadata']
        f.write(f"Execution Time: {exec_meta['execution_time_seconds']:.4f} seconds\n")
        f.write(f"Success: {exec_meta['success']}\n")
        f.write(f"Sage Script: {exec_meta['sage_script']}\n\n")
        
        # Mathematical input analysis
        math_input = log_entry['mathematical_input']
        f.write("MATHEMATICAL INPUT ANALYSIS\n")
        f.write("-" * 50 + "\n")
        f.write(f"Algorithm: {math_input['algorithm']}\n")
        
        conv_params = math_input['convergence_parameters']
        f.write(f"Convergence Parameters:\n")
        f.write(f"  - Epsilon: {conv_params['epsilon']}\n")
        f.write(f"  - Max Iterations: {conv_params['max_iterations']}\n")
        f.write(f"  - Learning Rate: {conv_params['learning_rate_type']}\n\n")
        
        prob_dim = math_input['problem_dimension']
        f.write(f"Problem Dimensions:\n")
        f.write(f"  - Documents: {prob_dim['n_documents']}\n")
        f.write(f"  - Comparisons: {prob_dim['n_comparisons']}\n")
        f.write(f"  - Matrix Density: {prob_dim['matrix_density']:.4f}\n\n")
        
        matrix_props = prob_dim['preference_matrix_properties']
        f.write(f"Preference Matrix Properties:\n")
        f.write(f"  - Diagonal Sum: {matrix_props['diagonal_sum']:.6f}\n")
        f.write(f"  - Off-Diagonal Sum: {matrix_props['off_diagonal_sum']:.6f}\n")
        f.write(f"  - Matrix Trace: {matrix_props['matrix_trace']:.6f}\n")
        f.write(f"  - Symmetry Measure: {matrix_props['matrix_symmetry_measure']:.6f}\n\n")
        
        # Preference matrix
        f.write("PREFERENCE MATRIX:\n")
        f.write("-" * 30 + "\n")
        pref_matrix = math_input['preference_matrix']
        n = len(pref_matrix)
        
        # Header
        f.write("     ")
        for j in range(n):
            f.write(f"  Doc{j:2d} ")
        f.write("\n")
        
        # Matrix rows
        for i in range(n):
            f.write(f"Doc{i:2d}")
            for j in range(n):
                f.write(f" {pref_matrix[i][j]:6.3f}")
            f.write("\n")
        f.write("\n")
        
        # Document mapping
        f.write("DOCUMENT MAPPING:\n")
        f.write("-" * 30 + "\n")
        for i, doc_id in enumerate(math_input['document_ids']):
            f.write(f"Doc{i:2d} -> {doc_id}\n")
        f.write("\n")
        
        # Mathematical output analysis
        math_output = log_entry['mathematical_output']
        f.write("MATHEMATICAL OUTPUT ANALYSIS\n")
        f.write("-" * 50 + "\n")
        
        elo_scores = math_output['elo_scores']
        f.write("ELO SCORES:\n")
        sorted_scores = sorted(elo_scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (doc_id, score) in enumerate(sorted_scores, 1):
            f.write(f"  {rank}. {doc_id}: {score:+.12f}\n")
        f.write("\n")
        
        # Score statistics
        stats = math_output['score_statistics']
        if stats:
            f.write("SCORE STATISTICS:\n")
            f.write(f"  - Count: {stats['count']}\n")
            f.write(f"  - Mean: {stats['mean']:+.12f}\n")
            f.write(f"  - Std Dev: {stats['std_dev']:.12f}\n")
            f.write(f"  - Range: {stats['range']:.12f}\n")
            f.write(f"  - Min: {stats['min']:+.12f}\n")
            f.write(f"  - Max: {stats['max']:+.12f}\n")
            f.write(f"  - Sum Constraint OK: {stats['sum_constraint_satisfied']}\n\n")
        
        # Mathematical properties
        props = math_output['mathematical_properties']
        if props:
            f.write("MATHEMATICAL PROPERTIES:\n")
            f.write(f"  - Unique Scores: {props['unique_scores']}\n")
            f.write(f"  - Tied Documents: {props['tied_documents']}\n")
            f.write(f"  - Perfect Ranking: {props['perfect_ranking']}\n")
            f.write(f"  - Max Gap: {props['max_gap']:.12f}\n")
            f.write(f"  - Min Gap: {props['min_gap']:.12f}\n\n")
        
        # Convergence analysis
        conv_status = math_output['convergence_status']
        f.write("CONVERGENCE STATUS:\n")
        f.write(f"  - Converged: {conv_status.get('converged', False)}\n")
        f.write(f"  - Iterations: {conv_status.get('iterations', 0)}\n")
        f.write(f"  - Final Loss: {conv_status.get('final_loss', 'N/A')}\n")
        f.write(f"  - Time: {conv_status.get('time_seconds', 0):.4f}s\n")
        f.write(f"  - Model: {conv_status.get('model', 'N/A')}\n\n")
        
        # Detailed analysis
        detailed = log_entry['detailed_analysis']
        
        # Algorithm trace
        trace = detailed['algorithm_trace']
        if trace['mathematical_operations']:
            f.write("ALGORITHM TRACE:\n")
            f.write("-" * 30 + "\n")
            for op in trace['mathematical_operations'][:10]:  # Show first 10 iterations
                f.write(f"  {op}\n")
            if len(trace['mathematical_operations']) > 10:
                f.write(f"  ... and {len(trace['mathematical_operations']) - 10} more iterations\n")
            f.write("\n")
        
        if trace['convergence_messages']:
            f.write("CONVERGENCE MESSAGES:\n")
            for msg in trace['convergence_messages']:
                f.write(f"  {msg}\n")
            f.write("\n")
        
        # Convergence analysis
        conv_analysis = detailed['convergence_analysis']
        if conv_analysis:
            f.write("CONVERGENCE ANALYSIS:\n")
            f.write(f"  - Quality: {conv_analysis.get('convergence_quality', 'unknown')}\n")
            f.write(f"  - Efficiency Score: {conv_analysis.get('efficiency_score', 0):.4f}\n")
            f.write(f"  - Numerical Precision: {conv_analysis.get('numerical_precision', 'unknown')}\n\n")
        
        # Numerical stability
        stability = detailed['numerical_stability']
        if stability:
            f.write("NUMERICAL STABILITY ASSESSMENT:\n")
            f.write(f"  - Matrix Conditioning: {stability.get('matrix_conditioning', 'unknown')}\n")
            f.write(f"  - Extreme Values: {stability.get('extreme_preference_values', 0)}\n")
            f.write(f"  - Score Range: {stability.get('score_range', 0):.6f}\n")
            f.write(f"  - Score Magnitude: {stability.get('score_magnitude', 0):.6f}\n")
            
            warnings = stability.get('numerical_warnings', [])
            if warnings:
                f.write("  - Warnings:\n")
                for warning in warnings:
                    f.write(f"    * {warning}\n")
            f.write("\n")
        
        # Raw SageMath output
        sage_output = log_entry['sage_output']
        if sage_output['stdout']:
            f.write("SAGEMATH STDOUT:\n")
            f.write("-" * 30 + "\n")
            f.write(sage_output['stdout'])
            f.write("\n\n")
        
        if sage_output['stderr']:
            f.write("SAGEMATH STDERR:\n")
            f.write("-" * 30 + "\n")
            f.write(sage_output['stderr'])
            f.write("\n\n")
        
        f.write("=" * 100 + "\n\n")
    
    def calculate_elos(self, 
                      document_ids: List[str], 
                      comparisons: List[Dict[str, Any]], 
                      model: str = "bradley-terry",
                      epsilon: float = 1e-4,
                      max_iters: int = 1000,
                      verbose: bool = False) -> Dict[str, Any]:
        """
        Calculate Elo scores using SageMath engine
        
        Args:
            document_ids: List of document identifiers
            comparisons: List of comparisons, each with 'i', 'j', 'weight'
            model: Model type ('bradley-terry' or 'thurstone')
            epsilon: Convergence threshold
            max_iters: Maximum iterations
            verbose: Print detailed output
        
        Returns:
            Dict with 'scores' and 'status' keys
        
        Raises:
            RuntimeError: If SageMath calculation fails
        """
        # Prepare input data
        input_data = {
            "document_ids": document_ids,
            "comparisons": comparisons,
            "config": {
                "model": model,
                "epsilon": epsilon,
                "max_iters": max_iters,
                "verbose": verbose
            }
        }
        
        # Use temporary files for communication
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as input_file:
            json.dump(input_data, input_file)
            input_path = input_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as output_file:
            output_path = output_file.name
        
        try:
            # Time the execution
            start_time = time.time()
            
            # Call SageMath
            result = subprocess.run(
                ["sage", str(self.sage_script), input_path, output_path],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            execution_time = time.time() - start_time
            
            if result.returncode != 0:
                raise RuntimeError(f"SageMath calculation failed: {result.stderr}")
            
            # Read results
            with open(output_path, 'r') as f:
                output_data = json.load(f)
            
            # Create and log detailed execution record
            if self.enable_logging:
                log_entry = self._create_log_entry(
                    input_data, output_data, execution_time, 
                    result.stdout, result.stderr
                )
                self._log_execution(log_entry)
            
            return output_data
            
        finally:
            # Clean up temp files
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def calculate_elos_from_matrix(self, 
                                  preference_matrix: List[List[float]], 
                                  document_ids: Optional[List[str]] = None,
                                  **kwargs) -> Dict[str, Any]:
        """
        Calculate Elos from a full preference matrix
        
        Args:
            preference_matrix: n x n preference matrix
            document_ids: Document IDs (auto-generated if None)
            **kwargs: Additional arguments for calculate_elos
        
        Returns:
            Dict with 'scores' and 'status' keys
        """
        n = len(preference_matrix)
        
        if document_ids is None:
            document_ids = [f"doc_{i}" for i in range(n)]
        
        if len(document_ids) != n:
            raise ValueError(f"Document IDs length ({len(document_ids)}) doesn't match matrix size ({n})")
        
        # Convert matrix to sparse comparisons
        comparisons = []
        for i in range(n):
            for j in range(n):
                if preference_matrix[i][j] > 0 and i != j:  # Skip diagonal and zeros
                    comparisons.append({
                        "i": i,
                        "j": j,
                        "weight": float(preference_matrix[i][j])
                    })
        
        return self.calculate_elos(document_ids, comparisons, **kwargs)
    
    def test_connection(self) -> bool:
        """
        Test if SageMath is available and working
        
        Returns:
            True if test passes, False otherwise
        """
        try:
            # Simple test calculation
            result = self.calculate_elos(
                document_ids=["test_doc_1", "test_doc_2"],
                comparisons=[{"i": 0, "j": 1, "weight": 1.0}],
                verbose=False
            )
            
            # Verify we got reasonable output
            return (
                'scores' in result and 
                'status' in result and
                len(result['scores']) == 2
            )
        except Exception:
            return False

# Convenience function for ZBench integration
def calculate_elos_sage(preference_matrix: List[List[float]], 
                       document_ids: Optional[List[str]] = None,
                       **kwargs) -> tuple:
    """
    ZBench-compatible wrapper function
    
    Args:
        preference_matrix: n x n numpy-like array
        document_ids: Document IDs (optional)
        **kwargs: Additional configuration
    
    Returns:
        Tuple of (elos_array, status_dict)
    """
    interface = SageMathInterface()
    
    # Convert numpy array to list if needed
    if hasattr(preference_matrix, 'tolist'):
        matrix_list = preference_matrix.tolist()
    else:
        matrix_list = preference_matrix
    
    result = interface.calculate_elos_from_matrix(matrix_list, document_ids, **kwargs)
    
    # Extract Elo scores in order
    n = len(matrix_list)
    if document_ids is None:
        document_ids = [f"doc_{i}" for i in range(n)]
    
    elos = [result['scores'][doc_id] for doc_id in document_ids]
    
    # Convert to numpy array if numpy is available
    try:
        import numpy as np
        elos = np.array(elos)
    except ImportError:
        pass  # Return as list if numpy not available
    
    return elos, result['status']

def main():
    """CLI interface for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SageMath Elo calculation interface")
    parser.add_argument("--test", action="store_true", help="Run connection test")
    parser.add_argument("--input", help="Input JSON file")
    parser.add_argument("--output", help="Output JSON file")
    args = parser.parse_args()
    
    interface = SageMathInterface()
    
    if args.test:
        print("Testing SageMath connection...")
        if interface.test_connection():
            print("✓ SageMath connection successful")
        else:
            print("✗ SageMath connection failed")
            return 1
    
    elif args.input and args.output:
        # Process file
        with open(args.input, 'r') as f:
            input_data = json.load(f)
        
        result = interface.calculate_elos(**input_data)
        
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Results written to {args.output}")
    
    else:
        # Demo calculation
        print("SageMath Elo Calculation Demo")
        print("-" * 40)
        
        result = interface.calculate_elos(
            document_ids=["Document A", "Document B", "Document C"],
            comparisons=[
                {"i": 0, "j": 1, "weight": 1.0},  # A beats B
                {"i": 0, "j": 2, "weight": 0.8},  # A beats C
                {"i": 1, "j": 2, "weight": 0.6}   # B beats C
            ],
            verbose=True
        )
        
        print("\nResults:")
        for doc_id, score in result['scores'].items():
            print(f"  {doc_id}: {score:.6f}")
        
        status = result['status']
        print(f"\nStatus: Converged={status['converged']}, Iterations={status['iterations']}")
    
    return 0

if __name__ == "__main__":
    exit(main())