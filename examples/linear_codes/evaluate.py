"""
Evaluator for linear codes
"""

import os
import json
import argparse
import numpy as np
import galois
from typing import Tuple, Optional, List, Dict, Any

from shinka.core import run_shinka_eval

def adapted_validate_codes(
    run_output: Tuple[np.ndarray, List[float], int],
    atol=1e-6,
) -> Tuple[bool, Optional[str]]:
    """
    Validates linear codes results based on the output of 'run_codes'.

    Args:
        run_output: Tuple (centers, radii, reported_sum) from run_codes.

    Returns:
        (is_valid: bool, error_message: Optional[str])
    """
    code, weight_dist, minimum_dist = run_output

    n=80
    k=20
    galois_code = galois.GF(2)(code)

    if not isinstance(code, np.ndarray):
        return False, f"OUTPUT_STRUCT ERROR: The output structure is not a numpy array: {code}"
    if not code.shape == (k, n):
        return False, f"SIZE ERROR: The output structure is not of the right size: {code}"
    if np.linalg.matrix_rank(galois_code) != k:
        return False, f"RANK ERROR: The output structure is a degenerate matrix: {code}"
    return True, "All the checks were passed, continuing"


def get_linear_codes_kwargs(run_index: int) -> Dict[str, Any]:
    """Provides keyword arguments for linear codes runs (none needed)."""
    return {}

def aggregate_linear_codes_metrics(
    results: List[Tuple[np.ndarray, List[float], int]], results_dir: str
) -> Dict[str, Any]:
    """
    Aggregates metrics for linear codes. Assumes num_runs=1.
    Saves extra.npz with detailed coding information. 
    """
    if not results:
        return {"combined_score": 0.0, "error": "No results to aggregate"}
    dist=26
    code, weight_dist, minimum_dist = results[0]
    new_score = - sum([weight_dist[j] * 0.92 ** (j + 1) for j in range(1, dist+1)])
    weight_dist=np.array(weight_dist)

    public_metrics = {
        "minimum_dist": minimum_dist,
        "new_score": new_score,
    }
    private_metrics = {
        "new_score": new_score,
    }
    metrics = {
        "combined_score": new_score,
        "public": public_metrics,
        "private": private_metrics,
    }

    extra_file = os.path.join(results_dir, "extra.npz")
    try:
        np.savez(
            extra_file,
            code=code,
            weight_dist=weight_dist,
            minimum_dist=minimum_dist,
            new_score=new_score,
        )
        print(f"Detailed coding data saved to {extra_file}")
    except Exception as e:
        print(f"Error saving extra.npz: {e}")  
        metrics["extra_npz_save_error"] = str(e)

    return metrics


def main(program_path: str, results_dir: str):
    """Runs the linear codes evaluation using shinka.eval."""
    print(f"Evaluating program: {program_path}")
    print(f"Saving results to: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)

    num_experiment_runs = 1

    # Define a nested function to pass results_dir to the aggregator
    def _aggregator_with_context(
        r: List[Tuple[np.ndarray, List[float], int]],
    ) -> Dict[str, Any]:
        return aggregate_linear_codes_metrics(r, results_dir)
    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_codes",
        num_runs=num_experiment_runs,
        get_experiment_kwargs=get_linear_codes_kwargs,
        validate_fn=adapted_validate_codes,
        aggregate_metrics_fn=_aggregator_with_context,
    )
    if correct:
        print("Evaluation and Validation completed successfully.")
    else:
        print(f"Evaluation or Validation failed: {error_msg}")

    print("Metrics:")
    for key, value in metrics.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"  {key}: <string_too_long_to_display>")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Linear codes evaluator using shinka.eval"
    )
    parser.add_argument(
        "--program_path",
        type=str,
        default="initial.py",
        help="Path to program to evaluate (must contain 'run_codes')",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Dir to save results (metrics.json, correct.json, extra.npz)",
    )
    parsed_args = parser.parse_args()
    print(parsed_args.program_path, parsed_args.results_dir)
    main(parsed_args.program_path, parsed_args.results_dir)
