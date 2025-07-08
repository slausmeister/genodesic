import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Tuple

from .PathEvaluation import (
    calculate_euclidean_segment_lengths,
    calculate_path_density,
    calculate_fermat_length
)

def calculate_path_statistics(
    paths_dict: Dict[str, torch.Tensor],
    model: Any,
    beta_param: float,
    integration_steps: int
) -> Dict[str, Any]:
    """
    Calculates key statistics for one or more paths without plotting.

    This function is dependency-free from any plotting libraries.

    Args:
        paths_dict (Dict[str, torch.Tensor]): A dictionary where keys are path names
                                             and values are the path data tensors (L, D).
        model: The model used for density and Fermat length calculations.
        beta_param (float): The hyperparameter for the Fermat metric.
        integration_steps (int): The number of integration steps for Fermat length.

    Returns:
        Dict[str, Any]: A nested dictionary containing the calculated statistics for each path.
    """
    print("--- Calculating Path Statistics ---")
    results = {}

    for name, phi in paths_dict.items():
        print(f"Processing '{name}'...")
        
        # Perform all core calculations
        lengths = calculate_euclidean_segment_lengths(phi)
        densities = calculate_path_density(phi, model, 3)
        fermat_len = calculate_fermat_length(phi, model, beta_param, integration_steps)
        
        # Store results in a structured dictionary
        results[name] = {
            'lengths': lengths,
            'densities': densities,
            'fermat_length': fermat_len,
            'total_length': np.sum(lengths),
            'mean_log_likelihood': np.mean(densities),
            'std_log_likelihood': np.std(densities),
            'length_stats': {
                'mean': np.mean(lengths),
                'std': np.std(lengths),
                'min': np.min(lengths),
                'max': np.max(lengths),
            }
        }
    print("Calculations complete.\n")
    return results

def report_path_statistics(
    stats_dict: Dict[str, Any],
    beta_param: float,
    reference_path_name: Optional[str] = None,
    figsize: tuple = (12, 8)
):
    """
    Generates plots and print summaries from a pre-calculated statistics dictionary.

    This function handles all user-facing output, including matplotlib plots
    and formatted print statements.

    Args:
        stats_dict (Dict[str, Any]): The output from calculate_path_statistics.
        beta_param (float): The Fermat metric hyperparameter (for display purposes).
        reference_path_name (str, optional): Name of the path to use as a baseline
                                             for length comparison. Defaults to the first path.
        figsize (tuple, optional): The size of the output plot.
    """
    # --- 1. Plot Log-Density Distributions ---
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(stats_dict)))
    
    text_str = ""
    for (name, stats), color in zip(stats_dict.items(), colors):
        ax.hist(stats['densities'], bins=40, alpha=0.7, label=name, density=True, color=color)
        text_str += f"Mean Log-Likelihood ({name}): {stats['mean_log_likelihood']:.2f}\n"

    ax.set_title('Distribution of Midpoint Log-Densities', fontsize=16)
    ax.set_xlabel('Log-Density', fontsize=12)
    ax.set_ylabel('Normalized Frequency', fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
    ax.text(0.05, 0.95, text_str.strip(), transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    plt.tight_layout(pad=3.0)
    plt.show()

    # --- 2. Print Descriptive Statistics ---
    print("--- Path Statistics ---")
    for name, stats in stats_dict.items():
        ls = stats['length_stats']
        print(f"\n{name}:")
        print(f"  Euclidean Length: Mean={ls['mean']:.3f}, Std={ls['std']:.3f}, Min={ls['min']:.3f}, Max={ls['max']:.3f}, Total={stats['total_length']:.3f}")
        print(f"  Log-Density:      Mean={stats['mean_log_likelihood']:.3f}, Std={stats['std_log_likelihood']:.3f}")
        print(f"  Fermat Length:    {stats['fermat_length']:.3f} (beta={beta_param})")

    # --- 3. Compare Total Arc Lengths ---
    if len(stats_dict) > 1:
        if not reference_path_name or reference_path_name not in stats_dict:
            reference_path_name = list(stats_dict.keys())[0]

        ref_length = stats_dict[reference_path_name]['total_length']
        
        print("\n--- Total Arc Length Comparison ---")
        print(f"Reference Path: '{reference_path_name}' (Total Length: {ref_length:.3f})")

        for name, stats in stats_dict.items():
            if name == reference_path_name:
                continue
            
            change = ((stats['total_length'] - ref_length) / ref_length) * 100
            print(f"  vs. '{name}': {stats['total_length']:.3f} (Change: {change:+.2f}%)")





def plot_pseudotime_progressions(
    results: List[Tuple[torch.Tensor, torch.Tensor, str]], 
    colors: List[str] = None
):
    """
    Plots multiple pseudotime progressions on a single graph with a normalized x-axis.
    
    Args:
        results: A list of tuples, where each tuple contains
                 (avg_times, std_times, label_for_path).
        colors: An optional list of colors for the plots.
    """
    print("Plotting multiple pseudotime progressions on a normalized axis...")
    plt.figure(figsize=(12, 6))
    
    if colors is None:
        colors = ['blue', 'green', 'red', 'purple', 'orange']

    if len(colors) < len(results):
        raise ValueError("Not enough colors provided for the number of results.")

    for i, (avg_times, std_times, label) in enumerate(results):
        # Normalize the x-axis to be from 0.0 to 1.0
        x = torch.linspace(0, 1, len(avg_times))
        
        lower_bound = avg_times - std_times
        upper_bound = avg_times + std_times
        color = colors[i]

        plt.fill_between(x, lower_bound, upper_bound, color=color, alpha=0.2)
        plt.plot(x, avg_times, color=color, label=label)
        plt.scatter(x, avg_times, s=10, color=color)

    plt.title('Pseudotime Progression Along Geodesic Paths')
    
    plt.xlabel('Normalized Path Progression')
    
    plt.ylabel('Average Pseudotime')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()