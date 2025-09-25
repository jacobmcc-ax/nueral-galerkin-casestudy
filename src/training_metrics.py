#!/usr/bin/env python3
"""
Training Metrics and Loss Tracking for Neural Galerkin Methods
Tracks and visualizes training losses, convergence, and solution accuracy
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Tuple
import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Add utils to path for logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.logger import log_code_execution, log_mathematical_result

class TrainingMetrics:
    """
    Comprehensive training metrics tracking for Neural Galerkin methods

    Tracks multiple types of losses and metrics:
    - Residual loss (PDE residual norm)
    - Solution error (vs analytical solution)
    - Parameter evolution
    - Convergence rates
    - Computational efficiency
    """

    def __init__(self, save_dir: str = "training_results"):
        """
        Initialize training metrics tracker

        Args:
            save_dir: Directory to save metrics and plots
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        # Training history storage
        self.history = {
            'residual_loss': [],
            'solution_error': [],
            'parameter_norm': [],
            'time_steps': [],
            'iterations': [],
            'convergence_rate': [],
            'sparse_efficiency': [],
            'computation_time': []
        }

        # Current training session info
        self.session_info = {
            'start_time': datetime.now(),
            'algorithm': 'RSNG',
            'pde_type': None,
            'spatial_points': None,
            'time_span': None
        }

        log_code_execution(
            "TrainingMetrics.__init__",
            f"Initialized metrics tracking, save_dir: {self.save_dir}"
        )

    def log_training_step(self,
                         step: int,
                         residual_loss: float,
                         solution_error: Optional[float] = None,
                         parameter_norm: Optional[float] = None,
                         computation_time: Optional[float] = None,
                         sparse_efficiency: Optional[float] = None):
        """
        Log metrics for a single training step

        Args:
            step: Training step/iteration number
            residual_loss: PDE residual norm
            solution_error: Error vs analytical solution
            parameter_norm: Norm of neural network parameters
            computation_time: Time taken for this step
            sparse_efficiency: Computational efficiency from sparsity
        """
        self.history['iterations'].append(step)
        self.history['residual_loss'].append(residual_loss)

        if solution_error is not None:
            self.history['solution_error'].append(solution_error)

        if parameter_norm is not None:
            self.history['parameter_norm'].append(parameter_norm)

        if computation_time is not None:
            self.history['computation_time'].append(computation_time)

        if sparse_efficiency is not None:
            self.history['sparse_efficiency'].append(sparse_efficiency)

        # Compute convergence rate
        if len(self.history['residual_loss']) > 1:
            current_loss = self.history['residual_loss'][-1]
            previous_loss = self.history['residual_loss'][-2]
            convergence_rate = np.log10(current_loss / previous_loss) if previous_loss > 0 else 0
            self.history['convergence_rate'].append(convergence_rate)

        error_str = f"{solution_error:.2e}" if solution_error is not None else "N/A"
        log_mathematical_result(
            f"Training step {step}",
            "INFO",
            f"Residual: {residual_loss:.2e}, Error: {error_str}",
            "Training progress"
        )

    def log_time_step(self, time: float, residual: float, solution_error: Optional[float] = None):
        """
        Log metrics for a time evolution step

        Args:
            time: Current time value
            residual: PDE residual at this time
            solution_error: Solution error vs analytical
        """
        self.history['time_steps'].append(time)

        # Also log as training step
        step = len(self.history['time_steps'])
        self.log_training_step(step, residual, solution_error)

    def set_session_info(self, **kwargs):
        """Set information about current training session"""
        self.session_info.update(kwargs)

        log_code_execution(
            "TrainingMetrics.set_session_info",
            f"Updated session info: {list(kwargs.keys())}"
        )

    def plot_training_losses(self, save_plot: bool = True, show_plot: bool = True) -> str:
        """
        Create comprehensive training loss plots

        Args:
            save_plot: Whether to save plot to file
            show_plot: Whether to display plot

        Returns:
            Path to saved plot file
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Neural Galerkin Training Metrics', fontsize=16, fontweight='bold')

        # Use consistent iteration axis for all plots
        main_iterations = self.history['iterations'] or range(len(self.history['residual_loss']))

        # Plot 1: Residual Loss vs Iterations
        ax1 = axes[0, 0]
        if self.history['residual_loss']:
            ax1.semilogy(main_iterations, self.history['residual_loss'], 'b-o', linewidth=2, markersize=4)
            ax1.set_xlabel('Training Iteration')
            ax1.set_ylabel('Residual Loss (log scale)')
            ax1.set_title('PDE Residual Loss Convergence')
            ax1.grid(True, alpha=0.3)

            # Add convergence rate annotation
            if len(self.history['residual_loss']) > 1:
                final_loss = self.history['residual_loss'][-1]
                ax1.annotate(f'Final: {final_loss:.2e}',
                           xy=(main_iterations[-1], final_loss),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        # Plot 2: Solution Error vs Iterations (use same iteration axis)
        ax2 = axes[0, 1]
        if self.history['solution_error']:
            # Match solution error data to main iteration axis by taking the right subset
            if len(self.history['solution_error']) < len(main_iterations):
                # If we have fewer solution error points, space them evenly across main iterations
                error_iterations = np.linspace(0, main_iterations[-1], len(self.history['solution_error']))
            else:
                # If we have same or more points, use the main iteration axis
                error_iterations = main_iterations[:len(self.history['solution_error'])]

            ax2.semilogy(error_iterations, self.history['solution_error'], 'r-s', linewidth=2, markersize=4)
            ax2.set_xlabel('Training Iteration')
            ax2.set_ylabel('Solution Error (log scale)')
            ax2.set_title('Solution Accuracy vs Analytical')
            ax2.grid(True, alpha=0.3)

        # Plot 3: Convergence Rate (use main iteration axis)
        ax3 = axes[1, 0]
        if self.history['convergence_rate']:
            # Match convergence rate data to main iterations
            if len(self.history['convergence_rate']) < len(main_iterations):
                conv_iterations = np.linspace(0, main_iterations[-1], len(self.history['convergence_rate']))
            else:
                conv_iterations = main_iterations[:len(self.history['convergence_rate'])]

            ax3.plot(conv_iterations, self.history['convergence_rate'], 'g-^', linewidth=2, markersize=4)
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax3.set_xlabel('Training Iteration')
            ax3.set_ylabel('Convergence Rate (log10)')
            ax3.set_title('Convergence Rate (Negative = Improving)')
            ax3.grid(True, alpha=0.3)

        # Plot 4: Parameter Evolution and Efficiency (use consistent iteration axis)
        ax4 = axes[1, 1]
        if self.history['parameter_norm'] or self.history['computation_time']:
            ax4_twin = ax4.twinx()

            # Parameter norm (match to main iterations)
            if self.history['parameter_norm']:
                if len(self.history['parameter_norm']) < len(main_iterations):
                    param_iterations = np.linspace(0, main_iterations[-1], len(self.history['parameter_norm']))
                else:
                    param_iterations = main_iterations[:len(self.history['parameter_norm'])]

                line1 = ax4.plot(param_iterations, self.history['parameter_norm'], 'purple',
                               linewidth=2, label='Parameter Norm')
                ax4.set_ylabel('Parameter Norm', color='purple')
                ax4.tick_params(axis='y', labelcolor='purple')

            # Computation time (match to main iterations)
            if self.history['computation_time']:
                if len(self.history['computation_time']) < len(main_iterations):
                    time_iterations = np.linspace(0, main_iterations[-1], len(self.history['computation_time']))
                else:
                    time_iterations = main_iterations[:len(self.history['computation_time'])]

                line2 = ax4_twin.plot(time_iterations, self.history['computation_time'], 'orange',
                                    linewidth=2, label='Computation Time')
                ax4_twin.set_ylabel('Time (s)', color='orange')
                ax4_twin.tick_params(axis='y', labelcolor='orange')

            ax4.set_xlabel('Training Iteration')
            ax4.set_title('Parameter Evolution & Efficiency')
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_filename = f"training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plot_path = self.save_dir / plot_filename

        if save_plot:
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            log_code_execution(
                "TrainingMetrics.plot_training_losses",
                f"Saved training plot to {plot_path}"
            )

        if show_plot:
            plt.show()
        else:
            plt.close()

        return str(plot_path)

    def plot_time_evolution(self, save_plot: bool = True, show_plot: bool = True) -> str:
        """
        Plot metrics evolution over time (for time-dependent PDEs)

        Args:
            save_plot: Whether to save plot
            show_plot: Whether to display plot

        Returns:
            Path to saved plot file
        """
        if not self.history['time_steps']:
            log_code_execution(
                "TrainingMetrics.plot_time_evolution",
                "No time evolution data available"
            )
            return ""

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle('Time Evolution of Training Metrics', fontsize=16, fontweight='bold')

        time_points = self.history['time_steps']

        # Residual vs Time
        ax1.semilogy(time_points, self.history['residual_loss'][:len(time_points)],
                     'b-o', linewidth=2, markersize=4)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Residual Loss (log scale)')
        ax1.set_title('PDE Residual Evolution Over Time')
        ax1.grid(True, alpha=0.3)

        # Solution Error vs Time
        if len(self.history['solution_error']) >= len(time_points):
            ax2.semilogy(time_points, self.history['solution_error'][:len(time_points)],
                         'r-s', linewidth=2, markersize=4)
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Solution Error (log scale)')
            ax2.set_title('Solution Accuracy Over Time')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_filename = f"time_evolution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plot_path = self.save_dir / plot_filename

        if save_plot:
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            log_code_execution(
                "TrainingMetrics.plot_time_evolution",
                f"Saved time evolution plot to {plot_path}"
            )

        if show_plot:
            plt.show()
        else:
            plt.close()

        return str(plot_path)

    def save_metrics(self, filename: Optional[str] = None) -> str:
        """
        Save training metrics to JSON file

        Args:
            filename: Optional custom filename

        Returns:
            Path to saved metrics file
        """
        if filename is None:
            filename = f"training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = self.save_dir / filename

        # Prepare data for JSON serialization
        metrics_data = {
            'session_info': {
                'start_time': self.session_info['start_time'].isoformat(),
                'algorithm': self.session_info['algorithm'],
                'pde_type': self.session_info['pde_type'],
                'spatial_points': self.session_info['spatial_points'],
                'time_span': self.session_info['time_span']
            },
            'training_history': self.history,
            'summary_statistics': self.get_summary_statistics()
        }

        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)

        log_code_execution(
            "TrainingMetrics.save_metrics",
            f"Saved metrics to {filepath}"
        )

        return str(filepath)

    def get_summary_statistics(self) -> Dict:
        """
        Compute summary statistics for training session

        Returns:
            Dictionary with summary stats
        """
        stats = {}

        if self.history['residual_loss']:
            residuals = np.array(self.history['residual_loss'])
            stats['residual_stats'] = {
                'initial': float(residuals[0]),
                'final': float(residuals[-1]),
                'min': float(np.min(residuals)),
                'mean': float(np.mean(residuals)),
                'reduction_ratio': float(residuals[0] / residuals[-1]) if residuals[-1] > 0 else float('inf')
            }

        if self.history['solution_error']:
            errors = np.array(self.history['solution_error'])
            stats['error_stats'] = {
                'final': float(errors[-1]),
                'min': float(np.min(errors)),
                'mean': float(np.mean(errors))
            }

        if self.history['computation_time']:
            times = np.array(self.history['computation_time'])
            stats['efficiency_stats'] = {
                'total_time': float(np.sum(times)),
                'avg_time_per_step': float(np.mean(times)),
                'total_steps': len(times)
            }

        stats['convergence_achieved'] = (
            self.history['residual_loss'][-1] < 1e-6
            if self.history['residual_loss'] else False
        )

        return stats

    def print_summary(self):
        """Print training summary to console"""
        print("=" * 60)
        print("NEURAL GALERKIN TRAINING SUMMARY")
        print("=" * 60)

        stats = self.get_summary_statistics()

        if 'residual_stats' in stats:
            rs = stats['residual_stats']
            print(f"Residual Loss:")
            print(f"  Initial: {rs['initial']:.2e}")
            print(f"  Final:   {rs['final']:.2e}")
            print(f"  Reduction: {rs['reduction_ratio']:.2e}x")

        if 'error_stats' in stats:
            es = stats['error_stats']
            print(f"Solution Error:")
            print(f"  Final:   {es['final']:.2e}")
            print(f"  Minimum: {es['min']:.2e}")

        if 'efficiency_stats' in stats:
            eff = stats['efficiency_stats']
            print(f"Computational Efficiency:")
            print(f"  Total steps: {eff['total_steps']}")
            print(f"  Total time:  {eff['total_time']:.2f}s")
            print(f"  Avg per step: {eff['avg_time_per_step']:.4f}s")

        print(f"Convergence: {'✅ Achieved' if stats['convergence_achieved'] else '❌ Not achieved'}")
        print("=" * 60)

# Example usage and integration
if __name__ == "__main__":
    print("=" * 60)
    print("Training Metrics Demonstration")
    print("=" * 60)

    # Create metrics tracker
    metrics = TrainingMetrics(save_dir="demo_training_results")

    # Set session info
    metrics.set_session_info(
        pde_type="heat",
        spatial_points=50,
        time_span=(0.0, 0.1)
    )

    # Simulate training progression
    print("Simulating training progression...")

    np.random.seed(42)  # For reproducible demo

    # Generate realistic training data
    n_steps = 50
    initial_residual = 1e-2

    for step in range(n_steps):
        # Simulate decreasing residual with some noise
        progress = step / n_steps
        residual = initial_residual * np.exp(-3 * progress) * (1 + 0.1 * np.random.randn())
        residual = max(residual, 1e-8)  # Minimum bound

        # Simulate solution error
        solution_error = residual * 0.1 * (1 + 0.2 * np.random.randn())
        solution_error = max(solution_error, 1e-10)

        # Simulate parameter norm
        parameter_norm = 1.0 + 0.1 * np.sin(step * 0.2) + 0.05 * np.random.randn()

        # Simulate computation time
        comp_time = 0.01 + 0.002 * np.random.randn()
        comp_time = max(comp_time, 0.005)

        # Log metrics
        metrics.log_training_step(
            step=step,
            residual_loss=residual,
            solution_error=solution_error,
            parameter_norm=parameter_norm,
            computation_time=comp_time,
            sparse_efficiency=0.8  # 80% efficiency from sparsity
        )

        # Also simulate time evolution for some steps
        if step % 5 == 0:
            time_point = step * 0.002  # 0.1 / 50
            metrics.log_time_step(time_point, residual, solution_error)

    # Generate plots
    print("Generating training plots...")
    plot_path = metrics.plot_training_losses(show_plot=False)
    print(f"Training metrics plot saved to: {plot_path}")

    time_plot_path = metrics.plot_time_evolution(show_plot=False)
    print(f"Time evolution plot saved to: {time_plot_path}")

    # Save metrics
    metrics_path = metrics.save_metrics()
    print(f"Metrics data saved to: {metrics_path}")

    # Print summary
    metrics.print_summary()

    print("\n✅ Training metrics demonstration complete!")
    print("Check the 'demo_training_results' directory for generated plots and data.")