#!/usr/bin/env python3
"""
RSNG (Randomized Sparse Neural Galerkin) Demonstration

Implementation of the algorithm from the paper:
"Randomized Sparse Neural Galerkin Schemes for Solving Evolution Equations with Deep Networks"

This demonstrates the proper sequential-in-time approach where:
1. Network represents u(x; θ(t)) at single time points
2. Parameters θ(t) evolve according to PDE dynamics
3. Sparse random updates prevent overfitting and reduce computational cost
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from rsng_neural_approximator import RSNGNeuralApproximator, RSNGSolver, PDERightHandSide

def analytical_solution(x, t):
    """Analytical solution for heat equation: u_t = u_xx with initial condition"""
    # Single mode solution: u(x,t) = exp(-π²t) * sin(πx)
    return np.exp(-np.pi**2 * t) * np.sin(np.pi * x)

def create_rsng_solution_comparison(x_points, t_points, rsng_solutions, analytical_solutions, save_path):
    """Create solution comparison visualization"""
    # Select time points for 3x3 grid
    selected_times = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('RSNG vs Analytical Solution Comparison (Heat Equation)', fontsize=16, fontweight='bold')

    for idx, t_target in enumerate(selected_times):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        # Find closest time point
        time_idx = np.argmin(np.abs(t_points - t_target))
        t_actual = t_points[time_idx]

        # Get solutions at this time
        rsng_sol = rsng_solutions[time_idx]
        analytical_sol = analytical_solutions[time_idx]

        # Compute error metrics
        l2_error = np.sqrt(np.mean((rsng_sol - analytical_sol)**2))
        linf_error = np.max(np.abs(rsng_sol - analytical_sol))

        # Plot analytical solution (blue solid)
        ax.plot(x_points, analytical_sol, 'b-', linewidth=2, label='Analytical', alpha=0.8)

        # Plot RSNG solution (red dashed)
        ax.plot(x_points, rsng_sol, 'r--', linewidth=2, label='RSNG', alpha=0.8)

        # Set title and error metrics
        ax.set_title(f'Solution at t = {t_actual:.2f}', fontsize=12)
        ax.text(0.02, 0.95, f'L2: {l2_error:.3e}\\nL∞: {linf_error:.3e}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                fontsize=10)

        # Set labels and grid
        ax.set_xlabel('x')
        ax.set_ylabel('u(x,t)')
        ax.grid(True, alpha=0.3)

        # Add legend only for first subplot
        if idx == 0:
            ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ RSNG solution comparison saved: {save_path}")
    return fig

def create_rsng_metrics_plot(training_history, save_path):
    """Create RSNG training metrics visualization"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('RSNG Training Metrics', fontsize=16, fontweight='bold')

    iterations = range(len(training_history['residual_norm']))

    # 1. PDE Residual Evolution
    ax1.semilogy(iterations, training_history['residual_norm'], 'b-', linewidth=2, alpha=0.8)
    ax1.set_title('RSNG PDE Residual Evolution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Residual Norm (log scale)')
    ax1.grid(True, alpha=0.3)

    # 2. Solution Error vs Analytical
    ax2.semilogy(iterations, training_history['solution_error'], 'r-', linewidth=2, alpha=0.8)
    ax2.set_title('RSNG Solution Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('L∞ Error (log scale)')
    ax2.grid(True, alpha=0.3)

    # 3. Parameter Update Magnitude
    ax3.semilogy(iterations, training_history['delta_theta_norm'], 'g-', linewidth=2, alpha=0.8)
    ax3.set_title('Parameter Update Magnitude', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('||Δθ|| (log scale)')
    ax3.grid(True, alpha=0.3)

    # 4. Sparse Update Statistics
    ax4.plot(iterations, training_history['n_updated'], 'm-', linewidth=2, alpha=0.8)
    ax4.set_title('Sparse Update Count', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Number of Updated Parameters')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ RSNG training metrics saved: {save_path}")
    return fig

def run_rsng_demo():
    """Run comprehensive RSNG demonstration"""
    print("🚀 Starting RSNG (Randomized Sparse Neural Galerkin) Demonstration")
    print("=" * 70)

    # Setup problem parameters
    x_min, x_max = 0.0, 1.0
    t_final = 0.5
    n_spatial = 50
    n_time_steps = 100
    dt = t_final / n_time_steps

    x_points = np.linspace(x_min, x_max, n_spatial)
    t_points = np.linspace(0.0, t_final, n_time_steps + 1)

    print(f"📊 Problem setup:")
    print(f"   Spatial domain: [{x_min}, {x_max}] with {n_spatial} points")
    print(f"   Time domain: [0, {t_final}] with {n_time_steps} steps (dt = {dt:.4f})")

    # Initialize RSNG components
    print(f"\\n🏗️  Initializing RSNG components...")

    # Create neural network (sequential-in-time: u(x; θ(t)))
    network = RSNGNeuralApproximator(spatial_dim=1, hidden_units=25, n_layers=4)

    # Create RSNG solver with sparsity
    sparsity_ratio = 0.2  # Update 20% of parameters at each step
    solver = RSNGSolver(network, sparsity_ratio=sparsity_ratio)

    # Create PDE right-hand side (heat equation)
    pde_rhs = PDERightHandSide(pde_type='heat')

    print(f"   Network: {len(network.theta)} parameters, {network.n_layers} layers")
    print(f"   RSNG sparsity: {sparsity_ratio:.1%} ({solver.n_sparse} params updated per step)")

    # Fit initial condition: u(x, 0) = sin(πx)
    print(f"\\n🎯 Fitting initial condition u(x,0) = sin(πx)...")
    u0 = np.sin(np.pi * x_points)
    network.fit_initial_condition(x_points, u0, max_iterations=1000, tolerance=1e-6)

    # Verify initial condition fit
    u0_fitted = network.forward(x_points)
    ic_error = np.max(np.abs(u0_fitted - u0))
    print(f"   ✅ Initial condition error: {ic_error:.2e}")

    # Initialize tracking arrays
    rsng_solutions = [u0_fitted.copy()]
    analytical_solutions = [u0.copy()]

    training_history = {
        'residual_norm': [],
        'solution_error': [],
        'delta_theta_norm': [],
        'n_updated': [],
        'step_time_ms': []
    }

    # RSNG time-stepping loop (Algorithm 1 from paper)
    print(f"\\n🔄 RSNG time-stepping integration...")
    print(f"   Following Algorithm 1: Sequential parameter evolution with sparse updates")

    for k in range(1, n_time_steps + 1):
        t_current = t_points[k]

        # Perform RSNG time step
        step_info = solver.time_step(x_points, pde_rhs, dt)

        # Get updated solution
        u_rsng = network.forward(x_points)
        u_analytical = analytical_solution(x_points, t_current)

        # Compute solution error
        solution_error = np.max(np.abs(u_rsng - u_analytical))

        # Store solutions and metrics
        rsng_solutions.append(u_rsng.copy())
        analytical_solutions.append(u_analytical.copy())

        training_history['residual_norm'].append(step_info['residual_norm'])
        training_history['solution_error'].append(solution_error)
        training_history['delta_theta_norm'].append(step_info['delta_theta_norm'])
        training_history['n_updated'].append(step_info['n_updated'])
        training_history['step_time_ms'].append(step_info['step_time_ms'])

        # Progress report
        if k % 20 == 0 or k == n_time_steps:
            avg_time = np.mean(training_history['step_time_ms'][-20:])
            print(f"   Step {k:3d}/{n_time_steps}: t={t_current:.3f}, "
                  f"Error={solution_error:.2e}, Residual={step_info['residual_norm']:.2e}, "
                  f"Updated={step_info['n_updated']}, Time={avg_time:.1f}ms")

    print(f"\\n✅ RSNG integration completed!")

    # Generate result plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "rsng_results"
    os.makedirs(results_dir, exist_ok=True)

    print(f"\\n🎨 Generating RSNG result visualizations...")

    # 1. Solution comparison plot
    comparison_path = f"{results_dir}/rsng_solution_comparison_{timestamp}.png"
    create_rsng_solution_comparison(x_points, t_points[:-1], rsng_solutions[:-1],
                                    analytical_solutions[:-1], comparison_path)

    # 2. Training metrics plot
    metrics_path = f"{results_dir}/rsng_training_metrics_{timestamp}.png"
    create_rsng_metrics_plot(training_history, metrics_path)

    # Performance summary
    print(f"\\n📈 RSNG Performance Summary:")
    print(f"   Final solution error: {training_history['solution_error'][-1]:.2e}")
    print(f"   Average residual: {np.mean(training_history['residual_norm']):.2e}")
    print(f"   Average parameters updated: {np.mean(training_history['n_updated']):.1f} / {len(network.theta)}")
    print(f"   Average step time: {np.mean(training_history['step_time_ms']):.1f}ms")
    print(f"   Sparsity achieved: {np.mean(training_history['n_updated']) / len(network.theta):.1%}")

    print(f"\\n🎯 Key RSNG Features Demonstrated:")
    print(f"   ✅ Sequential-in-time parameter evolution θ(t)")
    print(f"   ✅ Sparse random parameter updates ({sparsity_ratio:.1%} per step)")
    print(f"   ✅ Dirac-Frenkel variational principle implementation")
    print(f"   ✅ PDE residual minimization via Galerkin projection")
    print(f"   ✅ Computational efficiency through sparsity")

    print(f"\\n📊 Generated Results:")
    print(f"   • Solution comparison: {comparison_path}")
    print(f"   • Training metrics: {metrics_path}")

    print(f"\\n🎉 RSNG Demonstration completed successfully!")

    return training_history, rsng_solutions, analytical_solutions

if __name__ == "__main__":
    run_rsng_demo()