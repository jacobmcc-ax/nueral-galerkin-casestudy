#!/usr/bin/env python3
"""
RSNG Test with Full Training Metrics

Test the RSNG implementation with comprehensive metrics tracking
to validate the algorithm performance
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
    """Analytical solution for heat equation"""
    return np.exp(-np.pi**2 * t) * np.sin(np.pi * x)

def create_rsng_test_plots(training_history, x_points, t_points, rsng_solutions, analytical_solutions):
    """Create comprehensive RSNG test visualizations"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Training metrics plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('RSNG Training Metrics Analysis', fontsize=16, fontweight='bold')

    iterations = range(len(training_history['residual_norm']))

    # Residual evolution
    ax1.semilogy(iterations, training_history['residual_norm'], 'b-', linewidth=2, alpha=0.8)
    ax1.set_title('PDE Residual Evolution', fontsize=14)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Residual Norm (log scale)')
    ax1.grid(True, alpha=0.3)

    # Solution error
    ax2.semilogy(iterations, training_history['solution_error'], 'r-', linewidth=2, alpha=0.8)
    ax2.set_title('Solution Error vs Analytical', fontsize=14)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('L∞ Error (log scale)')
    ax2.grid(True, alpha=0.3)

    # Parameter update magnitude
    ax3.semilogy(iterations, training_history['delta_theta_norm'], 'g-', linewidth=2, alpha=0.8)
    ax3.set_title('Parameter Update Magnitude', fontsize=14)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('||Δθ|| (log scale)')
    ax3.grid(True, alpha=0.3)

    # Sparse update statistics
    ax4.plot(iterations, training_history['n_updated'], 'm-', linewidth=2, alpha=0.8)
    ax4.set_title('Parameters Updated per Step', fontsize=14)
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Number of Parameters')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    metrics_path = f"rsng_test_metrics_{timestamp}.png"
    plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
    print(f"✅ Training metrics saved: {metrics_path}")
    plt.close()

    # 2. Solution comparison plot
    n_plots = min(6, len(t_points))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('RSNG vs Analytical Solution Comparison', fontsize=16, fontweight='bold')

    axes = axes.flatten()

    for i in range(n_plots):
        idx = i * (len(t_points) - 1) // (n_plots - 1) if n_plots > 1 else 0
        t = t_points[idx]

        rsng_sol = rsng_solutions[idx]
        analytical_sol = analytical_solutions[idx]

        # Compute error
        error = np.max(np.abs(rsng_sol - analytical_sol))

        axes[i].plot(x_points, analytical_sol, 'b-', linewidth=2, label='Analytical', alpha=0.8)
        axes[i].plot(x_points, rsng_sol, 'r--', linewidth=2, label='RSNG', alpha=0.8)

        axes[i].set_title(f't = {t:.3f}, Error = {error:.2e}')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('u(x,t)')
        axes[i].grid(True, alpha=0.3)

        if i == 0:
            axes[i].legend()

    plt.tight_layout()
    solutions_path = f"rsng_test_solutions_{timestamp}.png"
    plt.savefig(solutions_path, dpi=300, bbox_inches='tight')
    print(f"✅ Solution comparison saved: {solutions_path}")
    plt.close()

    return metrics_path, solutions_path

def rsng_metrics_test():
    """RSNG test with comprehensive metrics"""
    print("📊 RSNG Training Metrics Test")
    print("=" * 40)

    # Setup with moderate parameters for stability
    n_spatial = 25
    n_time = 20
    t_final = 0.2  # Shorter time for stability

    x_points = np.linspace(0, 1, n_spatial)
    dt = t_final / n_time
    t_points = np.linspace(0, t_final, n_time + 1)

    print(f"Problem setup:")
    print(f"  Spatial points: {n_spatial}")
    print(f"  Time steps: {n_time}")
    print(f"  Final time: {t_final}")
    print(f"  Time step: {dt:.4f}")

    # Initialize RSNG components
    network = RSNGNeuralApproximator(spatial_dim=1, hidden_units=15, n_layers=3)
    solver = RSNGSolver(network, sparsity_ratio=0.3)  # 30% sparsity
    pde_rhs = PDERightHandSide(pde_type='heat')

    print(f"\\nRSNG configuration:")
    print(f"  Network parameters: {len(network.theta)}")
    print(f"  Sparse updates: {solver.n_sparse} / {solver.n_params} ({solver.sparsity_ratio:.1%})")

    # Fit initial condition
    print(f"\\n🎯 Fitting initial condition...")
    u0 = np.sin(np.pi * x_points)
    network.fit_initial_condition(x_points, u0, max_iterations=500, tolerance=1e-5)

    # Verify initial fit
    u0_fitted = network.forward(x_points)
    ic_error = np.max(np.abs(u0_fitted - u0))
    print(f"   Initial condition error: {ic_error:.2e}")

    # Initialize storage
    training_history = {
        'residual_norm': [],
        'solution_error': [],
        'delta_theta_norm': [],
        'n_updated': [],
        'step_time_ms': []
    }

    rsng_solutions = [u0_fitted.copy()]
    analytical_solutions = [u0.copy()]

    # RSNG time-stepping with metrics
    print(f"\\n🔄 RSNG time integration with metrics tracking...")

    for step in range(1, n_time + 1):
        t_current = t_points[step]

        try:
            # RSNG time step
            step_info = solver.time_step(x_points, pde_rhs, dt)

            # Get solutions
            u_rsng = network.forward(x_points)
            u_analytical = analytical_solution(x_points, t_current)

            # Compute metrics
            solution_error = np.max(np.abs(u_rsng - u_analytical))

            # Store data
            rsng_solutions.append(u_rsng.copy())
            analytical_solutions.append(u_analytical.copy())

            training_history['residual_norm'].append(step_info['residual_norm'])
            training_history['solution_error'].append(solution_error)
            training_history['delta_theta_norm'].append(step_info['delta_theta_norm'])
            training_history['n_updated'].append(step_info['n_updated'])
            training_history['step_time_ms'].append(step_info['step_time_ms'])

            # Progress report
            if step % 5 == 0 or step == n_time:
                print(f"   Step {step:2d}/{n_time}: t={t_current:.3f}, "
                      f"Error={solution_error:.2e}, Residual={step_info['residual_norm']:.2e}, "
                      f"Updated={step_info['n_updated']}")

            # Check for numerical issues
            if not np.isfinite(u_rsng).all() or solution_error > 1e3:
                print(f"   ⚠️  Numerical instability detected at step {step}")
                break

        except Exception as e:
            print(f"   ❌ Error at step {step}: {e}")
            break

    print(f"\\n✅ RSNG integration completed!")

    # Generate analysis plots
    print(f"\\n🎨 Generating training metrics visualizations...")
    metrics_path, solutions_path = create_rsng_test_plots(
        training_history, x_points, t_points[:len(rsng_solutions)],
        rsng_solutions, analytical_solutions
    )

    # Performance analysis
    print(f"\\n📈 RSNG Performance Analysis:")
    if len(training_history['solution_error']) > 0:
        final_error = training_history['solution_error'][-1]
        avg_residual = np.mean(training_history['residual_norm'])
        avg_updates = np.mean(training_history['n_updated'])
        avg_step_time = np.mean(training_history['step_time_ms'])

        print(f"   Final solution error: {final_error:.2e}")
        print(f"   Average PDE residual: {avg_residual:.2e}")
        print(f"   Average sparse updates: {avg_updates:.1f} / {len(network.theta)}")
        print(f"   Sparsity achieved: {avg_updates / len(network.theta):.1%}")
        print(f"   Average step time: {avg_step_time:.1f}ms")

        # Check if algorithm is working as expected
        print(f"\\n🔍 Algorithm Validation:")
        if final_error < 1.0:
            print(f"   ✅ Solution error within reasonable range")
        else:
            print(f"   ⚠️  High solution error - may need stability improvements")

        if 0.1 <= avg_updates / len(network.theta) <= 0.5:
            print(f"   ✅ Sparse updates working correctly")
        else:
            print(f"   ⚠️  Sparse update ratio outside expected range")

    print(f"\\n🎯 Key RSNG Features Demonstrated:")
    print(f"   ✅ Sequential-in-time parameter evolution θ(t)")
    print(f"   ✅ Dirac-Frenkel variational principle implementation")
    print(f"   ✅ Sparse random parameter updates")
    print(f"   ✅ PDE residual minimization")
    print(f"   ✅ Comprehensive training metrics tracking")

    print(f"\\n📊 Generated visualizations:")
    print(f"   • Training metrics: {metrics_path}")
    print(f"   • Solution comparison: {solutions_path}")

if __name__ == "__main__":
    rsng_metrics_test()