#!/usr/bin/env python3
"""
Simple RSNG Test - Minimal implementation to verify algorithm works
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from rsng_neural_approximator import RSNGNeuralApproximator, RSNGSolver, PDERightHandSide

def analytical_solution(x, t):
    """Analytical solution for heat equation"""
    return np.exp(-np.pi**2 * t) * np.sin(np.pi * x)

def simple_rsng_test():
    """Simple RSNG test with minimal parameters"""
    print("🧪 Simple RSNG Test")
    print("=" * 30)

    # Minimal setup
    n_spatial = 20  # Reduce spatial points
    n_time = 5      # Just a few time steps
    t_final = 0.1   # Short time

    x_points = np.linspace(0, 1, n_spatial)
    dt = t_final / n_time

    # Small network
    network = RSNGNeuralApproximator(spatial_dim=1, hidden_units=10, n_layers=3)
    solver = RSNGSolver(network, sparsity_ratio=0.5)
    pde_rhs = PDERightHandSide(pde_type='heat')

    print(f"Network: {len(network.theta)} parameters")
    print(f"Spatial points: {n_spatial}")

    # Initial condition
    u0 = np.sin(np.pi * x_points)
    network.fit_initial_condition(x_points, u0, max_iterations=100, tolerance=1e-4)

    print("\\nTime stepping...")

    for step in range(n_time):
        t_current = (step + 1) * dt

        # RSNG step
        step_info = solver.time_step(x_points, pde_rhs, dt)

        # Check solution
        u_rsng = network.forward(x_points)
        u_analytical = analytical_solution(x_points, t_current)
        error = np.max(np.abs(u_rsng - u_analytical))

        print(f"  Step {step+1}: t={t_current:.3f}, error={error:.2e}, residual={step_info['residual_norm']:.2e}")

        # Check for numerical issues
        if not np.isfinite(u_rsng).all():
            print("  ⚠️  Non-finite values detected")
            break

    print("\\n✅ Simple RSNG test completed!")

if __name__ == "__main__":
    simple_rsng_test()