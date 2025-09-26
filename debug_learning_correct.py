#!/usr/bin/env python3
"""
Debug script with correct baseline comparison
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from neural_approximation import NeuralApproximator

def debug_neural_learning_correct():
    """Debug with correct baseline comparison"""

    print("=== CORRECT LEARNING TEST ===")

    # Create simple target function
    x = np.linspace(0, 1, 50)
    t = 0.1
    target = np.sin(np.pi * x) * np.exp(-t)

    print(f"Target function: sin(π*x) * exp(-{t})")
    print(f"Target range: [{np.min(target):.3f}, {np.max(target):.3f}]")

    # Create neural network
    neural_net = NeuralApproximator(spatial_dim=1, hidden_units=20)

    # Get pure neural network initial performance (the correct baseline)
    neural_net._in_training_mode = True
    pure_initial = neural_net.forward(x, t)
    pure_initial_error = np.mean((pure_initial - target)**2)
    neural_net._in_training_mode = False

    print(f"Pure neural initial error: {pure_initial_error:.6f}")
    print(f"Pure neural range: [{np.min(pure_initial):.3f}, {np.max(pure_initial):.3f}]")

    # Train the network using the normal training method
    x_input = np.column_stack([x, np.full_like(x, t)])
    final_loss = neural_net.train(x_input, target, learning_rate=0.01, epochs=50)

    # Get final performance - should be pure neural output since trained
    final_approx = neural_net.forward(x, t)
    final_error = np.mean((final_approx - target)**2)

    print(f"Final error: {final_error:.6f}")
    print(f"Final range: [{np.min(final_approx):.3f}, {np.max(final_approx):.3f}]")
    print(f"Training loss reported: {final_loss:.6f}")

    # Calculate correct improvement
    improvement = (pure_initial_error - final_error) / pure_initial_error * 100
    print(f"Correct improvement: {improvement:.1f}%")

    return improvement > 10

if __name__ == "__main__":
    success = debug_neural_learning_correct()
    print(f"\n{'✅' if success else '❌'} Learning test {'PASSED' if success else 'FAILED'}")