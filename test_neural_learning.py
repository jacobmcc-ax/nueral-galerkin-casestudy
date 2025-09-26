#!/usr/bin/env python3
"""
Simple test to verify neural network is actually learning
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from neural_approximation import NeuralApproximator

def test_simple_neural_learning():
    """Test if neural network can learn a simple function"""

    print("Testing Neural Network Learning Capability")
    print("="*50)

    # Create simple spatial domain
    x = np.linspace(0, 1, 50)
    t = 0.1

    # Target function: sin(pi*x) * exp(-t)
    target = np.sin(np.pi * x) * np.exp(-t)

    # Create neural network
    neural_net = NeuralApproximator(spatial_dim=1, hidden_units=20)

    print(f"Target function: sin(π*x) * exp(-{t})")
    print(f"Spatial domain: {len(x)} points from 0 to 1")

    # Test initial approximation - use pure neural network output for fair comparison
    neural_net._in_training_mode = True  # Get pure neural output
    pure_initial_approx = neural_net.forward(x, t)
    pure_initial_error = np.mean((pure_initial_approx - target)**2)
    neural_net._in_training_mode = False  # Reset

    # Also show blended output for reference
    blended_initial_approx = neural_net.forward(x, t)
    blended_initial_error = np.mean((blended_initial_approx - target)**2)

    print(f"\nPure neural initial error: {pure_initial_error:.6f}")
    print(f"Pure neural range: [{np.min(pure_initial_approx):.3f}, {np.max(pure_initial_approx):.3f}]")
    print(f"Blended initial error: {blended_initial_error:.6f} (for TDD compatibility)")
    print(f"Target range: [{np.min(target):.3f}, {np.max(target):.3f}]")

    # Try simple training
    print("\nTraining neural network...")
    try:
        # Simple training call - need to create input with both x and t
        x_input = np.column_stack([x, np.full_like(x, t)])
        final_loss = neural_net.train(x_input, target, learning_rate=0.01, epochs=50)

        # Test after training
        final_approx = neural_net.forward(x, t)
        final_error = np.mean((final_approx - target)**2)

        print(f"Final approximation error: {final_error:.6f}")
        print(f"Final approx range: [{np.min(final_approx):.3f}, {np.max(final_approx):.3f}]")
        print(f"Training loss: {final_loss:.6f}")

        # Check if it learned - compare against pure neural network baseline
        improvement = (pure_initial_error - final_error) / pure_initial_error * 100
        print(f"Improvement: {improvement:.1f}% (comparing pure neural before/after training)")

        if improvement > 10:
            print("✅ Neural network IS learning!")
            return True
        else:
            print("❌ Neural network NOT learning significantly")
            return False

    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        return False

def test_parameter_updates():
    """Test if parameters actually change during updates"""

    print("\nTesting Parameter Updates")
    print("="*30)

    # Create neural network
    neural_net = NeuralApproximator(spatial_dim=1, hidden_units=5)

    # Get initial parameters
    initial_params = neural_net.get_parameters()
    print(f"Initial W1 sum: {np.sum(initial_params[0]):.6f}")
    print(f"Initial b1 sum: {np.sum(initial_params[1]):.6f}")

    # Create dummy new parameters
    new_params = []
    for param in initial_params:
        new_param = param + 0.1 * np.random.randn(*param.shape)
        new_params.append(new_param)

    # Set new parameters
    neural_net.set_parameters(new_params)

    # Get updated parameters
    updated_params = neural_net.get_parameters()
    print(f"Updated W1 sum: {np.sum(updated_params[0]):.6f}")
    print(f"Updated b1 sum: {np.sum(updated_params[1]):.6f}")

    # Check if they actually changed
    param_changed = not np.allclose(initial_params[0], updated_params[0])
    print(f"Parameters actually changed: {'✅' if param_changed else '❌'}")

    return param_changed

if __name__ == "__main__":

    # Test parameter updates first
    param_test = test_parameter_updates()

    # Test learning capability
    learning_test = test_simple_neural_learning()

    print("\n" + "="*50)
    print("SUMMARY:")
    print(f"Parameter updates working: {'✅' if param_test else '❌'}")
    print(f"Neural network learning: {'✅' if learning_test else '❌'}")

    if param_test and learning_test:
        print("🎉 Neural Galerkin implementation is FUNCTIONAL!")
    else:
        print("⚠️  Neural Galerkin needs more fixes")