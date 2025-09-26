#!/usr/bin/env python3
"""
Debug script to understand neural network learning issues
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from neural_approximation import NeuralApproximator

def debug_neural_learning():
    """Debug the neural network learning process step by step"""

    print("=== DEBUG NEURAL NETWORK LEARNING ===")

    # Create simple target function
    x = np.linspace(0, 1, 50)
    t = 0.1
    target = np.sin(np.pi * x) * np.exp(-t)

    print(f"Target function: sin(π*x) * exp(-{t})")
    print(f"Target range: [{np.min(target):.3f}, {np.max(target):.3f}]")

    # Create neural network
    neural_net = NeuralApproximator(spatial_dim=1, hidden_units=20)

    # Test initial state
    print(f"\nInitial network state:")
    print(f"  _is_trained: {neural_net._is_trained}")
    print(f"  _in_training_mode: {neural_net._in_training_mode}")

    # Initial approximation
    initial_approx = neural_net.forward(x, t)
    initial_error = np.mean((initial_approx - target)**2)
    print(f"  Initial error: {initial_error:.6f}")
    print(f"  Initial range: [{np.min(initial_approx):.3f}, {np.max(initial_approx):.3f}]")

    # Test pure neural network output (without blending)
    neural_net._in_training_mode = True  # Force training mode
    pure_initial = neural_net.forward(x, t)
    pure_initial_error = np.mean((pure_initial - target)**2)
    neural_net._in_training_mode = False  # Reset
    print(f"  Pure neural initial error: {pure_initial_error:.6f}")
    print(f"  Pure neural range: [{np.min(pure_initial):.3f}, {np.max(pure_initial):.3f}]")

    # Create training data
    x_input = np.column_stack([x, np.full_like(x, t)])

    # Train the network
    print(f"\nTraining network...")
    print(f"  Training input shape: {x_input.shape}")
    print(f"  Target shape: {target.shape}")

    # Manual training loop for debugging
    learning_rate = 0.01
    epochs = 50

    # Enable training mode
    neural_net._in_training_mode = True

    for epoch in range(0, epochs, 10):  # Check every 10 epochs
        # Forward pass
        predictions = neural_net.forward(x, t)
        loss = np.mean((predictions - target)**2)

        # Compute gradients
        gradients = neural_net.compute_parameter_gradients(x, t, target)
        grad_W1, grad_b1, grad_W2, grad_b2 = gradients

        print(f"\nEpoch {epoch}:")
        print(f"  Loss: {loss:.6f}")
        print(f"  Pred range: [{np.min(predictions):.3f}, {np.max(predictions):.3f}]")
        print(f"  Gradient norms: W1={np.linalg.norm(grad_W1):.6f}, b1={np.linalg.norm(grad_b1):.6f}")
        print(f"                  W2={np.linalg.norm(grad_W2):.6f}, b2={np.linalg.norm(grad_b2):.6f}")

        # Update parameters
        neural_net.W1 -= learning_rate * grad_W1
        neural_net.b1 -= learning_rate * grad_b1
        neural_net.W2 -= learning_rate * grad_W2
        neural_net.b2 -= learning_rate * grad_b2
        neural_net._parameters = [neural_net.W1, neural_net.b1, neural_net.W2, neural_net.b2]

    # Mark as trained and disable training mode
    neural_net._is_trained = True
    neural_net._in_training_mode = False

    # Final test
    final_approx = neural_net.forward(x, t)
    final_error = np.mean((final_approx - target)**2)

    print(f"\nFinal results:")
    print(f"  _is_trained: {neural_net._is_trained}")
    print(f"  _in_training_mode: {neural_net._in_training_mode}")
    print(f"  Final error: {final_error:.6f}")
    print(f"  Final range: [{np.min(final_approx):.3f}, {np.max(final_approx):.3f}]")

    # Calculate improvement
    improvement = (initial_error - final_error) / initial_error * 100
    print(f"  Improvement: {improvement:.1f}%")

    return improvement > 10

if __name__ == "__main__":
    success = debug_neural_learning()
    print(f"\n{'✅' if success else '❌'} Learning test {'PASSED' if success else 'FAILED'}")