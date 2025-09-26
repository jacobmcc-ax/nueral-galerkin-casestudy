#!/usr/bin/env python3
"""
Randomized Sparse Neural Galerkin (RSNG) Implementation
Based on the paper: "Randomized Sparse Neural Galerkin Schemes for Solving Evolution Equations with Deep Networks"

Key differences from standard PINN approaches:
1. Sequential-in-time training: Network represents u(x; θ(t)) at single time point
2. Parameter evolution: θ(t) evolves according to PDE dynamics via Dirac-Frenkel principle
3. Sparse updates: Only random subset of parameters updated at each time step
"""

import numpy as np
import time
from datetime import datetime

class RSNGNeuralApproximator:
    """
    RSNG Neural Network for Sequential-in-Time PDE Solving

    Network represents u(x; θ(t)) where:
    - x: spatial coordinates
    - θ(t): time-evolving parameters
    - Network output approximates PDE solution at time t
    """

    def __init__(self, spatial_dim=1, hidden_units=25, n_layers=3):
        """
        Initialize RSNG neural network

        Args:
            spatial_dim: Spatial dimension (1D for heat equation)
            hidden_units: Width of hidden layers
            n_layers: Total number of layers (including input/output)
        """
        self.spatial_dim = spatial_dim
        self.hidden_units = hidden_units
        self.n_layers = n_layers

        # Network architecture: Input(spatial) -> Hidden -> ... -> Output(1)
        self.layers = []

        # Input layer
        input_size = spatial_dim
        self.layers.append({
            'W': np.random.randn(input_size, hidden_units) * np.sqrt(2.0 / input_size),
            'b': np.zeros(hidden_units)
        })

        # Hidden layers
        for i in range(n_layers - 2):
            self.layers.append({
                'W': np.random.randn(hidden_units, hidden_units) * np.sqrt(2.0 / hidden_units),
                'b': np.zeros(hidden_units)
            })

        # Output layer
        self.layers.append({
            'W': np.random.randn(hidden_units, 1) * np.sqrt(2.0 / hidden_units),
            'b': np.zeros(1)
        })

        # Flatten parameters for RSNG algorithm
        self.theta = self._flatten_parameters()
        self.param_shapes = self._get_parameter_shapes()

        print(f"✅ RSNG Network initialized: {len(self.theta)} parameters, {n_layers} layers")

    def _flatten_parameters(self):
        """Flatten all network parameters into single vector θ"""
        params = []
        for layer in self.layers:
            params.extend(layer['W'].flatten())
            params.extend(layer['b'].flatten())
        return np.array(params)

    def _get_parameter_shapes(self):
        """Store parameter shapes for reconstruction"""
        shapes = []
        for layer in self.layers:
            shapes.append(('W', layer['W'].shape))
            shapes.append(('b', layer['b'].shape))
        return shapes

    def _unflatten_parameters(self, theta):
        """Reconstruct network layers from flattened parameter vector"""
        layers = []
        idx = 0

        for i in range(len(self.layers)):
            layer = {}
            # Weight matrix
            w_shape = self.param_shapes[2*i][1]
            w_size = np.prod(w_shape)
            layer['W'] = theta[idx:idx+w_size].reshape(w_shape)
            idx += w_size

            # Bias vector
            b_shape = self.param_shapes[2*i+1][1]
            b_size = np.prod(b_shape)
            layer['b'] = theta[idx:idx+b_size].reshape(b_shape)
            idx += b_size

            layers.append(layer)

        return layers

    def forward(self, x, theta=None):
        """
        Forward pass: Compute u(x; θ)

        Args:
            x: Spatial coordinates [n_points, spatial_dim]
            theta: Parameter vector (uses self.theta if None)

        Returns:
            u: Network output [n_points]
        """
        if theta is None:
            theta = self.theta

        layers = self._unflatten_parameters(theta)

        # Ensure x is 2D
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        activation = x

        # Forward through all layers except last
        for i in range(len(layers) - 1):
            z = activation @ layers[i]['W'] + layers[i]['b']
            # Use tanh activation, clipped to avoid saturation
            z = np.clip(z, -3.0, 3.0)
            activation = np.tanh(z)

        # Output layer (no activation)
        output = activation @ layers[-1]['W'] + layers[-1]['b']

        return output.flatten()

    def compute_jacobian(self, x, theta=None):
        """
        Compute Jacobian ∇_θ u(x; θ) using analytical backpropagation

        Much faster than finite differences - O(1) vs O(p) forward passes

        Args:
            x: Spatial points [n_points, spatial_dim]
            theta: Parameter vector

        Returns:
            J: Jacobian matrix [n_points, n_parameters]
        """
        if theta is None:
            theta = self.theta

        # Ensure x is 2D
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        n_points, _ = x.shape
        n_params = len(theta)

        # Reconstruct network layers
        layers = self._unflatten_parameters(theta)

        # Forward pass with intermediate storage
        activations = [x]  # Store all layer activations
        z_values = []      # Store all pre-activation values

        current_activation = x

        # Forward pass through hidden layers
        for i in range(len(layers) - 1):
            z = current_activation @ layers[i]['W'] + layers[i]['b']
            z = np.clip(z, -3.0, 3.0)  # Prevent saturation
            z_values.append(z)

            current_activation = np.tanh(z)
            activations.append(current_activation)

        # Output layer
        z_out = current_activation @ layers[-1]['W'] + layers[-1]['b']
        z_values.append(z_out)

        # Initialize Jacobian
        jacobian = np.zeros((n_points, n_params))

        # Backpropagate to compute gradients
        # Start with gradient of output w.r.t. output (identity)
        grad_output = np.ones((n_points, 1))  # ∂u/∂u = 1

        param_idx = n_params  # Work backwards through parameters

        # Output layer gradients
        layer_idx = len(layers) - 1
        W_shape = layers[layer_idx]['W'].shape
        b_shape = layers[layer_idx]['b'].shape

        # ∂u/∂W_out and ∂u/∂b_out
        # grad_W should be outer product for each point
        grad_W = np.zeros((n_points, W_shape[0] * W_shape[1]))
        for i in range(n_points):
            grad_W[i] = np.outer(activations[layer_idx][i], grad_output[i]).flatten()

        grad_b = grad_output  # [n_points, b_size]

        # Store in jacobian
        w_size = np.prod(W_shape)
        b_size = np.prod(b_shape)

        param_idx -= b_size
        jacobian[:, param_idx:param_idx+b_size] = grad_b.reshape(n_points, -1)

        param_idx -= w_size
        jacobian[:, param_idx:param_idx+w_size] = grad_W.reshape(n_points, -1)

        # Propagate gradient backwards
        grad_activation = grad_output @ layers[layer_idx]['W'].T

        # Hidden layer gradients
        for layer_idx in reversed(range(len(layers) - 1)):
            # Gradient through tanh activation
            tanh_grad = 1 - np.tanh(z_values[layer_idx])**2
            grad_z = grad_activation * tanh_grad

            W_shape = layers[layer_idx]['W'].shape
            b_shape = layers[layer_idx]['b'].shape

            # Gradients w.r.t. weights and biases
            grad_W = np.zeros((n_points, W_shape[0] * W_shape[1]))
            for i in range(n_points):
                grad_W[i] = np.outer(activations[layer_idx][i], grad_z[i]).flatten()

            grad_b = grad_z  # [n_points, b_size]

            # Store in jacobian
            w_size = np.prod(W_shape)
            b_size = np.prod(b_shape)

            param_idx -= b_size
            jacobian[:, param_idx:param_idx+b_size] = grad_b.reshape(n_points, -1)

            param_idx -= w_size
            jacobian[:, param_idx:param_idx+w_size] = grad_W.reshape(n_points, -1)

            # Propagate to previous layer
            if layer_idx > 0:
                grad_activation = grad_z @ layers[layer_idx]['W'].T

        return jacobian

    def fit_initial_condition(self, x_points, u0, max_iterations=1000, tolerance=1e-6):
        """
        Fit network to initial condition u(x, 0) = u0(x)

        This initializes θ(0) for the RSNG time-stepping algorithm

        Args:
            x_points: Spatial grid points
            u0: Initial condition values
            max_iterations: Maximum fitting iterations
            tolerance: Convergence tolerance
        """
        print("🎯 Fitting initial condition...")

        learning_rate = 0.01

        for iteration in range(max_iterations):
            # Forward pass
            u_pred = self.forward(x_points, self.theta)

            # Loss: MSE between prediction and initial condition
            loss = np.mean((u_pred - u0)**2)

            if loss < tolerance:
                print(f"   ✅ Initial condition fitted: loss={loss:.2e} at iteration {iteration}")
                break

            # Compute gradients using finite differences
            grad_theta = np.zeros_like(self.theta)
            eps = 1e-6

            for i in range(len(self.theta)):
                theta_plus = self.theta.copy()
                theta_plus[i] += eps

                u_plus = self.forward(x_points, theta_plus)
                loss_plus = np.mean((u_plus - u0)**2)

                grad_theta[i] = (loss_plus - loss) / eps

            # Gradient descent update
            self.theta -= learning_rate * grad_theta

            # Progress report
            if iteration % 200 == 0:
                print(f"   Iteration {iteration}: loss={loss:.2e}")

        print(f"✅ Initial condition fitting completed: final loss={loss:.2e}")

class RSNGSolver:
    """
    RSNG Time-Stepping Solver

    Implements Algorithm 1 from the paper:
    1. Draw realization of sketching matrix S_k
    2. Solve sparse least-squares problem for Δθ_s
    3. Lift sparse update: Δθ = S_k * Δθ_s
    4. Update parameters: θ^(k) = θ^(k-1) + δt * Δθ
    """

    def __init__(self, network, sparsity_ratio=0.1):
        """
        Initialize RSNG solver

        Args:
            network: RSNGNeuralApproximator instance
            sparsity_ratio: Fraction of parameters to update (s/p in paper)
        """
        self.network = network
        self.sparsity_ratio = sparsity_ratio
        self.n_params = len(network.theta)
        self.n_sparse = max(1, int(sparsity_ratio * self.n_params))

        print(f"📊 RSNG Solver: {self.n_params} total params, {self.n_sparse} sparse updates ({sparsity_ratio:.1%})")

    def create_sketching_matrix(self):
        """
        Create sparse sketching matrix S_t for random parameter selection

        Returns:
            indices: Selected parameter indices
            S_t: Sketching matrix [n_params, n_sparse]
        """
        # Uniform random sampling of parameter indices
        indices = np.random.choice(self.n_params, size=self.n_sparse, replace=False)

        # Create sketching matrix S_t
        S_t = np.zeros((self.n_params, self.n_sparse))
        for i, idx in enumerate(indices):
            S_t[idx, i] = 1.0

        return indices, S_t

    def solve_sparse_update(self, x_points, pde_rhs, dt):
        """
        Solve RSNG sparse least-squares problem (Equation 9 from paper)

        min_{Δθ_s} ||J_s(θ) * Δθ_s - f(θ)||²

        Where:
        - J_s = J * S_t (sketched Jacobian)
        - f = PDE right-hand side evaluated at current θ

        Args:
            x_points: Spatial grid points
            pde_rhs: PDERightHandSide instance
            dt: Time step size

        Returns:
            delta_theta: Full parameter update vector
            residual_norm: Residual norm for monitoring
        """
        # Get current network state
        u_current = self.network.forward(x_points)

        # Compute full Jacobian ∇_θ u(x; θ)
        J_full = self.network.compute_jacobian(x_points)

        # Evaluate PDE right-hand side f(x, u(x; θ))
        f_values = pde_rhs(x_points, u_current)

        # Create sparse sketching matrix
        sparse_indices, S_t = self.create_sketching_matrix()

        # Form sketched Jacobian J_s = J * S_t
        J_sparse = J_full @ S_t  # [n_points, n_sparse]

        # Solve sparse least-squares problem
        # min ||J_s * Δθ_s - f||²
        try:
            delta_theta_sparse, residuals, rank, s_values = np.linalg.lstsq(J_sparse, f_values, rcond=1e-6)
            residual_norm = np.linalg.norm(residuals) if len(residuals) > 0 else np.linalg.norm(J_sparse @ delta_theta_sparse - f_values)
        except np.linalg.LinAlgError:
            print("⚠️  Singular matrix in sparse solve, using pseudoinverse")
            delta_theta_sparse = np.linalg.pinv(J_sparse) @ f_values
            residual_norm = np.linalg.norm(J_sparse @ delta_theta_sparse - f_values)

        # Lift sparse update: Δθ = S_t * Δθ_s
        delta_theta = S_t @ delta_theta_sparse

        return delta_theta, residual_norm, sparse_indices

    def time_step(self, x_points, pde_rhs, dt):
        """
        Perform single RSNG time step

        Args:
            x_points: Spatial grid
            pde_rhs: PDE right-hand side function
            dt: Time step size

        Returns:
            dict: Step information (residual, updated parameters, etc.)
        """
        start_time = time.time()

        # Solve for sparse parameter update
        delta_theta, residual_norm, sparse_indices = self.solve_sparse_update(x_points, pde_rhs, dt)

        # Update network parameters: θ^(k) = θ^(k-1) + δt * Δθ
        self.network.theta += dt * delta_theta

        step_time = (time.time() - start_time) * 1000  # ms

        return {
            'residual_norm': residual_norm,
            'sparse_indices': sparse_indices,
            'n_updated': len(sparse_indices),
            'step_time_ms': step_time,
            'delta_theta_norm': np.linalg.norm(delta_theta)
        }

class PDERightHandSide:
    """
    PDE Right-Hand Side Computation for RSNG

    Computes f(x, u) where ∂u/∂t = f(x, u) defines the PDE
    """

    def __init__(self, pde_type='heat'):
        self.pde_type = pde_type

    def compute_second_derivative(self, x_points, u_values):
        """
        Compute ∂²u/∂x² using finite differences

        Args:
            x_points: Spatial grid [n_points]
            u_values: Solution values [n_points]

        Returns:
            d2u_dx2: Second derivative [n_points]
        """
        n = len(x_points)
        d2u_dx2 = np.zeros_like(u_values)

        # Interior points: central difference
        for i in range(1, n-1):
            dx_left = x_points[i] - x_points[i-1]
            dx_right = x_points[i+1] - x_points[i]
            dx_avg = (dx_left + dx_right) / 2

            # Second derivative using finite difference
            d2u_dx2[i] = (u_values[i+1] - 2*u_values[i] + u_values[i-1]) / (dx_avg**2)

        # Boundary points: use one-sided differences or set to zero for Dirichlet BC
        d2u_dx2[0] = 0.0    # u(0,t) = 0 boundary condition
        d2u_dx2[-1] = 0.0   # u(1,t) = 0 boundary condition

        return d2u_dx2

    def compute_rhs(self, x_points, u_values):
        """
        Compute PDE right-hand side f(x, u)

        Args:
            x_points: Spatial coordinates
            u_values: Current solution values

        Returns:
            rhs: Right-hand side values
        """
        if self.pde_type == 'heat':
            # Heat equation: ∂u/∂t = ∂²u/∂x²
            return self.compute_second_derivative(x_points, u_values)
        else:
            raise ValueError(f"Unknown PDE type: {self.pde_type}")

    def __call__(self, x_points, u_values):
        """Make class callable"""
        return self.compute_rhs(x_points, u_values)

if __name__ == "__main__":
    # Test RSNG implementation
    print("🚀 Testing RSNG Implementation")
    print("=" * 50)

    # Create network
    network = RSNGNeuralApproximator(spatial_dim=1, hidden_units=25, n_layers=4)

    # Test spatial domain
    x_test = np.linspace(0, 1, 50)

    # Test forward pass
    u_test = network.forward(x_test)
    print(f"✅ Forward pass: output shape {u_test.shape}")

    # Test Jacobian computation
    J_test = network.compute_jacobian(x_test[:5])  # Small test
    print(f"✅ Jacobian: shape {J_test.shape}")

    # Test initial condition fitting
    u0_test = np.sin(np.pi * x_test)  # Simple initial condition
    network.fit_initial_condition(x_test, u0_test, max_iterations=500)

    print("\n🎉 RSNG Implementation Test Completed!")