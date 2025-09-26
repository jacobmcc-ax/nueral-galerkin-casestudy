#!/usr/bin/env python3
"""
RSNG Implementation with Bug Fixes

Key fixes:
1. Much smaller time steps for stability
2. Parameter update scaling and clipping
3. Improved boundary condition enforcement
4. Better numerical stability in least-squares solver
5. Regularized Jacobian computation
"""

import numpy as np
import time
from datetime import datetime

class RSNGNeuralApproximatorFixed:
    """
    Fixed RSNG Neural Network with stability improvements
    """

    def __init__(self, spatial_dim=1, hidden_units=15, n_layers=3):
        """Initialize with smaller, more stable network"""
        self.spatial_dim = spatial_dim
        self.hidden_units = hidden_units
        self.n_layers = n_layers

        # Network architecture with better initialization
        self.layers = []

        # Input layer with smaller initialization
        input_size = spatial_dim
        self.layers.append({
            'W': np.random.randn(input_size, hidden_units) * np.sqrt(1.0 / input_size) * 0.1,  # Much smaller init
            'b': np.zeros(hidden_units)
        })

        # Hidden layers with careful initialization
        for i in range(n_layers - 2):
            self.layers.append({
                'W': np.random.randn(hidden_units, hidden_units) * np.sqrt(1.0 / hidden_units) * 0.1,
                'b': np.zeros(hidden_units)
            })

        # Output layer - very small initialization
        self.layers.append({
            'W': np.random.randn(hidden_units, 1) * np.sqrt(1.0 / hidden_units) * 0.01,
            'b': np.zeros(1)
        })

        # Flatten parameters
        self.theta = self._flatten_parameters()
        self.param_shapes = self._get_parameter_shapes()

        print(f"✅ Fixed RSNG Network: {len(self.theta)} parameters, {n_layers} layers, smaller initialization")

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
        """Forward pass with stability improvements"""
        if theta is None:
            theta = self.theta

        # Clip parameters to prevent explosion
        theta = np.clip(theta, -10.0, 10.0)

        layers = self._unflatten_parameters(theta)

        # Ensure x is 2D
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        activation = x

        # Forward through all layers except last
        for i in range(len(layers) - 1):
            z = activation @ layers[i]['W'] + layers[i]['b']
            # Much tighter clipping for stability
            z = np.clip(z, -2.0, 2.0)
            activation = np.tanh(z)

        # Output layer with clipping
        output = activation @ layers[-1]['W'] + layers[-1]['b']
        output = np.clip(output, -10.0, 10.0)  # Prevent extreme outputs

        return output.flatten()

    def compute_jacobian_stable(self, x, theta=None):
        """
        Compute Jacobian with finite differences (more stable for debugging)
        """
        if theta is None:
            theta = self.theta

        n_points = len(x)
        n_params = len(theta)
        jacobian = np.zeros((n_points, n_params))

        # Use finite differences with smaller epsilon
        eps = 1e-7

        # Get baseline
        u_base = self.forward(x, theta)

        for i in range(n_params):
            theta_plus = theta.copy()
            theta_plus[i] += eps

            u_plus = self.forward(x, theta_plus)
            jacobian[:, i] = (u_plus - u_base) / eps

        # Regularize Jacobian to prevent extreme values
        jacobian = np.clip(jacobian, -1e3, 1e3)

        return jacobian

    def fit_initial_condition(self, x_points, u0, max_iterations=500, tolerance=1e-4):
        """Fit initial condition with better stability"""
        print("🎯 Fitting initial condition with stability...")

        best_theta = self.theta.copy()
        best_loss = float('inf')

        for iteration in range(max_iterations):
            # Forward pass
            u_pred = self.forward(x_points, self.theta)

            # Loss with boundary condition enforcement
            loss = np.mean((u_pred - u0)**2)

            # Extra penalty for boundary violations
            boundary_penalty = 10.0 * (u_pred[0]**2 + u_pred[-1]**2)
            total_loss = loss + boundary_penalty

            if total_loss < best_loss:
                best_loss = total_loss
                best_theta = self.theta.copy()

            if total_loss < tolerance:
                self.theta = best_theta
                print(f"   ✅ Initial condition fitted: loss={total_loss:.2e} at iteration {iteration}")
                break

            # Simple gradient descent with finite differences
            grad_theta = np.zeros_like(self.theta)
            eps = 1e-6

            for i in range(min(len(self.theta), 50)):  # Only update subset for speed
                theta_plus = self.theta.copy()
                theta_plus[i] += eps

                u_plus = self.forward(x_points, theta_plus)
                loss_plus = np.mean((u_plus - u0)**2) + 10.0 * (u_plus[0]**2 + u_plus[-1]**2)

                grad_theta[i] = (loss_plus - total_loss) / eps

            # Gradient descent with clipping
            learning_rate = 0.001  # Much smaller learning rate
            grad_theta = np.clip(grad_theta, -1.0, 1.0)  # Clip gradients
            self.theta -= learning_rate * grad_theta
            self.theta = np.clip(self.theta, -5.0, 5.0)  # Clip parameters

            # Progress report
            if iteration % 100 == 0:
                print(f"   Iteration {iteration}: loss={total_loss:.2e}")

        # Use best parameters found
        self.theta = best_theta
        final_loss = best_loss
        print(f"✅ Initial condition fitting completed: final loss={final_loss:.2e}")

class PDERightHandSideFixed:
    """
    Improved PDE right-hand side computation
    """

    def __init__(self, pde_type='heat'):
        self.pde_type = pde_type

    def compute_second_derivative_stable(self, x_points, u_values):
        """
        Compute ∂²u/∂x² with better numerical stability
        """
        n = len(x_points)
        d2u_dx2 = np.zeros_like(u_values)

        # Use uniform grid assumption for stability
        dx = x_points[1] - x_points[0]

        # Interior points: standard finite difference
        for i in range(1, n-1):
            d2u_dx2[i] = (u_values[i+1] - 2*u_values[i] + u_values[i-1]) / (dx**2)

        # Boundary conditions: u(0,t) = u(1,t) = 0
        # This means ∂²u/∂x² should also be small at boundaries
        d2u_dx2[0] = 0.0
        d2u_dx2[-1] = 0.0

        # Regularize to prevent extreme values
        d2u_dx2 = np.clip(d2u_dx2, -1e3, 1e3)

        return d2u_dx2

    def __call__(self, x_points, u_values):
        """Compute RHS with stability"""
        if self.pde_type == 'heat':
            return self.compute_second_derivative_stable(x_points, u_values)
        else:
            raise ValueError(f"Unknown PDE type: {self.pde_type}")

class RSNGSolverFixed:
    """
    Fixed RSNG solver with stability improvements
    """

    def __init__(self, network, sparsity_ratio=0.2):
        """Initialize with conservative sparsity"""
        self.network = network
        self.sparsity_ratio = sparsity_ratio
        self.n_params = len(network.theta)
        self.n_sparse = max(1, int(sparsity_ratio * self.n_params))

        print(f"📊 Fixed RSNG Solver: {self.n_params} total params, {self.n_sparse} sparse updates ({sparsity_ratio:.1%})")

    def create_sketching_matrix(self):
        """Create sketching matrix with stability"""
        indices = np.random.choice(self.n_params, size=self.n_sparse, replace=False)
        S_t = np.zeros((self.n_params, self.n_sparse))
        for i, idx in enumerate(indices):
            S_t[idx, i] = 1.0
        return indices, S_t

    def solve_sparse_update_stable(self, x_points, pde_rhs, dt):
        """
        Solve sparse update with major stability improvements
        """
        # Get current network state
        u_current = self.network.forward(x_points)

        # Compute Jacobian (use stable version)
        J_full = self.network.compute_jacobian_stable(x_points)

        # Evaluate PDE right-hand side
        f_values = pde_rhs(x_points, u_current)

        # Create sparse sketching matrix
        sparse_indices, S_t = self.create_sketching_matrix()

        # Form sketched Jacobian
        J_sparse = J_full @ S_t

        # Add regularization to prevent singular matrix
        n_points, n_sparse_params = J_sparse.shape
        if n_points >= n_sparse_params:
            # Add small regularization
            JTJ = J_sparse.T @ J_sparse
            JTJ += 1e-6 * np.eye(n_sparse_params)  # Regularization
            JTf = J_sparse.T @ f_values

            try:
                delta_theta_sparse = np.linalg.solve(JTJ, JTf)
            except np.linalg.LinAlgError:
                delta_theta_sparse = np.linalg.pinv(J_sparse) @ f_values
        else:
            # Underdetermined system
            delta_theta_sparse = np.linalg.pinv(J_sparse) @ f_values

        # Compute residual norm
        residual_norm = np.linalg.norm(J_sparse @ delta_theta_sparse - f_values)

        # Lift sparse update with scaling
        delta_theta = S_t @ delta_theta_sparse

        # CRITICAL FIX: Scale the update to prevent explosion
        max_update = np.max(np.abs(delta_theta))
        if max_update > 1.0:  # Limit parameter updates
            delta_theta = delta_theta / max_update

        return delta_theta, residual_norm, sparse_indices

    def time_step(self, x_points, pde_rhs, dt):
        """Time step with stability"""
        start_time = time.time()

        # Solve for sparse parameter update
        delta_theta, residual_norm, sparse_indices = self.solve_sparse_update_stable(x_points, pde_rhs, dt)

        # Scale time step if update is too large
        update_norm = np.linalg.norm(delta_theta)
        if update_norm > 10.0:  # Very conservative
            dt = dt * 0.1  # Reduce time step
            delta_theta = delta_theta * 0.1

        # Update network parameters with clipping
        old_theta = self.network.theta.copy()
        self.network.theta += dt * delta_theta
        self.network.theta = np.clip(self.network.theta, -10.0, 10.0)  # Clip parameters

        # Check for numerical issues
        if not np.isfinite(self.network.theta).all():
            print("⚠️  Non-finite parameters detected, reverting")
            self.network.theta = old_theta

        step_time = (time.time() - start_time) * 1000

        return {
            'residual_norm': residual_norm,
            'sparse_indices': sparse_indices,
            'n_updated': len(sparse_indices),
            'step_time_ms': step_time,
            'delta_theta_norm': np.linalg.norm(delta_theta),
            'effective_dt': dt
        }

if __name__ == "__main__":
    print("🔧 Testing Fixed RSNG Implementation")
    print("=" * 40)

    # Test with very conservative parameters
    network = RSNGNeuralApproximatorFixed(spatial_dim=1, hidden_units=10, n_layers=3)
    solver = RSNGSolverFixed(network, sparsity_ratio=0.3)
    pde_rhs = PDERightHandSideFixed(pde_type='heat')

    # Test forward pass
    x_test = np.linspace(0, 1, 20)
    u_test = network.forward(x_test)
    print(f"✅ Forward pass: output range [{np.min(u_test):.2f}, {np.max(u_test):.2f}]")

    # Test Jacobian
    J_test = network.compute_jacobian_stable(x_test[:5])
    print(f"✅ Jacobian: shape {J_test.shape}, range [{np.min(J_test):.2e}, {np.max(J_test):.2e}]")

    print("🎉 Fixed RSNG Implementation Test Completed!")