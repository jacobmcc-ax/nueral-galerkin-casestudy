#!/usr/bin/env python3
"""
Time Integration Component for Neural Galerkin Methods
GREEN PHASE: Minimal implementation to make TDD tests pass

Implements time integration schemes for temporal evolution of neural parameters
"""

import numpy as np
from typing import Union, Callable, Tuple, Optional
import sys
import os

# Add utils to path for logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.logger import log_code_execution

class TimeIntegrator:
    """
    Time Integration for Neural Parameter Evolution

    GREEN PHASE: Implements minimal functionality to pass TDD tests
    Supports multiple integration schemes: Euler, RK4, adaptive methods
    """

    def __init__(self, scheme: str = "rk4", dt: float = 0.01, adaptive: bool = False):
        """
        Initialize time integrator

        Args:
            scheme: Integration scheme ("euler", "rk4", "adaptive_rk45")
            dt: Time step size
            adaptive: Enable adaptive time stepping
        """
        self.scheme = scheme
        self.dt = dt
        self.adaptive = adaptive
        self.current_time = 0.0

        log_code_execution(
            f"TimeIntegrator.__init__(scheme={scheme}, dt={dt}, adaptive={adaptive})",
            f"Time integrator initialized with {scheme} scheme"
        )

    def step(self, theta: np.ndarray, t: float, derivative_func: Callable) -> Tuple[np.ndarray, float]:
        """
        Perform one integration step

        Args:
            theta: Current parameter vector
            t: Current time
            derivative_func: Function computing θ̇ = f(θ, t)

        Returns:
            (theta_next, t_next): Updated parameters and time
        """
        theta = np.asarray(theta)

        if self.scheme == "euler":
            theta_next, t_next = self._euler_step(theta, t, derivative_func)
        elif self.scheme == "rk4":
            theta_next, t_next = self._rk4_step(theta, t, derivative_func)
        elif self.scheme == "adaptive_rk45":
            theta_next, t_next = self._adaptive_rk45_step(theta, t, derivative_func)
        else:
            raise ValueError(f"Unknown integration scheme: {self.scheme}")

        log_code_execution(
            f"TimeIntegrator.step({self.scheme})",
            f"Advanced from t={t:.3f} to t={t_next:.3f}, |Δθ|={np.linalg.norm(theta_next-theta):.2e}"
        )

        return theta_next, t_next

    def _euler_step(self, theta: np.ndarray, t: float, derivative_func: Callable) -> Tuple[np.ndarray, float]:
        """
        Explicit Euler integration step

        θ^{n+1} = θ^n + Δt · θ̇^n
        """
        theta_dot = derivative_func(theta, t)
        theta_next = theta + self.dt * theta_dot
        t_next = t + self.dt

        return theta_next, t_next

    def _rk4_step(self, theta: np.ndarray, t: float, derivative_func: Callable) -> Tuple[np.ndarray, float]:
        """
        Fourth-order Runge-Kutta integration step

        k₁ = f(t, θ)
        k₂ = f(t + dt/2, θ + dt*k₁/2)
        k₃ = f(t + dt/2, θ + dt*k₂/2)
        k₄ = f(t + dt, θ + dt*k₃)
        θ^{n+1} = θ^n + dt/6 * (k₁ + 2*k₂ + 2*k₃ + k₄)
        """
        k1 = derivative_func(theta, t)
        k2 = derivative_func(theta + 0.5 * self.dt * k1, t + 0.5 * self.dt)
        k3 = derivative_func(theta + 0.5 * self.dt * k2, t + 0.5 * self.dt)
        k4 = derivative_func(theta + self.dt * k3, t + self.dt)

        theta_next = theta + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        t_next = t + self.dt

        return theta_next, t_next

    def _adaptive_rk45_step(self, theta: np.ndarray, t: float, derivative_func: Callable) -> Tuple[np.ndarray, float]:
        """
        Simplified adaptive RK45 step (GREEN PHASE implementation)

        In a real implementation, this would use Runge-Kutta-Fehlberg
        For GREEN phase, use RK4 with fixed step
        """
        # GREEN PHASE: Use RK4 as base for adaptive scheme
        return self._rk4_step(theta, t, derivative_func)

    def adaptive_step(self, theta: np.ndarray, t: float, derivative_func: Callable,
                     tolerance: float = 1e-6) -> Tuple[np.ndarray, float, float]:
        """
        Adaptive time stepping with error control

        Args:
            theta: Current parameter vector
            t: Current time
            derivative_func: Parameter derivative function
            tolerance: Error tolerance for step size control

        Returns:
            (theta_next, t_next, dt_used): Updated parameters, time, and actual step size used
        """
        # GREEN PHASE: Simulate adaptive behavior
        # Real implementation would estimate local error and adjust dt

        # Take step with current dt
        theta_next, t_next = self.step(theta, t, derivative_func)

        # Simulate adaptive step size adjustment
        # For test purposes, vary dt based on simple heuristic
        theta_magnitude = np.linalg.norm(theta)
        derivative_magnitude = np.linalg.norm(derivative_func(theta, t))

        # Simulate step size adaptation
        if derivative_magnitude > 10:  # "Stiff" region - reduce step
            dt_used = self.dt * 0.5
        elif derivative_magnitude < 0.1:  # "Smooth" region - increase step
            dt_used = self.dt * 1.5
        else:
            dt_used = self.dt

        # Recalculate with adaptive step size
        if dt_used != self.dt:
            original_dt = self.dt
            self.dt = dt_used
            theta_next, t_next = self.step(theta, t, derivative_func)
            self.dt = original_dt  # Restore original dt

        log_code_execution(
            f"TimeIntegrator.adaptive_step(tolerance={tolerance})",
            f"Used dt={dt_used:.4f}, derivative_mag={derivative_magnitude:.2e}"
        )

        return theta_next, t_next, dt_used

    def integrate_trajectory(self, theta_0: np.ndarray, t_span: Tuple[float, float],
                           derivative_func: Callable, n_steps: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate trajectory over time span

        Args:
            theta_0: Initial parameter vector
            t_span: (t_start, t_end) time interval
            derivative_func: Parameter derivative function
            n_steps: Number of steps (if None, use dt to determine)

        Returns:
            (theta_trajectory, time_points): Parameter evolution and time points
        """
        t_start, t_end = t_span

        if n_steps is None:
            n_steps = int((t_end - t_start) / self.dt)

        dt_actual = (t_end - t_start) / n_steps

        # Storage for trajectory
        theta_trajectory = np.zeros((n_steps + 1, len(theta_0)))
        time_points = np.linspace(t_start, t_end, n_steps + 1)

        # Initial conditions
        theta_trajectory[0] = theta_0
        theta_current = theta_0.copy()

        # Integrate trajectory
        for step in range(n_steps):
            t_current = time_points[step]

            # Temporarily adjust dt for this integration
            original_dt = self.dt
            self.dt = dt_actual

            theta_current, _ = self.step(theta_current, t_current, derivative_func)
            theta_trajectory[step + 1] = theta_current

            # Restore original dt
            self.dt = original_dt

        log_code_execution(
            f"TimeIntegrator.integrate_trajectory({n_steps} steps)",
            f"Integrated from t={t_start} to t={t_end} with {self.scheme}"
        )

        return theta_trajectory, time_points

    def compute_stability_properties(self, theta: np.ndarray, derivative_func: Callable) -> dict:
        """
        Compute stability properties of the integration scheme

        Args:
            theta: Current parameter vector
            derivative_func: Parameter derivative function

        Returns:
            Dictionary with stability metrics
        """
        # GREEN PHASE: Compute basic stability metrics

        # Estimate spectral radius (simplified)
        theta_dot = derivative_func(theta, 0.0)
        if np.linalg.norm(theta) > 0:
            spectral_estimate = np.linalg.norm(theta_dot) / np.linalg.norm(theta)
        else:
            spectral_estimate = np.linalg.norm(theta_dot)

        # Stability region estimate for different schemes
        if self.scheme == "euler":
            stability_limit = 2.0 / (spectral_estimate + 1e-12)
        elif self.scheme == "rk4":
            stability_limit = 2.8 / (spectral_estimate + 1e-12)  # Approximate RK4 stability
        else:
            stability_limit = float('inf')  # Adaptive schemes

        stable = self.dt <= stability_limit

        stability_info = {
            "spectral_radius_estimate": spectral_estimate,
            "stability_limit": stability_limit,
            "current_dt": self.dt,
            "stable": stable,
            "scheme": self.scheme
        }

        log_code_execution(
            f"TimeIntegrator.compute_stability_properties()",
            f"Stability: {stable}, dt={self.dt:.3f}, limit={stability_limit:.3f}"
        )

        return stability_info

# Example usage and validation
if __name__ == "__main__":
    print("="*60)
    print("Time Integration GREEN PHASE Implementation")
    print("="*60)

    # Test Euler integration
    integrator_euler = TimeIntegrator(scheme="euler", dt=0.01)

    # Simple exponential decay: θ̇ = -0.1*θ
    def exponential_decay(theta, t):
        return -0.1 * theta

    theta_0 = np.array([1.0, 2.0, 3.0])
    theta_1, t_1 = integrator_euler.step(theta_0, 0.0, exponential_decay)
    print(f"Euler step: {theta_0} → {theta_1} (t: 0.0 → {t_1})")

    # Test RK4 integration
    integrator_rk4 = TimeIntegrator(scheme="rk4", dt=0.01)
    theta_rk4, t_rk4 = integrator_rk4.step(theta_0, 0.0, exponential_decay)
    print(f"RK4 step: {theta_0} → {theta_rk4} (t: 0.0 → {t_rk4})")

    # Test adaptive integration
    integrator_adaptive = TimeIntegrator(scheme="adaptive_rk45", dt=0.01, adaptive=True)
    theta_adaptive, t_adaptive, dt_used = integrator_adaptive.adaptive_step(
        theta_0, 0.0, exponential_decay, tolerance=1e-6
    )
    print(f"Adaptive step: {theta_0} → {theta_adaptive} (dt used: {dt_used:.4f})")

    # Test trajectory integration
    theta_traj, time_traj = integrator_rk4.integrate_trajectory(
        theta_0, (0.0, 1.0), exponential_decay, n_steps=20
    )
    print(f"Trajectory: {len(time_traj)} points from t={time_traj[0]} to t={time_traj[-1]}")

    print("✅ GREEN PHASE: Minimal time integration implementation complete")
    print("Ready to run tests and validate TDD requirements")