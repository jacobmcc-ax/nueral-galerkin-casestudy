# RSNG Mathematical Formulation - Theory Validation

## Overview
This document contains the extracted and validated mathematical formulations for the Randomized Sparse Neural Galerkin (RSNG) method, ready for TDD implementation.

## Core Mathematical Components

### 1. Evolution PDE and Initial Condition

**PDE Evolution Equation:**
```
∂_t u(t, x) = f(x, u(t, x))
```

**Initial Condition:**
```
u(0, x) = u0(x)
```

### 2. Neural Network Approximation (Ansatz)

**Neural Approximation:**
```
u_hat(x; θ(t)) ≈ u(t, x)
```

Where `θ(t)` are time-dependent neural network parameters.

### 3. Galerkin Projection and Residual Formulation

**Continuous Residual:**
```
r(x; θ(t), θ̇(t)) = ∇_θ u_hat(x; θ(t)) · θ̇(t) - f(x, u_hat(x; θ(t)))
```

**Discrete Residual (Sampled Form):**
```
r_vec = J(θ(t)) θ̇(t) - f(θ(t))
```

Where:
- `J(θ(t)) ∈ R^{n×p}` is the Jacobian matrix at sampled points
- `f(θ(t)) ∈ R^n` is the PDE right-hand side at sampled points
- `θ̇(t) ∈ R^p` is the parameter time derivative

**Least-Squares Objective:**
```
minimize ||J(θ(t)) θ̇(t) - f(θ(t))||_2^2
```

### 4. Sparse Sampling Strategy

**Sketch Matrix:**
```
S_t ∈ R^{p×s}  where s << p
```

**Element-wise Definition:**
```
S_t[j, ℓ] = δ_{j, ξ_ℓ(t)}
```

Where `ξ_ℓ(t) ∈ {1, 2, ..., p}` are randomly selected indices.

**Parameter Relation:**
```
θ̇(t) = S_t θ̇_s(t)
```

Where `θ̇_s(t) ∈ R^s` is the sparse parameter update.

### 5. Sketched Galerkin System

**Sketched Jacobian:**
```
J_s(θ(t)) = J(θ(t)) S_t
```

**Sketched Residual:**
```
r_s = J_s(θ(t)) θ̇_s(t) - f(θ(t))
```

**Sketched Objective:**
```
minimize ||J_s(θ(t)) θ̇_s(t) - f(θ(t))||_2^2
```

### 6. Time Integration (Explicit Euler)

**Time Update:**
```
θ^(k) = θ^(k-1) + Δt S_k Δθ_s^(k-1)
```

**Step Objective:**
```
minimize ||J_s(θ^(k-1)) Δθ_s^(k-1) - f(θ^(k-1))||_2^2
```

## Implementation Parameters

### Dimensions
- `n`: Number of spatial sample points
- `p`: Number of neural network parameters
- `s`: Sparse sketch size (s << p)

### Key Variables
- `θ(t)`: Full parameter vector (p×1)
- `θ̇_s(t)`: Sparse parameter update (s×1)
- `J(θ(t))`: Jacobian matrix (n×p)
- `S_t`: Sketch/selector matrix (p×s)
- `Δt`: Time step size

## TDD Implementation Requirements

### Test Categories for Implementation

#### 1. Neural Network Approximation Tests
- **Test**: Neural network can approximate known analytical solutions
- **Tolerance**: 1e-6 for smooth problems
- **Benchmark**: Heat equation with analytical solution

#### 2. Galerkin Projection Tests
- **Test**: Residual minimization reduces PDE error
- **Tolerance**: Residual reduction > 90%
- **Benchmark**: Linear PDE with known solution

#### 3. Sparse Sampling Tests
- **Test**: Sparse method maintains accuracy with efficiency gain
- **Tolerance**: Accuracy loss < 10%, speedup > 2x
- **Benchmark**: Compare full vs sparse Galerkin

#### 4. Time Integration Tests
- **Test**: Temporal evolution preserves solution accuracy
- **Tolerance**: 1e-5 over integration period
- **Benchmark**: Long-time stability analysis

#### 5. Convergence Tests
- **Test**: Method achieves theoretical convergence rates
- **Expected Rates**: Spatial O(h^p), temporal O(Δt)
- **Validation**: Refinement studies

## Mathematical Validation Checklist

- [x] Core equations extracted and validated
- [x] Parameter dimensions and constraints identified
- [x] Implementation requirements documented
- [x] Test tolerances specified
- [x] Benchmark problems identified
- [ ] TDD test suite implementation
- [ ] Algorithm implementation with TDD
- [ ] Numerical validation and convergence studies

## Next Steps for TDD Implementation

1. **RED Phase**: Write failing tests for neural approximation accuracy
2. **GREEN Phase**: Implement minimal neural network architecture
3. **VERIFY Phase**: Validate numerical accuracy with user
4. **Iterate**: Continue TDD cycle for each component

---

*This mathematical formulation serves as the theoretical foundation for TDD implementation of the RSNG algorithm.*