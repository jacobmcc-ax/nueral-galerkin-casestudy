---
allowed-tools: Task, Read, Write, Edit, Bash, mcp__ide__executeCode, TodoWrite
description: Implement Neural Galerkin algorithm component using strict TDD methodology
argument-hint: <algorithm-component> [--test-tolerance=1e-6] [--benchmark-problem=heat_equation]
---

# Neural Galerkin Algorithm Implementation Command

Implement a specific Neural Galerkin algorithm component using Test-Driven Development methodology, ensuring mathematical accuracy and numerical stability.

## Usage Examples

```bash
# Implement core RSNG algorithm
/neural-galerkin/implement-algorithm rsng --test-tolerance=1e-4 --benchmark-problem=heat_equation

# Implement sparse sampling strategy
/neural-galerkin/implement-algorithm sparse_sampling --test-tolerance=1e-5

# Implement time integration scheme
/neural-galerkin/implement-algorithm time_integration --benchmark-problem=advection_diffusion
```

## Algorithm Components Available

### Core Algorithms
- **rsng**: Randomized Sparse Neural Galerkin main algorithm
- **neural_approximation**: Neural network approximation for PDE solutions
- **galerkin_projection**: Galerkin projection and weak form computation
- **sparse_sampling**: Random subset selection for computational efficiency

### Supporting Components
- **time_integration**: Temporal evolution schemes (RK4, adaptive stepping)
- **loss_functions**: Physics-informed loss function design
- **convergence_analysis**: Error estimation and convergence monitoring
- **benchmark_validation**: Standard PDE test problem validation

## TDD Methodology Workflow

### Phase 1: RED - Mathematical Test Creation

The command automatically:
1. **Creates Failing Mathematical Tests**: Based on theoretical requirements
2. **Defines Numerical Tolerances**: Appropriate for the mathematical component
3. **Sets up Benchmark Problems**: Standard test cases for validation
4. **Establishes Success Criteria**: Clear mathematical goals

Example test structure:
```python
def test_neural_network_approximates_heat_equation():
    """Test neural network approximates analytical heat equation solution"""
    # Setup: analytical solution u(x,t) = exp(-pi^2*t)*sin(pi*x)
    # Test: neural network approximation within specified tolerance
    # Assert: |u_neural - u_analytical| < tolerance

def test_rsng_reduces_computational_cost():
    """Test RSNG algorithm achieves computational speedup vs full Galerkin"""
    # Setup: full Galerkin projection computation time
    # Test: RSNG sparse projection computation time
    # Assert: RSNG_time < full_time * efficiency_factor
```

### Phase 2: GREEN - Minimal Mathematical Implementation

1. **Mathematical Correctness First**: Implement core mathematical operations
2. **Minimal Viable Algorithm**: Only what's needed to pass tests
3. **Clear Mathematical Structure**: Readable implementation of equations
4. **No Premature Optimization**: Focus on correctness over performance

### Phase 3: VERIFY - Mathematical Validation

1. **Numerical Results Display**: Show convergence plots, error curves
2. **Benchmark Comparison**: Compare against analytical solutions
3. **User Mathematical Review**: Request validation of numerical behavior
4. **Quality Assessment**: Mathematical accuracy and stability metrics

## Implementation Strategy

### Neural Network Components
```python
class NeuralGalerkinApproximator:
    def __init__(self, spatial_dim, temporal_evolution=True):
        # Neural architecture for PDE approximation

    def forward(self, x, t, theta):
        # Neural network forward pass: u_theta(x,t)

    def compute_residual(self, x, t, theta, pde_function):
        # PDE residual: ∂u/∂t - f(x,u)
```

### Galerkin Projection Implementation
```python
def galerkin_projection(residual_function, test_functions, domain):
    """Compute Galerkin projection of PDE residual"""
    # Weak form: ∫ residual * test_function dx = 0
    # Return: projected residual for optimization

def sparse_galerkin_projection(residual_function, test_functions, sparse_indices):
    """Sparse version using random subset of test functions"""
    # Sparse weak form using subset of test functions
    # Return: sparse projected residual
```

### Time Integration Schemes
```python
def runge_kutta_neural_evolution(theta_current, dt, pde_residual):
    """RK4 time stepping for neural parameters"""
    # Time evolution: θ(t+dt) = θ(t) + dt * k_avg
    # Return: updated neural parameters

def adaptive_time_stepping(theta, pde_residual, error_tolerance):
    """Adaptive time step control for stability"""
    # Adjust dt based on local error estimates
    # Return: optimal time step and updated parameters
```

## Mathematical Validation Framework

### Convergence Testing
- **Spatial Convergence**: Refinement in spatial discretization
- **Temporal Convergence**: Refinement in time stepping
- **Neural Convergence**: Network size and training convergence
- **Sparse Convergence**: Effect of sparsity level on accuracy

### Benchmark Problems
1. **Heat Equation**: 1D/2D diffusion with analytical solutions
2. **Wave Equation**: Hyperbolic PDE with known solutions
3. **Burgers Equation**: Nonlinear advection-diffusion
4. **Allen-Cahn**: Phase field equation with complex dynamics

### Error Metrics
- **L2 Error**: ||u_neural - u_exact||_L2
- **H1 Error**: Including derivative accuracy
- **Maximum Error**: sup|u_neural - u_exact|
- **Relative Error**: Normalized by solution magnitude

## Quality Assurance

### Mathematical Tests Must Pass
- [ ] Numerical accuracy within specified tolerance
- [ ] Convergence rate matches theoretical expectations
- [ ] Stability over long time integration
- [ ] Computational efficiency gains from sparsity
- [ ] Benchmark problem validation

### Code Quality Requirements
- [ ] Clear mathematical documentation
- [ ] Readable equation implementations
- [ ] Appropriate numerical stability measures
- [ ] Error handling for edge cases
- [ ] Comprehensive test coverage

## Integration with Paper Analysis

The command automatically:
1. **References Paper Equations**: Cross-validates against theoretical formulation
2. **Uses Extracted Mathematical Forms**: Leverages equation extraction tools
3. **Validates Implementation**: Ensures numerical behavior matches theory
4. **Documents Assumptions**: Clear mathematical assumptions and limitations

## Output Structure

```
neural-galerkin-implementation/
├── tests/
│   ├── test_[algorithm_component].py    # TDD test suite
│   └── benchmarks/
│       └── [benchmark_problem].py       # Standard test problems
├── src/
│   ├── [algorithm_component].py         # Core implementation
│   └── utils/
│       ├── neural_networks.py          # Neural architecture utilities
│       ├── pde_problems.py             # PDE problem definitions
│       └── visualization.py            # Result plotting and analysis
└── results/
    ├── convergence_analysis.png         # Numerical analysis plots
    ├── benchmark_comparison.png         # Validation against analytical solutions
    └── mathematical_validation.md       # Detailed mathematical assessment
```

## Success Criteria

### Must Achieve for Algorithm Component
- [ ] All mathematical tests pass with appropriate tolerances
- [ ] Numerical convergence matches theoretical rates
- [ ] Benchmark problems solved within expected accuracy
- [ ] Computational efficiency demonstrated (for sparse methods)
- [ ] User verification of mathematical behavior completed

### Ready for Integration
- [ ] Component works with existing neural Galerkin framework
- [ ] Clear interfaces for composition with other components
- [ ] Comprehensive documentation of mathematical assumptions
- [ ] Validation against paper's theoretical formulation

Remember: **Mathematical rigor first, computational efficiency second**. The TDD approach ensures each algorithm component is mathematically sound before optimization.