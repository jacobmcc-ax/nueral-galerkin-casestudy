# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with the Neural Galerkin case study repository.

## Project Overview

This is a Test-Driven Development (TDD) educational repository demonstrating the implementation of Neural Galerkin methods for solving evolution equations. The project follows a rigorous mathematical TDD approach, combining theoretical validation with numerical implementation to ensure mathematical accuracy and computational efficiency.

## Development Philosophy

### Mathematical TDD Cycle
Follow the **Red → Green → Verify** cycle specifically adapted for mathematical algorithms:

1. **RED Phase**: Write failing tests that capture mathematical properties (convergence, accuracy, stability)
2. **GREEN Phase**: Implement minimal mathematical solution to pass tests
3. **VERIFY Phase**: User validation of numerical accuracy and mathematical behavior

### Quality Standards
- **Mathematical Rigor**: All implementations must be mathematically sound
- **Numerical Accuracy**: Tests must validate numerical behavior within appropriate tolerances
- **Theoretical Consistency**: Implementations must match extracted theoretical formulations
- **Computational Efficiency**: Sparse methods should demonstrate efficiency gains

## Repository Structure

```
nueral-galerkin-casestudy/
├── nueral-galerkin.pdf              # Original research paper
├── nueral-galerkin.md               # Extracted markdown with mathematical content
├── test_pdf_extraction.py           # PDF extraction validation tests
├── .claude/
│   ├── agents/
│   │   ├── neural-galerkin-expert.md      # Mathematical algorithm specialist
│   │   ├── pdf-extraction-tester.md       # Document extraction specialist
│   │   └── paper-extraction-agent.md      # Paper analysis specialist
│   ├── output-styles/
│   │   └── pragmatic-tdd-neural-galerkin.md  # TDD methodology enforcement
│   └── commands/
│       └── neural-galerkin/
│           ├── implement-algorithm.md      # Algorithm implementation workflow
│           └── validate-theory.md          # Theoretical validation workflow
├── src/                             # Implementation directory (TDD-generated)
├── tests/                           # Test suites (TDD-driven)
└── results/                         # Numerical analysis and validation
```

## TDD Workflow for Mathematical Algorithms

### Phase 1: Theoretical Foundation (VERIFY before RED)
Before writing any tests, establish mathematical foundations:

```bash
# Extract and validate mathematical formulations from paper
/neural-galerkin/validate-theory rsng_formulation --extract-equations

# Verify core algorithm theory
/neural-galerkin/validate-theory convergence_analysis --section=theoretical_results
```

### Phase 2: Algorithm Implementation (RED → GREEN → VERIFY)
Implement mathematical components using TDD:

```bash
# Implement core RSNG algorithm with TDD
/neural-galerkin/implement-algorithm rsng --test-tolerance=1e-4

# Implement sparse sampling strategy
/neural-galerkin/implement-algorithm sparse_sampling --test-tolerance=1e-5

# Implement time integration
/neural-galerkin/implement-algorithm time_integration --benchmark-problem=heat_equation
```

## Specialized Agents

### @neural-galerkin-expert
Use for mathematical algorithm implementation:
- Neural network architecture for PDE approximation
- Galerkin projection and weak form computation
- Sparse sampling strategies
- Time integration schemes
- Mathematical validation and convergence analysis

### @pdf-extraction-tester
Use for document analysis and extraction:
- TDD-based PDF to Markdown conversion
- Mathematical notation preservation
- Academic paper structure validation
- Content quality assessment

### @paper-extraction-agent
Use for paper analysis and understanding:
- Research paper comprehension
- Mathematical formulation extraction
- Algorithm identification and analysis
- Theoretical foundation establishment

## Mathematical Implementation Standards

### Neural Network Components
```python
class NeuralGalerkinApproximator:
    """Neural network for PDE solution approximation"""

    def forward(self, x, t, theta):
        """Compute u_θ(x,t) - neural approximation of PDE solution"""
        pass

    def compute_residual(self, x, t, theta, pde_func):
        """Compute PDE residual: ∂u/∂t - f(x,u)"""
        pass
```

### Galerkin Projection Implementation
```python
def galerkin_projection(residual_func, test_functions, domain):
    """
    Compute Galerkin projection: ⟨residual, test_function⟩ = 0

    Args:
        residual_func: PDE residual function
        test_functions: Basis functions for projection
        domain: Computational domain

    Returns:
        projected_residual: Weak form residual for optimization
    """
    pass

def sparse_galerkin_projection(residual_func, test_functions, sparse_indices):
    """Sparse Galerkin projection using random subset"""
    pass
```

### Test Structure for Mathematical Components
```python
def test_neural_approximation_accuracy():
    """Test neural network approximates analytical solution within tolerance"""
    # Setup analytical solution
    analytical_solution = lambda x, t: np.exp(-np.pi**2 * t) * np.sin(np.pi * x)

    # Neural network approximation
    nn = NeuralGalerkinApproximator(spatial_dim=1)
    neural_solution = nn.forward(x_test, t_test, theta_optimal)

    # Validate accuracy
    error = np.abs(neural_solution - analytical_solution(x_test, t_test))
    assert np.max(error) < tolerance, f"Neural approximation error {np.max(error)} exceeds tolerance {tolerance}"

def test_sparse_sampling_efficiency():
    """Test sparse sampling maintains accuracy with computational speedup"""
    # Compare full vs sparse Galerkin projection
    # Assert: sparse_time < full_time * efficiency_factor
    # Assert: accuracy_loss < acceptable_degradation
```

## Mathematical Validation Framework

### Convergence Testing
- **Spatial Convergence**: h-refinement studies
- **Temporal Convergence**: Δt-refinement studies
- **Neural Convergence**: Network size scaling
- **Sparse Convergence**: Sparsity level impact

### Benchmark Problems
1. **Heat Equation**: u_t = u_xx with analytical solution
2. **Wave Equation**: u_tt = u_xx with known solutions
3. **Burgers Equation**: u_t + u*u_x = ν*u_xx (nonlinear)
4. **Allen-Cahn**: u_t = ε²*Δu + u - u³ (phase field)

### Error Metrics
- **L² Error**: ||u_neural - u_exact||_L²
- **H¹ Error**: Including derivative accuracy
- **Maximum Error**: sup|u_neural - u_exact|
- **Relative Error**: Normalized error measures

## Integration with Available MCP Tools

### Equation Analysis (Axiomatic)
```python
# Extract mathematical formulations from paper
equation_form = mcp__axiomatic-mcp__AxEquationExplorer_find_functional_form(
    document="nueral-galerkin.md",
    task="Extract the RSNG algorithm mathematical formulation"
)

# Verify implementation matches theory
equation_check = mcp__axiomatic-mcp__AxEquationExplorer_check_equation(
    document="nueral-galerkin.md",
    task="Verify neural Galerkin projection implementation matches equation (12)"
)
```

### Code Execution (Jupyter)
```python
# Execute mathematical implementations and show results
mcp__ide__executeCode(code="""
import numpy as np
import matplotlib.pyplot as plt

# Test neural network convergence
# Show convergence plots, error analysis
""")
```

## Quality Assurance Checklist

### Mathematical Implementation
- [ ] All algorithms match theoretical formulations from paper
- [ ] Numerical accuracy validated within appropriate tolerances
- [ ] Convergence rates match theoretical predictions
- [ ] Stability demonstrated over relevant time scales
- [ ] Computational efficiency gains validated for sparse methods

### Code Quality
- [ ] Clear mathematical documentation with equation references
- [ ] Readable implementation of mathematical operations
- [ ] Appropriate numerical stability measures
- [ ] Comprehensive test coverage for mathematical properties
- [ ] Error handling for edge cases and numerical issues

### TDD Methodology
- [ ] All mathematical tests written before implementation
- [ ] Red phase shows appropriate test failures
- [ ] Green phase implements minimal mathematical solution
- [ ] Verify phase includes user validation of numerical results
- [ ] Progressive complexity building through TDD cycles

## Usage Examples

### Complete Algorithm Implementation Workflow
```bash
# 1. Validate theoretical foundations
/neural-galerkin/validate-theory rsng_formulation --extract-equations

# 2. Implement core components with TDD
/neural-galerkin/implement-algorithm neural_approximation --test-tolerance=1e-6
/neural-galerkin/implement-algorithm galerkin_projection --benchmark-problem=heat_equation
/neural-galerkin/implement-algorithm sparse_sampling --test-tolerance=1e-5

# 3. Integrate and validate complete algorithm
/neural-galerkin/implement-algorithm rsng --benchmark-problem=burgers_equation

# 4. Run comprehensive validation suite
pytest tests/ -v --cov=src/ --cov-report=html
```

### Mathematical Analysis Workflow
```python
# Use specialized agents for mathematical tasks
@neural-galerkin-expert Implement the sparse Galerkin projection with TDD methodology, ensuring numerical accuracy within 1e-5 tolerance for the heat equation benchmark.

@pdf-extraction-tester Validate that all mathematical equations were correctly extracted from the PDF with proper LaTeX formatting.

@paper-extraction-agent Analyze the convergence theory section and extract the theoretical convergence rates for implementation validation.
```

## Expected Outcomes

### Technical Deliverables
- **Mathematical Implementation**: Complete RSNG algorithm implementation
- **Validation Suite**: Comprehensive test coverage of mathematical properties
- **Benchmark Results**: Validation against standard PDE test problems
- **Performance Analysis**: Computational efficiency assessment
- **Documentation**: Mathematical foundations and implementation guide

### Educational Value
- **TDD for Mathematics**: Demonstrates TDD methodology for mathematical algorithms
- **Theory-Practice Bridge**: Shows how theoretical formulations translate to code
- **Numerical Analysis**: Practical convergence and error analysis
- **Research Implementation**: Academic paper to working code workflow

Remember: **Mathematical rigor first, computational optimization second**. The TDD methodology ensures mathematical correctness at every step of the implementation process.