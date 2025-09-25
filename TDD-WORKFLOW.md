# Neural Galerkin TDD Workflow Guide

## Overview

This repository now includes a comprehensive Test-Driven Development (TDD) workflow specifically designed for implementing Neural Galerkin methods for solving evolution equations. The workflow combines mathematical rigor with software engineering best practices.

## TDD Workflow Components

### 1. **Pragmatic TDD Neural Galerkin Output Style**
`.claude/output-styles/pragmatic-tdd-neural-galerkin.md`

Enforces the mathematical TDD cycle:
- **RED**: Write failing mathematical tests first
- **GREEN**: Implement minimal mathematical solution
- **VERIFY**: User validation of numerical accuracy

**To Enable:**
```bash
/output-style
# Select "Pragmatic TDD Neural Galerkin Developer"
```

### 2. **Specialized Agents**

#### Neural Galerkin Expert (`@neural-galerkin-expert`)
- Neural network architecture for PDE approximation
- Galerkin projection and sparse sampling implementation
- Mathematical validation and convergence analysis
- TDD methodology for mathematical algorithms

#### PDF Extraction Tester (`@pdf-extraction-tester`)
- TDD-based document conversion validation
- Mathematical notation preservation testing
- Academic paper structure verification

#### Paper Extraction Agent (`@paper-extraction-agent`)
- Research paper analysis and comprehension
- Mathematical formulation extraction
- Algorithm identification from academic papers

### 3. **Specialized Commands**

#### Theory Validation Command
```bash
/neural-galerkin/validate-theory rsng_formulation --extract-equations
/neural-galerkin/validate-theory convergence_analysis --section=theoretical_results
```

**Purpose**: Extract and validate mathematical formulations from the paper before implementation

#### Algorithm Implementation Command
```bash
/neural-galerkin/implement-algorithm rsng --test-tolerance=1e-4
/neural-galerkin/implement-algorithm sparse_sampling --test-tolerance=1e-5
/neural-galerkin/implement-algorithm time_integration --benchmark-problem=heat_equation
```

**Purpose**: Implement specific algorithm components using strict TDD methodology

## Complete Workflow Example

### Step 1: Enable TDD Output Style
```bash
/output-style
# Select "Pragmatic TDD Neural Galerkin Developer"
```

### Step 2: Validate Theoretical Foundations
```bash
/neural-galerkin/validate-theory rsng_formulation --extract-equations
```

This will:
- Extract key mathematical equations from the paper
- Validate theoretical assumptions and parameters
- Create theory-to-implementation mapping
- Document expected numerical behavior

### Step 3: Implement Core Components with TDD

#### Neural Network Approximation
```bash
/neural-galerkin/implement-algorithm neural_approximation --test-tolerance=1e-6
```

**TDD Cycle:**
1. **RED**: Write test for neural network approximating analytical PDE solution
2. **GREEN**: Implement minimal neural network architecture
3. **VERIFY**: User validates numerical approximation accuracy

#### Galerkin Projection
```bash
/neural-galerkin/implement-algorithm galerkin_projection --benchmark-problem=heat_equation
```

**TDD Cycle:**
1. **RED**: Write test for weak form computation and residual minimization
2. **GREEN**: Implement Galerkin projection mathematics
3. **VERIFY**: User validates projection reduces PDE residual

#### Sparse Sampling
```bash
/neural-galerkin/implement-algorithm sparse_sampling --test-tolerance=1e-5
```

**TDD Cycle:**
1. **RED**: Write test for computational efficiency without accuracy loss
2. **GREEN**: Implement random subset selection strategy
3. **VERIFY**: User validates efficiency gains and accuracy preservation

### Step 4: Complete RSNG Algorithm
```bash
/neural-galerkin/implement-algorithm rsng --benchmark-problem=burgers_equation
```

**TDD Cycle:**
1. **RED**: Write comprehensive test for complete RSNG algorithm
2. **GREEN**: Integrate components into full algorithm
3. **VERIFY**: User validates against benchmark PDE problems

## Key Features

### Mathematical TDD Methodology
- **Theory First**: Always validate theoretical foundations before coding
- **Test First**: Write mathematical tests before implementation
- **Minimal Implementation**: Focus on mathematical correctness over optimization
- **User Validation**: Mathematical accuracy confirmed at each step

### Integration with Axiomatic MCP Tools
- **Equation Extraction**: Automatically extract mathematical formulations from papers
- **Theory Validation**: Verify implementations match theoretical formulations
- **Mathematical Analysis**: Advanced equation manipulation and verification

### Comprehensive Testing Framework
- **Convergence Tests**: Validate numerical convergence rates
- **Accuracy Tests**: Compare with analytical solutions
- **Efficiency Tests**: Validate computational speedup from sparse methods
- **Stability Tests**: Ensure long-term numerical stability

## Comparison with Browserbase TDD Workflow

| Feature | Browserbase Workflow | Neural Galerkin Workflow |
|---------|---------------------|---------------------------|
| **Domain** | Web UI Testing | Mathematical Algorithm Implementation |
| **Test Focus** | User interactions, visual elements | Numerical accuracy, convergence, stability |
| **Languages** | Natural language (Stagehand) | Mathematical formulations and code |
| **Validation** | Visual/behavioral validation | Numerical and theoretical validation |
| **Tools** | Stagehand, Playwright | Axiomatic MCP, Jupyter, Scientific Computing |
| **Success Metrics** | UI tests pass, user workflows work | Mathematical tests pass, numerical accuracy achieved |

### Shared TDD Principles
- **Red → Green → Verify cycle**
- **User validation at each step**
- **Minimal implementation approach**
- **Strict methodology enforcement**
- **Specialized agents for domain expertise**

## Expected Repository Structure After Implementation

```
nueral-galerkin-casestudy/
├── src/
│   ├── neural_networks/
│   │   ├── approximator.py              # Neural network for PDE approximation
│   │   └── architectures.py             # Network architecture utilities
│   ├── galerkin/
│   │   ├── projection.py                # Galerkin projection implementation
│   │   ├── sparse_sampling.py           # Random subset selection
│   │   └── weak_form.py                 # Weak form computation
│   ├── time_integration/
│   │   ├── runge_kutta.py              # RK4 time stepping
│   │   └── adaptive_stepping.py         # Adaptive time control
│   ├── algorithms/
│   │   ├── rsng.py                     # Complete RSNG algorithm
│   │   └── classical_galerkin.py        # Comparison methods
│   └── utils/
│       ├── pde_problems.py             # Benchmark PDE definitions
│       ├── analytical_solutions.py     # Known analytical solutions
│       └── visualization.py            # Result plotting utilities
├── tests/
│   ├── test_neural_approximation.py    # Neural network accuracy tests
│   ├── test_galerkin_projection.py     # Galerkin projection tests
│   ├── test_sparse_sampling.py         # Sparse sampling efficiency tests
│   ├── test_time_integration.py        # Time stepping accuracy tests
│   ├── test_rsng_algorithm.py          # Complete algorithm validation
│   └── benchmarks/
│       ├── test_heat_equation.py       # Heat equation benchmark
│       ├── test_wave_equation.py       # Wave equation benchmark
│       └── test_burgers_equation.py    # Burgers equation benchmark
├── results/
│   ├── convergence_analysis/           # Convergence study results
│   ├── benchmark_validation/           # Benchmark problem results
│   ├── efficiency_analysis/            # Computational efficiency studies
│   └── mathematical_validation/        # Theory vs implementation validation
└── docs/
    ├── mathematical_foundations.md     # Extracted theoretical formulations
    ├── implementation_guide.md         # TDD implementation documentation
    └── numerical_analysis.md           # Convergence and error analysis
```

## Usage Tips

### Agent Selection
- Use `@neural-galerkin-expert` for mathematical algorithm implementation
- Use `@pdf-extraction-tester` for document validation tasks
- Use `@paper-extraction-agent` for paper analysis and understanding

### Command Selection
- Start with `/neural-galerkin/validate-theory` to establish foundations
- Use `/neural-galerkin/implement-algorithm` for TDD implementation of components
- Always specify appropriate test tolerances based on mathematical requirements

### Mathematical Validation
- Always validate numerical results against analytical solutions when available
- Request user confirmation of mathematical behavior before proceeding
- Document any deviations from theoretical predictions

## Success Metrics

### Technical Achievement
- [ ] Complete RSNG algorithm implementation following TDD methodology
- [ ] All mathematical tests pass with appropriate tolerances
- [ ] Numerical convergence rates match theoretical predictions
- [ ] Computational efficiency gains demonstrated for sparse methods
- [ ] Benchmark problems solved with validated accuracy

### Educational Value
- [ ] Clear demonstration of TDD for mathematical algorithms
- [ ] Bridge between theoretical formulations and practical implementation
- [ ] Comprehensive testing methodology for numerical methods
- [ ] Integration of academic research with software engineering practices

---

**Remember**: This TDD workflow prioritizes mathematical rigor and numerical accuracy over implementation speed. The systematic approach ensures each component is mathematically sound before building complexity.