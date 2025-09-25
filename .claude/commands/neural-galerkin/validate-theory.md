---
allowed-tools: Task, mcp__axiomatic-mcp__AxEquationExplorer_find_functional_form, mcp__axiomatic-mcp__AxEquationExplorer_check_equation, Read, Write, TodoWrite
description: Validate theoretical foundations by extracting and verifying mathematical formulations from the paper
argument-hint: <mathematical-component> [--section=theory] [--extract-equations]
---

# Neural Galerkin Theory Validation Command

Extract and validate mathematical formulations from the neural Galerkin paper to ensure implementation matches theoretical foundations.

## Usage Examples

```bash
# Validate core RSNG mathematical formulation
/neural-galerkin/validate-theory rsng_formulation --section=methodology --extract-equations

# Verify convergence theory
/neural-galerkin/validate-theory convergence_analysis --section=theoretical_results

# Extract and validate loss function formulation
/neural-galerkin/validate-theory loss_functions --extract-equations

# Comprehensive theory validation
/neural-galerkin/validate-theory complete --section=all --extract-equations
```

## Mathematical Components for Validation

### Core Theoretical Foundations
- **rsng_formulation**: Main RSNG algorithm mathematical definition
- **galerkin_projection**: Weak form and projection theory
- **neural_approximation**: Universal approximation and function space theory
- **sparse_sampling**: Random sampling theory and convergence guarantees

### Advanced Theory Components
- **convergence_analysis**: Theoretical convergence rates and error bounds
- **stability_analysis**: Temporal stability conditions and criteria
- **efficiency_theory**: Computational complexity and sparse approximation theory
- **loss_functions**: Physics-informed loss formulation and optimization theory

## Validation Methodology

### Phase 1: Mathematical Extraction

Using Axiomatic Equation Explorer tools:
1. **Extract Key Equations**: Identify and extract mathematical formulations
2. **Parse Mathematical Notation**: Convert to implementable form
3. **Identify Parameters**: Extract key mathematical parameters and constraints
4. **Document Assumptions**: Capture theoretical assumptions and conditions

Example extraction workflow:
```python
# Extract core RSNG equation
rsng_equation = extract_functional_form(
    document=paper_content,
    task="Extract the main RSNG algorithm formulation including the sparse Galerkin projection"
)

# Verify implementation matches theory
implementation_check = check_equation(
    document=paper_content,
    task="Verify that the implemented RSNG matches the theoretical formulation in equation (12)"
)
```

### Phase 2: Theoretical Validation

1. **Cross-Reference Implementation**: Compare code against extracted equations
2. **Parameter Validation**: Ensure mathematical parameters match theory
3. **Assumption Verification**: Validate implementation assumptions against paper
4. **Mathematical Consistency**: Check for internal mathematical consistency

### Phase 3: Documentation and Verification

1. **Theory-Implementation Map**: Create mapping between theory and code
2. **Mathematical Documentation**: Document theoretical foundations
3. **User Review**: Request validation of theoretical interpretation
4. **Implementation Guidance**: Provide guidance for TDD implementation

## Mathematical Extraction Strategy

### Key Equations to Extract

#### Core RSNG Formulation
- Evolution equation: $\partial_t u = f(x, u)$
- Neural approximation: $u_\theta(t, x)$
- Galerkin projection: $\langle \partial_t u_\theta - f(x, u_\theta), v_i \rangle = 0$
- Sparse sampling: Random subset $S(t) \subset \{1, 2, \ldots, n\}$

#### Loss Function Formulation
- Residual loss: $L_{res}(\theta) = \|\partial_t u_\theta - f(x, u_\theta)\|_{L^2(\Omega)}^2$
- Boundary conditions: $L_{bc}(\theta) = \|u_\theta|_{\partial\Omega} - g\|^2$
- Initial conditions: $L_{ic}(\theta) = \|u_\theta(0, x) - u_0(x)\|^2$
- Total loss: $L(\theta) = L_{res} + \lambda_{bc} L_{bc} + \lambda_{ic} L_{ic}$

#### Convergence Theory
- Error bounds: $\|u - u_\theta\|_{L^2} \leq C h^p + C \tau^q$
- Convergence rates: Spatial order $p$, temporal order $q$
- Sparse approximation error: $\|P_S u - P u\|$ where $P_S$ is sparse projection

### Theoretical Analysis Framework

#### Mathematical Rigor Checks
1. **Well-posedness**: Verify problem is well-posed mathematically
2. **Existence and Uniqueness**: Check theoretical guarantees
3. **Regularity**: Verify solution regularity assumptions
4. **Stability**: Theoretical stability conditions

#### Implementation Compatibility
1. **Discretization Consistency**: Ensure discrete formulation matches continuous theory
2. **Approximation Properties**: Verify neural network approximation capabilities
3. **Numerical Stability**: Check numerical scheme stability
4. **Convergence Guarantees**: Validate theoretical convergence translates to implementation

## Validation Output Structure

### Mathematical Formulation Document
```markdown
# Mathematical Foundations: [Component]

## Theoretical Formulation
- **Equation**: [Extracted mathematical equation]
- **Parameters**: [Mathematical parameters and ranges]
- **Assumptions**: [Theoretical assumptions]
- **Domain**: [Mathematical domain and boundary conditions]

## Implementation Requirements
- **Discretization**: [Required discretization approach]
- **Numerical Scheme**: [Recommended numerical methods]
- **Stability Conditions**: [Numerical stability requirements]
- **Convergence Criteria**: [Expected convergence behavior]

## Validation Checklist
- [ ] Equation extraction verified against paper
- [ ] Implementation parameters match theory
- [ ] Numerical scheme maintains theoretical properties
- [ ] User verification of mathematical interpretation
```

### Theory-Implementation Mapping
```python
# Mathematical Theory → Implementation Map
theory_implementation_map = {
    "continuous_equation": "discretized_pde_residual()",
    "galerkin_projection": "compute_weak_form_residual()",
    "sparse_sampling": "random_subset_selection()",
    "neural_approximation": "NeuralNetwork.forward()",
    "time_evolution": "runge_kutta_step()"
}
```

## Integration with TDD Workflow

### Pre-Implementation Validation
Before writing tests:
1. Extract theoretical formulation for component
2. Validate mathematical assumptions and parameters
3. Create theory-to-test mapping
4. Document expected numerical behavior

### During Implementation
1. Cross-reference implementation against extracted theory
2. Validate numerical parameters match theoretical bounds
3. Ensure approximation properties are maintained
4. Check implementation preserves theoretical guarantees

### Post-Implementation Verification
1. Verify numerical results match theoretical predictions
2. Validate convergence rates against theory
3. Check implementation assumptions against paper
4. Document any theoretical deviations or approximations

## Quality Assurance

### Mathematical Validation Requirements
- [ ] All key equations extracted and verified
- [ ] Implementation parameters validated against theory
- [ ] Mathematical assumptions documented and checked
- [ ] Theoretical guarantees preserved in implementation
- [ ] User verification of theoretical interpretation

### Theory-Implementation Consistency
- [ ] Code structure reflects mathematical formulation
- [ ] Variable names match mathematical notation
- [ ] Numerical methods preserve theoretical properties
- [ ] Implementation assumptions clearly documented
- [ ] Deviations from theory justified and documented

## Success Criteria

### Complete Theoretical Foundation
- [ ] All relevant mathematical formulations extracted
- [ ] Theory-implementation mapping documented
- [ ] Mathematical assumptions and parameters validated
- [ ] Expected numerical behavior characterized
- [ ] Implementation guidance provided for TDD process

### Ready for Algorithm Implementation
- [ ] Clear mathematical requirements established
- [ ] Numerical parameters and tolerances defined
- [ ] Test cases derived from theoretical expectations
- [ ] Implementation approach validated against theory
- [ ] User approval of theoretical interpretation

Remember: **Theory validation enables confident TDD implementation**. Understanding the mathematical foundations ensures tests capture the right properties and implementations preserve theoretical guarantees.