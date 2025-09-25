from __future__ import annotations

"""
RSNG (Randomized Sparse Neural Galerkin) core mathematical formulation in SymPy.

This module encodes, symbolically, the equations extracted for the RSNG method:
1) PDE evolution equation ∂_t u = f(x, u) with IC.
2) Neural network ansatz u_hat(x; θ(t)).
3) Dirac–Frenkel/Galerkin least-squares residual minimization (continuous and discrete).
4) Random sparse sketch S_t and its element-wise definition via selector indices ξ_k(t).
5) Discrete-time (Euler) stepping with sparse updates.

It exposes a composed_equation dictionary that contains the labeled equations.
A pytest test suite validates structural/shape properties and relationships.

Strategy
- Represent fields and operators with SymPy Function, Eq, Derivative, and MatrixSymbol.
- Use Euclidean discrete least-squares objective r^T r to avoid introducing a custom L2 norm.
- Encode sketch selection by an element-wise Kronecker delta relation S_t[j, ℓ] = δ_{j, ξ_ℓ(t)}.
- Derive sketched residuals from the full residual via substitution θ̇ = S_t θ̇_s and verify equivalence.
- Provide discrete-time update equations with new sketch S_k at each step.

Notes
- Symbols n, p, s denote sample count, parameter dimension, and sketch size respectively.
- Unicode appears in printed names (e.g., θ), but Python identifiers remain ASCII for robustness.
- The variable composed_equation collects the final canonical symbolic content for downstream use.
"""

from sympy import (
    symbols,
    Function,
    Eq,
    Derivative,
    MatrixSymbol,
    IndexedBase,
    KroneckerDelta,
)

# -----------------------------------------------------------------------------
# 0. Common symbols and dimensions
# -----------------------------------------------------------------------------
# Time and space symbol (x denotes bold x collectively)
t = symbols('t', real=True)
x = symbols('x')  # stands in for bold x in the paper

# Dimensions: n (samples), p (parameters), s (sketch size)
n, p, s = symbols('n p s', integer=True, positive=True)

# -----------------------------------------------------------------------------
# 1. Evolution PDE and initial condition
# -----------------------------------------------------------------------------
u = Function('u')      # u(t, x)
f = Function('f')      # f(x, u)
u0 = Function('u0')    # u0(x)

pde_eq = Eq(Derivative(u(t, x), t), f(x, u(t, x)))
ic_eq = Eq(u(0, x), u0(x))

# -----------------------------------------------------------------------------
# 2. Neural network approximation (ansatz)
# -----------------------------------------------------------------------------
# u_hat(x; θ(t)) ≈ u(t, x) is represented symbolically by the expression below.
# θ(t) kept as a symbolic argument name for notational fidelity.
u_hat = Function('u_hat')
theta_t_sym = symbols('θ(t)')  # printed as θ(t)
ansatz_expr = u_hat(x, theta_t_sym)  # represents u_hat(x; θ(t))

# -----------------------------------------------------------------------------
# 3. Residual and least-squares (continuous and discrete forms)
# -----------------------------------------------------------------------------
# Continuous residual r(x; θ(t), θ̇(t)) = ∇_θ u_hat(x; θ(t)) · θ̇(t) - f(x, u_hat(x; θ(t)))
# Represent ∇_θ u_hat(x; θ(t)) by a 1×p symbolic row Jacobian Jx, and θ̇(t) as a p×1 vector.
Jx = MatrixSymbol('J(θ(t),x)', 1, p)         # row Jacobian at spatial point x
thetadot_full = MatrixSymbol('θ̇(t)', p, 1)  # full parameter time derivative
r_cont = Eq(Function('r')(x), (Jx * thetadot_full)[0, 0] - f(x, u_hat(x, theta_t_sym)))

# Discrete least-squares over sample points {x_i}_{i=1}^n:
# Minimize || J(θ(t)) θ̇(t) - f(θ(t)) ||_2^2
J = MatrixSymbol('J(θ(t))', n, p)            # stacked Jacobian at sampled points
f_vec = MatrixSymbol('f(θ(t))', n, 1)        # f at sampled points with u_hat substituted
r_vec = J * thetadot_full - f_vec            # discrete residual vector
ls_objective = (r_vec.T * r_vec)[0, 0]       # ||r||_2^2 as a scalar

# -----------------------------------------------------------------------------
# 4. Randomized Sparse Neural Galerkin (RSNG): sketching and sparse updates
# -----------------------------------------------------------------------------
# Sketch/selector matrix S_t ∈ R^{p×s} with columns drawn from canonical basis
S_t = MatrixSymbol('S_t', p, s)
# Element-wise definition via selector indices ξ_ℓ(t) ∈ {1, …, p}:
S = IndexedBase('S_t')
xi = Function('ξ')
# Use plain integer symbols for indices to ease substitution in tests
j, ell = symbols('j ℓ', integer=True)
S_element_def = Eq(S[j, ell], KroneckerDelta(j, xi(ell, t)))

# Reduced update θ̇_s(t) ∈ R^s and its lifting θ̇(t) = S_t θ̇_s(t)
thetadot_s = MatrixSymbol('θ̇_s(t)', s, 1)
thetadot_relation = Eq(thetadot_full, S_t * thetadot_s)

# Sketched Jacobian and sketched least-squares problem
J_s_expr = J * S_t                              # J_s(θ(t)) = J(θ(t)) S_t
r_s_vec = J_s_expr * thetadot_s - f_vec         # sketched residual
ls_objective_sketched = (r_s_vec.T * r_s_vec)[0, 0]

# -----------------------------------------------------------------------------
# 5. Discrete time-stepping (explicit Euler shown)
# -----------------------------------------------------------------------------
Δt = symbols('Δt', positive=True)               # time step
kk = symbols('k', integer=True, positive=True)  # time step index

# θ^(k) = θ^(k-1) + Δt Δθ^(k-1), with sparse Δθ^(k-1) = S_k Δθ_s^(k-1)
theta_k = MatrixSymbol('θ^(k)', p, 1)
theta_km1 = MatrixSymbol('θ^(k-1)', p, 1)
S_k = MatrixSymbol('S_k', p, s)
Δtheta_s_km1 = MatrixSymbol('Δθ_s^(k-1)', s, 1)
Δtheta_km1 = S_k * Δtheta_s_km1
update_eq = Eq(theta_k, theta_km1 + Δt * Δtheta_km1)

# LS system to determine Δθ_s^(k-1): minimize || J_s(θ^(k-1)) Δθ_s^(k-1) - f(θ^(k-1)) ||_2^2
J_km1 = MatrixSymbol('J(θ^(k-1))', n, p)
J_s_km1 = J_km1 * S_k
f_km1 = MatrixSymbol('f(θ^(k-1))', n, 1)
step_objective = (J_s_km1 * Δtheta_s_km1 - f_km1)
step_objective_scalar = (step_objective.T * step_objective)[0, 0]

# -----------------------------------------------------------------------------
# 6. Compose final canonical content for downstream consumption
# -----------------------------------------------------------------------------
composed_equation = {
    # Evolution PDE and IC
    'pde': pde_eq,
    'ic': ic_eq,
    # Ansatz
    'ansatz': ansatz_expr,  # u_hat(x; θ(t))
    # Residual and least-squares
    'residual_continuous': r_cont,
    'residual_discrete': r_vec,              # J(θ(t)) θ̇(t) - f(θ(t))
    'ls_objective': ls_objective,            # ||·||_2^2 in discrete form
    # Sketch definition and sketched least-squares
    'S_t_def_elementwise': S_element_def,    # S_t[j, ℓ] = δ_{j, ξ_ℓ(t)}
    'J_s': J_s_expr,                         # J_s(θ(t))
    'sketched_residual': r_s_vec,
    'sketched_objective': ls_objective_sketched,
    'thetadot_relation': thetadot_relation,  # θ̇(t) = S_t θ̇_s(t)
    # Discrete time stepping
    'time_update': update_eq,                # θ^(k) = θ^(k-1) + Δt S_k Δθ_s^(k-1)
    'step_objective': step_objective_scalar, # ||J_s Δθ_s - f||_2^2 at step k-1
}


# -----------------------------------------------------------------------------
#                                  Tests
# -----------------------------------------------------------------------------
# Pytest tests to validate structural properties and relationships

def test_pde_equation_structure():
    assert composed_equation['pde'].lhs == Derivative(u(t, x), t)
    assert composed_equation['pde'].rhs == f(x, u(t, x))
    assert composed_equation['ic'] == Eq(u(0, x), u0(x))


def test_discrete_residual_and_objective_shapes():
    J_local = J
    f_local = f_vec
    theta_dot_local = thetadot_full
    r_local = J_local * theta_dot_local - f_local
    assert r_local.shape == (n, 1)
    obj = (r_local.T * r_local)[0, 0]
    assert obj.free_symbols.issuperset({n, p})


def test_sketch_shapes_and_relations():
    # Shapes
    assert S_t.shape == (p, s)
    assert (J * S_t).shape == (n, s)
    assert thetadot_s.shape == (s, 1)
    # Sketched residual equals substitution-based form
    r_sub = (J * (S_t * thetadot_s) - f_vec)
    assert r_sub == r_s_vec
    # Theta-dot relation is consistent dimensionally
    assert thetadot_relation.lhs.shape == (p, 1)
    assert thetadot_relation.rhs.shape == (p, 1)


def test_S_elementwise_definition_behaviour():
    # Substitute a concrete selection ξ_ℓ(t) = 3 and j = 3 → Kronecker delta = 1
    S_jl = S_element_def.rhs.subs({j: 3, xi(ell, t): 3})
    assert S_jl.doit() == 1
    # For j = 2 and ξ_ℓ(t) = 3 → Kronecker delta = 0
    S_jl0 = S_element_def.rhs.subs({j: 2, xi(ell, t): 3})
    assert S_jl0.doit() == 0


def test_time_update_and_step_objective_shapes():
    assert composed_equation['time_update'].lhs.shape == (p, 1)
    assert composed_equation['time_update'].rhs.shape == (p, 1)
    # Step objective is scalar
    scalar = composed_equation['step_objective']
    assert scalar.shape == () or scalar.is_Number or scalar.is_Symbol

