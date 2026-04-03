# Three-Body Problem: Lie Algebra of Pairwise Dynamics

## Project Goal

We are exploring whether the **pairwise interaction Hamiltonians** of a three-body gravitational system form a finite-dimensional Lie algebra under Poisson brackets. This is a potentially novel approach to the three-body problem — instead of studying the symmetry algebra of the space the system lives in (the standard approach), we're studying the **Lie algebra of the dynamics themselves**.

## Origin and Motivation

This emerged from an extended conversation about the three-body problem, circular causality, and the intuition that complex system dynamics are fundamentally **oscillatory and mutually causal**. The key insight:

- In a three-body system, the fundamental dynamical units aren't the three bodies — they're the **three pairwise interactions**.
- Each pairwise interaction, in isolation, is a solved two-body problem (an oscillation).
- The three-body problem arises because these pairwise interactions **don't commute** — the order in which they act matters.
- The **Poisson bracket** of two pairwise Hamiltonians measures exactly how much they interfere with each other.
- If the resulting algebra closes into a finite-dimensional Lie algebra, it would reveal hidden structure in the three-body problem.

## Mathematical Setup

### Phase Space (Planar 3-Body)

We work in 2D to keep things tractable. The phase space is 12-dimensional:

- **Positions:** (x₁, y₁), (x₂, y₂), (x₃, y₃)
- **Momenta:** (px₁, py₁), (px₂, py₂), (px₃, py₃)
- **Masses:** m₁, m₂, m₃
- **Gravitational constant:** G

### Pairwise Hamiltonians

Each pairwise Hamiltonian is the full two-body Hamiltonian for that pair (as if the third body didn't exist):

```
H₁₂ = px₁²/(2m₁) + py₁²/(2m₁) + px₂²/(2m₂) + py₂²/(2m₂) - G·m₁·m₂/r₁₂
H₁₃ = px₁²/(2m₁) + py₁²/(2m₁) + px₃²/(2m₃) + py₃²/(2m₃) - G·m₁·m₃/r₁₃
H₂₃ = px₂²/(2m₂) + py₂²/(2m₂) + px₃²/(2m₃) + py₃²/(2m₃) - G·m₂·m₃/r₂₃
```

Where `r₁₂ = sqrt((x₁-x₂)² + (y₁-y₂)²)`, etc.

Note: The total Hamiltonian H = H₁₂ + H₁₃ + H₂₃ overcounts kinetic energy, but that's fine — we're studying the algebraic structure of these generators, not trying to recover the total H.

### Poisson Bracket

```
{f, g} = Σᵢ (∂f/∂qᵢ · ∂g/∂pᵢ - ∂f/∂pᵢ · ∂g/∂qᵢ)
```

### Key Analytical Result (Derived in Conversation)

We worked out that the bracket {H₁₂, H₁₃} simplifies beautifully:

```
{H₁₂, H₁₃} = (1/m₁) · [(F₁₃ - F₁₂) · p₁]
```

Where:
- F₁₂ = gravitational force on body 1 from body 2
- F₁₃ = gravitational force on body 1 from body 3
- p₁ = momentum of body 1

**Physical interpretation:** The bracket measures the **tidal competition** at the shared body — the difference of forces from the two interactions, projected along the shared body's velocity. It quantifies how much the two interactions compete for body 1's trajectory.

### Generator Definitions

**Level 0 (original generators):**
- H₁₂, H₁₃, H₂₃ — pairwise Hamiltonians (degree 2 in momenta)

**Level 1 (first brackets):**
- K₁ := {H₁₂, H₁₃} = (F₁₃ - F₁₂) · p₁/m₁ — tidal competition at body 1
- K₂ := {H₁₂, H₂₃} = (F₂₁ - F₂₃) · p₂/m₂ — tidal competition at body 2
- K₃ := {H₁₃, H₂₃} = (F₃₁ - F₃₂) · p₃/m₃ — tidal competition at body 3

Note the grading: H's are degree 2 in momenta, K's are degree 1 (the bracket reduces degree by 1).

### The Central Question

**Does the algebra close?** Specifically:

1. What are {Kᵢ, Hⱼₖ}? Can they be expressed in terms of H's and K's?
2. What are {Kᵢ, Kⱼ}? Can they be expressed in terms of H's and K's?
3. If new generators appear, do THOSE eventually close?

If the algebra closes at 6 generators (3 H's + 3 K's), it would be a 6-dimensional Lie algebra, possibly related to so(3,1), su(3), or sp(4).

### Speculative Connection to su(3)

There's a suggestive structural parallel:
- su(3) has 8 generators and naturally describes systems with three coupled degrees of freedom
- Its root structure has three pairs of raising/lowering operators connected through a Cartan subalgebra
- su(3) is the algebra of the strong force mediating interactions between three color charges
- Three quarks interacting through pairwise gluon exchange has the same **algebraic skeleton** as three bodies interacting through pairwise gravity — different physics, possibly the same algebraic structure

### Jacobi Identity Constraint

The Jacobi identity must hold and provides a relation between second-level brackets:

```
{H₁₂, K₃} - {H₁₃, K₂} + {H₂₃, K₁} = 0
```

This is both a consistency check and a constraint that reduces the number of independent second-level generators.

## Computational Task

### Step 1: Verify the First-Level Results

Using SymPy, compute {H₁₂, H₁₃}, {H₁₂, H₂₃}, {H₁₃, H₂₃} symbolically and confirm the "tidal competition" interpretation.

### Step 2: Compute Second-Level Brackets

Compute all brackets of K's with H's and K's with K's:

- {K₁, H₁₂}, {K₁, H₁₃}, {K₁, H₂₃}
- {K₂, H₁₂}, {K₂, H₁₃}, {K₂, H₂₃}
- {K₃, H₁₂}, {K₃, H₁₃}, {K₃, H₂₃}
- {K₁, K₂}, {K₁, K₃}, {K₂, K₃}

### Step 3: Analyze Closure

For each second-level bracket:
- What is its degree in momenta?
- Can it be expressed as a function (even nonlinear) of the existing generators?
- If not, define new generators and repeat.

### Step 4: Identify the Algebra

If it closes (or approximately closes), map the structure constants to known Lie algebras. If it doesn't close, characterize the growth rate — that itself encodes the nature of three-body chaos.

### Step 5: Numerical Verification

For specific mass values (e.g., equal masses m₁=m₂=m₃=1, G=1), evaluate the brackets at random phase space points and check the algebraic relations numerically. This catches symbolic simplification errors.

## Code

Here is the SymPy code to begin with. It sets up the full computation but may need optimization for the second-level brackets (which get symbolically heavy):

```python
"""
Three-Body Problem: Lie Algebra of Pairwise Interactions
=========================================================
"""

import sympy as sp
from sympy import symbols, sqrt, simplify, expand

# Phase space variables
x1, y1, x2, y2, x3, y3 = symbols('x1 y1 x2 y2 x3 y3', real=True)
px1, py1, px2, py2, px3, py3 = symbols('px1 py1 px2 py2 px3 py3', real=True)
m1, m2, m3 = symbols('m1 m2 m3', positive=True)
G = symbols('G', positive=True)

q_vars = [x1, y1, x2, y2, x3, y3]
p_vars = [px1, py1, px2, py2, px3, py3]

def poisson_bracket(f, g):
    """Compute {f, g} = sum_i (df/dq_i * dg/dp_i - df/dp_i * dg/dq_i)"""
    result = 0
    for qi, pi in zip(q_vars, p_vars):
        result += sp.diff(f, qi) * sp.diff(g, pi) - sp.diff(f, pi) * sp.diff(g, qi)
    return result

# Pairwise distances
r12 = sqrt((x1-x2)**2 + (y1-y2)**2)
r13 = sqrt((x1-x3)**2 + (y1-y3)**2)
r23 = sqrt((x2-x3)**2 + (y2-y3)**2)

# Potentials
V12 = -G * m1 * m2 / r12
V13 = -G * m1 * m3 / r13
V23 = -G * m2 * m3 / r23

# Kinetic energies
T1 = (px1**2 + py1**2) / (2*m1)
T2 = (px2**2 + py2**2) / (2*m2)
T3 = (px3**2 + py3**2) / (2*m3)

# Pairwise Hamiltonians
H12 = T1 + T2 + V12
H13 = T1 + T3 + V13
H23 = T2 + T3 + V23

# ---- LEVEL 1 BRACKETS ----
print("Computing level 1 brackets...")
K1 = simplify(poisson_bracket(H12, H13))  # mediated through body 1
K2 = simplify(poisson_bracket(H12, H23))  # mediated through body 2
K3 = simplify(poisson_bracket(H13, H23))  # mediated through body 3

print(f"K1 = {{H12, H13}} = {K1}")
print(f"K2 = {{H12, H23}} = {K2}")
print(f"K3 = {{H13, H23}} = {K3}")

# ---- VERIFY TIDAL INTERPRETATION ----
F12_x = -sp.diff(V12, x1)  # Force on body 1 from body 2
F12_y = -sp.diff(V12, y1)
F13_x = -sp.diff(V13, x1)  # Force on body 1 from body 3
F13_y = -sp.diff(V13, y1)

tidal_formula = (1/m1) * ((F13_x - F12_x)*px1 + (F13_y - F12_y)*py1)
print(f"\nTidal formula matches K1: {simplify(K1 - tidal_formula) == 0}")

# ---- LEVEL 2 BRACKETS ----
# WARNING: These are symbolically heavy. Consider:
# 1. Using cse() (common subexpression elimination) 
# 2. Substituting specific mass values first
# 3. Working with equal masses m1=m2=m3=1, G=1 as a first pass

print("\nComputing level 2 brackets (this may take a while)...")

# Bracket K's with H's
for name_k, k_gen in [("K1", K1), ("K2", K2), ("K3", K3)]:
    for name_h, h_gen in [("H12", H12), ("H13", H13), ("H23", H23)]:
        result = simplify(poisson_bracket(k_gen, h_gen))
        print(f"{{{name_k}, {name_h}}} = {result}")

# Bracket K's with K's
for (name_a, a), (name_b, b) in [
    (("K1", K1), ("K2", K2)),
    (("K1", K1), ("K3", K3)),
    (("K2", K2), ("K3", K3))
]:
    result = simplify(poisson_bracket(a, b))
    print(f"{{{name_a}, {name_b}}} = {result}")

# ---- JACOBI IDENTITY CHECK ----
print("\nJacobi identity check...")
jacobi = simplify(
    poisson_bracket(H12, K3) - poisson_bracket(H13, K2) + poisson_bracket(H23, K1)
)
print(f"Should be 0: {jacobi}")

# ---- NUMERICAL SPOT CHECK ----
import random
print("\nNumerical verification at random phase space point...")
subs_dict = {
    m1: 1, m2: 1, m3: 1, G: 1,
    x1: random.uniform(-2,2), y1: random.uniform(-2,2),
    x2: random.uniform(-2,2), y2: random.uniform(-2,2),
    x3: random.uniform(-2,2), y3: random.uniform(-2,2),
    px1: random.uniform(-1,1), py1: random.uniform(-1,1),
    px2: random.uniform(-1,1), py2: random.uniform(-1,1),
    px3: random.uniform(-1,1), py3: random.uniform(-1,1),
}
K1_num = float(K1.subs(subs_dict))
tidal_num = float(tidal_formula.subs(subs_dict))
print(f"K1 = {K1_num:.6f}, Tidal formula = {tidal_num:.6f}, Match: {abs(K1_num - tidal_num) < 1e-10}")
```

## Performance Notes

The second-level brackets involve differentiating expressions with nested square roots and inverse powers, which can cause SymPy to bog down. Strategies:

1. **Equal mass simplification first:** Set m₁=m₂=m₃=1, G=1 to reduce symbolic complexity. If the algebra closes in this case, then study the general mass case.
2. **Common subexpression elimination:** `sp.cse()` can dramatically speed up evaluation.
3. **Numerical approach:** Compute brackets numerically at many random phase space points, then fit to determine if they're linear combinations of known generators.
4. **Use r² instead of r:** Work with `r₁₂² = (x₁-x₂)² + (y₁-y₂)²` to avoid square roots where possible. The potential becomes `V₁₂ = -G·m₁·m₂ · (r₁₂²)^(-1/2)`.

## Broader Context

If this works, the implications connect to several deep ideas:

- **Oscillatory causation:** The original intuition — complex systems are fundamentally oscillatory and mutually causal — would be formalized as "the dynamics live in a Lie algebra whose generators are pairwise oscillatory interactions."
- **Analog computing:** If the algebra has finite structure, it might be implementable in analog hardware (like magnetic domain dynamics in bubble memory) where the physical substrate naturally performs the coupled oscillation.
- **Static universe / timeless graph:** The weight diagram of the resulting Lie algebra IS a timeless graph whose nodes are states and edges are generators. The dynamics is traversal, not evolution.
- **Universality:** If the three-body gravitational algebra is isomorphic to su(3) (or related), it suggests a deep structural reason why three interacting entities produce the same algebraic patterns across physics — from quarks to stars to human relationships.

## References to Explore

- Poisson-Lie groups and Lie-Poisson structures
- Dynamical algebras / spectrum-generating algebras
- Arnol'd's work on integrability and Lie algebra rank
- Causal set theory and relational quantum mechanics
- Wiener's cybernetics and circular causality
- Maturana & Varela's autopoiesis
