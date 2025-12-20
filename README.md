# Stochastic Control and Hamilton-Jacobi-Bellman Equations: Theory and Numerical Methods

A rigorous, research-level implementation of stochastic optimal control theory using the dynamic programming principle and Hamilton-Jacobi-Bellman (HJB) equations. This project provides end-to-end numerical methods for solving optimal control problems of stochastic differential equations, from theoretical derivation to simulation-based verification.

## Author

**Divyansh Atri**

## Project Scope

This project implements numerical methods for solving stochastic control problems governed by controlled stochastic differential equations (SDEs):

```
dX_t = b(X_t, u_t) dt + σ(X_t) dW_t
```

where `X_t` is the state process, `u_t` is the control, and `W_t` is standard Brownian motion.

The objective is to minimize the expected cost functional:

```
J(u) = E[∫₀ᵀ L(X_t, u_t) dt + g(X_T)]
```

by computing the value function:

```
V(t,x) = inf_u E_{t,x}[J(u)]
```

and extracting the optimal feedback control policy `u*(t,x)`.

### Core Components

1. **Theoretical Foundation**: Rigorous derivation of the Hamilton-Jacobi-Bellman equation from the dynamic programming principle
2. **Numerical Methods**: Finite difference schemes for solving the HJB PDE with Hamiltonian minimization
3. **Benchmark Problems**: Linear-Quadratic Regulator (LQR), controlled Brownian motion, and mean-reverting processes
4. **Verification**: Monte Carlo simulation with closed-loop feedback control to validate numerical solutions
5. **Analysis**: Convergence studies, stability analysis, and investigation of the curse of dimensionality

## Mathematical Framework

### Dynamic Programming Principle

For a controlled diffusion process, the value function satisfies:

```
V(t,x) = inf_u E[∫_t^{t+h} L(X_s, u_s) ds + V(t+h, X_{t+h}) | X_t = x]
```

### Hamilton-Jacobi-Bellman Equation

Taking the limit as `h → 0` yields the HJB PDE:

```
∂V/∂t + min_u {L(x,u) + b(x,u)·∇V + (1/2)σ²(x)ΔV} = 0
```

with terminal condition `V(T,x) = g(x)`.

The optimal control is obtained by minimizing the Hamiltonian:

```
u*(t,x) = argmin_u {L(x,u) + b(x,u)·∇V(t,x)}
```

### Viscosity Solutions

The HJB equation is a nonlinear PDE that may not have classical smooth solutions. The appropriate solution concept is that of viscosity solutions, which:
- Always exist under mild conditions
- Are unique
- Coincide with the value function from stochastic control
- Can be approximated by numerical schemes

## Project Structure

```
stochastic-control-hjb/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── notebooks/                         # Jupyter notebooks (execute in order)
│   ├── 01_dynamic_programming_and_control.ipynb
│   ├── 02_hjb_derivation.ipynb
│   ├── 03_numerical_hjb_solver.ipynb
│   ├── 04_lqr_validation.ipynb
│   ├── 05_policy_iteration.ipynb
│   ├── 06_controlled_sde_simulation.ipynb
│   └── 07_summary_and_limitations.ipynb
├── utils/                             # Core implementation modules
│   ├── __init__.py
│   ├── sde_models.py                 # SDE model definitions
│   ├── cost_functions.py             # Cost functionals and analytical solutions
│   ├── hjb_solvers.py                # HJB PDE solvers
│   └── simulation.py                 # Monte Carlo simulation tools
├── tests/                            # Validation and sanity checks
│   └── sanity_checks.ipynb
└── plots/                            # Generated figures
```

## Notebook Execution Order

The notebooks must be executed in the following sequence:

### 1. Dynamic Programming and Control (`01_dynamic_programming_and_control.ipynb`)
- Introduction to stochastic control problems
- Formulation of controlled SDEs
- Statement of the dynamic programming principle
- Intuition for value functions and optimal policies

### 2. HJB Derivation (`02_hjb_derivation.ipynb`)
- Rigorous derivation of the HJB equation from the DPP
- Infinitesimal generator and Itô's formula
- Interpretation as a nonlinear PDE
- Hamiltonian formulation and optimality conditions
- Connection to Bellman optimality

### 3. Numerical HJB Solver (`03_numerical_hjb_solver.ipynb`)
- Finite difference discretization in space and time
- Backward time stepping algorithm
- Pointwise Hamiltonian minimization
- Explicit vs implicit time integration
- Stability analysis and CFL conditions

### 4. LQR Validation (`04_lqr_validation.ipynb`)
- Linear-Quadratic Regulator problem setup
- Analytical solution via Riccati equation
- Numerical solution using HJB solver
- Comparison and error analysis
- Verification of convergence rates

### 5. Policy Iteration (`05_policy_iteration.ipynb`)
- Policy iteration algorithm for HJB equations
- Policy evaluation and policy improvement steps
- Comparison with value iteration
- Convergence analysis
- Computational efficiency

### 6. Controlled SDE Simulation (`06_controlled_sde_simulation.ipynb`)
- Euler-Maruyama scheme for controlled SDEs
- Closed-loop simulation with feedback control
- Monte Carlo verification of value function
- Empirical cost computation
- Trajectory visualization and analysis

### 7. Summary and Limitations (`07_summary_and_limitations.ipynb`)
- Summary of key results
- Comparison across different problems
- Discussion of numerical accuracy
- Curse of dimensionality
- Extensions to higher dimensions
- Limitations and future work

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

```bash
# Clone or navigate to project directory
cd stochastic-control-hjb

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

### Running the Notebooks

Execute notebooks in numerical order (01 through 07). Each notebook is self-contained but builds on concepts from previous notebooks.

```bash
# From the project root
jupyter notebook notebooks/01_dynamic_programming_and_control.ipynb
```

All notebooks support "Restart & Run All" functionality.

## Benchmark Problems

### 1. Linear-Quadratic Regulator (LQR)

**Dynamics:**
```
dX_t = (A X_t + B u_t) dt + σ dW_t
```

**Cost:**
```
J = E[∫₀ᵀ (1/2)(q X_t² + r u_t²) dt + (1/2) q_T X_T²]
```

**Analytical Solution:** The value function is `V(t,x) = (1/2) P(t) x² + ψ(t)` where `P(t)` solves the Riccati ODE.

**Purpose:** Validation of numerical methods against known analytical solution.

### 2. Controlled Brownian Motion

**Dynamics:**
```
dX_t = u_t dt + σ dW_t
```

**Cost:**
```
J = E[∫₀ᵀ (1/2)(q X_t² + r u_t²) dt + (1/2) q_T X_T²]
```

**Purpose:** Demonstrates direct control of drift with intuitive optimal policy.

### 3. Mean-Reverting Process with Control

**Dynamics:**
```
dX_t = (θ(μ - X_t) + u_t) dt + σ dW_t
```

**Cost:**
```
J = E[∫₀ᵀ (1/2)(q X_t² + r u_t²) dt + (1/2) q_T X_T²]
```

**Purpose:** Illustrates tradeoff between natural mean reversion and control intervention.

## Numerical Methods

### Finite Difference Discretization

**Spatial Grid:** Uniform grid on `[x_min, x_max]` with spacing `Δx`

**Time Grid:** Uniform grid on `[0, T]` with spacing `Δt`

**Derivatives:**
- First derivative: Central difference `V_x ≈ (V_{i+1} - V_{i-1})/(2Δx)`
- Second derivative: Central difference `V_xx ≈ (V_{i+1} - 2V_i + V_{i-1})/(Δx²)`

### Backward Time Stepping

Starting from terminal condition `V(T, x) = g(x)`, solve backward in time:

```
V^n = V^{n+1} - Δt · min_u {L(x,u) + b(x,u)V_x^{n+1} + (1/2)σ²V_xx^{n+1}}
```

### Hamiltonian Minimization

At each grid point `(t_n, x_i)`, solve:

```
u*_{n,i} = argmin_u {L(x_i, u) + b(x_i, u)V_x^{n+1}_i + (1/2)σ²(x_i)V_xx^{n+1}_i}
```

using numerical optimization (scipy.optimize).

### Stability Considerations

**CFL Condition:** For explicit schemes, stability requires:
```
Δt ≤ C · (Δx)² / σ²_max
```

**Implicit Schemes:** More stable but require iterative solution at each time step.

## Key Results

### Convergence Analysis

Numerical experiments demonstrate:
- **Spatial convergence:** `O(Δx²)` for smooth solutions
- **Temporal convergence:** `O(Δt)` for explicit Euler, `O(Δt²)` for implicit schemes
- **Policy convergence:** Optimal control converges to analytical solution for LQR

### LQR Validation

For the LQR problem with parameters `A=-1, B=1, σ=0.5, q=1, r=1`:
- Numerical and analytical solutions agree to within `10⁻⁴` relative error
- Convergence rate matches theoretical predictions
- Optimal control policy is linear in state: `u*(t,x) = -K(t) x`

### Curse of Dimensionality

For `d`-dimensional problems:
- Grid size scales as `O(N^d)` where `N` is points per dimension
- Computational cost scales as `O(N^d · M)` where `M` is time steps
- Practical limit: `d ≤ 3` for standard finite difference methods
- Higher dimensions require alternative methods (e.g., deep learning, sparse grids)

## Mathematical Assumptions

1. **Regularity:** Drift `b(x,u)` and diffusion `σ(x)` are Lipschitz continuous
2. **Growth:** Linear growth conditions on `b`, `σ`, and cost functions
3. **Coercivity:** Running cost `L(x,u)` grows sufficiently fast in `u` to ensure bounded controls
4. **Non-degeneracy:** `σ(x) > 0` for all `x` (ensures ellipticity of HJB equation)
5. **Boundedness:** Control set is compact or cost is coercive in control

These assumptions guarantee:
- Existence and uniqueness of SDE solutions
- Existence of viscosity solutions to HJB equation
- Convergence of numerical schemes

## Limitations and Extensions

### Current Limitations

1. **Dimension:** Implementation restricted to 1D state space
2. **Control Constraints:** Unbounded controls (box constraints only)
3. **Regularity:** Assumes sufficient smoothness for finite differences
4. **Computation:** Explicit schemes have restrictive stability conditions

### Possible Extensions

1. **Higher Dimensions:** Extend to 2D/3D using tensor grids or sparse grids
2. **State Constraints:** Implement obstacle problems and reflected diffusions
3. **Jump Processes:** Extend to jump-diffusions and regime-switching models
4. **Advanced Numerics:** Implement semi-Lagrangian schemes, monotone schemes
5. **Machine Learning:** Neural network approximations for high-dimensional problems
6. **Applications:** Portfolio optimization, optimal stopping, stochastic games

## Computational Complexity

### Time Complexity

- **HJB Solver:** `O(N_x · N_t · N_opt)` where `N_opt` is cost of optimization per point
- **Policy Iteration:** `O(K · N_x · N_t)` where `K` is number of iterations
- **Simulation:** `O(N_paths · N_t)` for Monte Carlo verification

### Space Complexity

- **Value Function Storage:** `O(N_x · N_t)`
- **Policy Storage:** `O(N_x · N_t)`

### Typical Parameters

For research-quality results:
- Spatial grid: `N_x = 200-500` points
- Time grid: `N_t = 100-500` steps
- Monte Carlo: `N_paths = 1000-10000` paths

## References

### Theoretical Foundations

1. Fleming, W. H., & Soner, H. M. (2006). *Controlled Markov Processes and Viscosity Solutions*. Springer.
2. Yong, J., & Zhou, X. Y. (1999). *Stochastic Controls: Hamiltonian Systems and HJB Equations*. Springer.
3. Pham, H. (2009). *Continuous-time Stochastic Control and Optimization with Financial Applications*. Springer.

### Numerical Methods

4. Kushner, H. J., & Dupuis, P. (2001). *Numerical Methods for Stochastic Control Problems in Continuous Time*. Springer.
5. Forsyth, P. A., & Labahn, G. (2007). Numerical methods for controlled Hamilton-Jacobi-Bellman PDEs in finance. *Journal of Computational Finance*, 11(2), 1-43.

### Viscosity Solutions

6. Crandall, M. G., Ishii, H., & Lions, P. L. (1992). User's guide to viscosity solutions of second order partial differential equations. *Bulletin of the American Mathematical Society*, 27(1), 1-67.

## Testing and Validation

### Sanity Checks

The `tests/sanity_checks.ipynb` notebook verifies:
- HJB residual is small for computed solutions
- Optimal control satisfies first-order optimality conditions
- Simulated costs match value function predictions
- Convergence rates match theoretical expectations

### Verification Strategy

1. **Analytical Comparison:** LQR solution vs Riccati equation
2. **Residual Analysis:** HJB equation residual magnitude
3. **Monte Carlo Verification:** Empirical cost vs value function
4. **Convergence Tests:** Grid refinement studies
5. **Stability Tests:** Time step variation

## License

This project is released for academic and research purposes.

## Citation

If you use this code in your research, please cite:

```
Atri, D. (2025). Stochastic Control and Hamilton-Jacobi-Bellman Equations: 
Theory and Numerical Methods. Research Implementation.
```

---

**Note:** This is a research-level implementation intended for educational and academic purposes. The code prioritizes clarity and correctness over computational efficiency. For production applications, consider specialized libraries and optimized implementations.
