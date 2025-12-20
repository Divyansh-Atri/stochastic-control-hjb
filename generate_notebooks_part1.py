#!/usr/bin/env python3
"""
Generate all remaining Jupyter notebooks for the Stochastic Control project
This script creates notebooks 02-07 with complete content
"""

import json
import os

def create_notebook(cells):
    """Helper to create notebook structure"""
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

def markdown_cell(content):
    return {"cell_type": "markdown", "metadata": {}, "source": content}

def code_cell(content):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": content}

# Base directory
base_dir = "/home/divyansh/project/jhb/stochastic-control-hjb"
nb_dir = f"{base_dir}/notebooks"
os.makedirs(f"{base_dir}/plots", exist_ok=True)

print("Generating Jupyter notebooks...")
print("=" * 70)

# ============================================================================
# NOTEBOOK 02: HJB Derivation
# ============================================================================
nb02_cells = [
    markdown_cell(["# Notebook 2: Derivation of the Hamilton-Jacobi-Bellman Equation\\n\\n",
                   "**Author:** Divyansh Atri\\n\\n",
                   "## Overview\\n\\n",
                   "This notebook rigorously derives the HJB equation from the dynamic programming principle.\\n\\n",
                   "**Topics:**\\n",
                   "1. Itô's formula and infinitesimal generator\\n",
                   "2. Formal derivation of HJB equation\\n",
                   "3. Hamiltonian formulation\\n",
                   "4. Optimal control extraction\\n",
                   "5. Viscosity solutions"]),
    
    code_cell(["import numpy as np\\n",
               "import matplotlib.pyplot as plt\\n",
               "import sys\\n",
               "sys.path.append('..')\\n\\n",
               "plt.style.use('seaborn-v0_8-darkgrid')\\n",
               "plt.rcParams['figure.figsize'] = (12, 6)\\n\\n",
               "print('HJB Derivation - Ready')"]),
    
    markdown_cell(["## 1. Dynamic Programming Principle\\n\\n",
                   "The value function satisfies:\\n\\n",
                   "$$V(t, x) = \\inf_{u} \\mathbb{E}_{t,x}\\left[ \\int_t^{t+h} L(X_s, u_s) ds + V(t+h, X_{t+h}) \\right]$$\\n\\n",
                   "**Goal:** Take limit $h \\to 0$ to derive a PDE."]),
    
    markdown_cell(["## 2. Itô's Formula\\n\\n",
                   "For $dX_t = b(X_t, u_t) dt + \\sigma(X_t) dW_t$:\\n\\n",
                   "$$df(t, X_t) = \\left(f_t + b f_x + \\frac{1}{2}\\sigma^2 f_{xx}\\right) dt + \\sigma f_x dW_t$$\\n\\n",
                   "The **infinitesimal generator** is:\\n\\n",
                   "$$\\mathcal{L}^u f = f_t + b(x,u) f_x + \\frac{1}{2}\\sigma^2(x) f_{xx}$$"]),
    
    markdown_cell(["## 3. HJB Equation Derivation\\n\\n",
                   "Applying Itô's formula to $V(t+h, X_{t+h})$ and taking $h \\to 0$:\\n\\n",
                   "$$\\boxed{V_t + \\min_u \\left\\{ L(x, u) + b(x, u) V_x + \\frac{1}{2} \\sigma^2(x) V_{xx} \\right\\} = 0}$$\\n\\n",
                   "with terminal condition $V(T, x) = g(x)$."]),
    
    code_cell(["# Visualize Hamiltonian\\n",
               "def hamiltonian(u, x, V_x, A, B, q, r):\\n",
               "    return 0.5 * (q * x**2 + r * u**2) + (A * x + B * u) * V_x\\n\\n",
               "A, B, q, r = -1.0, 1.0, 1.0, 1.0\\n",
               "x, V_x = 1.0, 2.0\\n\\n",
               "u_vals = np.linspace(-5, 5, 200)\\n",
               "H_vals = hamiltonian(u_vals, x, V_x, A, B, q, r)\\n",
               "u_opt = -B * V_x / r\\n\\n",
               "plt.figure(figsize=(10, 6))\\n",
               "plt.plot(u_vals, H_vals, 'b-', linewidth=2, label='Hamiltonian')\\n",
               "plt.plot(u_opt, hamiltonian(u_opt, x, V_x, A, B, q, r), 'ro', markersize=10, label=f'Optimal $u^*={u_opt:.2f}$')\\n",
               "plt.xlabel('Control $u$')\\n",
               "plt.ylabel('Hamiltonian')\\n",
               "plt.title('Hamiltonian Minimization')\\n",
               "plt.legend()\\n",
               "plt.grid(True, alpha=0.3)\\n",
               "plt.tight_layout()\\n",
               "plt.show()\\n\\n",
               "print(f'Optimal control: u* = {u_opt:.4f}')"]),
    
    markdown_cell(["## 4. Optimal Control\\n\\n",
                   "The optimal control minimizes the Hamiltonian:\\n\\n",
                   "$$u^*(t, x) = \\arg\\min_u \\left\\{ L(x, u) + b(x, u) V_x(t, x) \\right\\}$$\\n\\n",
                   "For quadratic cost: $u^* = -\\frac{B}{r} V_x$ (linear feedback)"]),
    
    markdown_cell(["## 5. Viscosity Solutions\\n\\n",
                   "The HJB equation may not have classical smooth solutions.\\n\\n",
                   "**Viscosity solutions:**\\n",
                   "- Always exist under mild conditions\\n",
                   "- Are unique\\n",
                   "- Coincide with the value function\\n",
                   "- Can be approximated numerically\\n\\n",
                   "This provides the theoretical foundation for our numerical methods."]),
    
    markdown_cell(["## Summary\\n\\n",
                   "We derived the HJB equation from the dynamic programming principle using Itô's formula.\\n\\n",
                   "**Key results:**\\n",
                   "- HJB PDE: $V_t + \\min_u \\{L + bV_x + \\frac{1}{2}\\sigma^2 V_{xx}\\} = 0$\\n",
                   "- Optimal control: $u^* = \\arg\\min_u \\{L + bV_x\\}$\\n",
                   "- Viscosity solutions provide existence and uniqueness\\n\\n",
                   "**Next:** Numerical methods to solve the HJB equation."])
]

nb02 = create_notebook(nb02_cells)
with open(f"{nb_dir}/02_hjb_derivation.ipynb", 'w') as f:
    json.dump(nb02, f, indent=1)
print("✓ Created 02_hjb_derivation.ipynb")

# ============================================================================
# NOTEBOOK 03: Numerical HJB Solver  
# ============================================================================
nb03_cells = [
    markdown_cell(["# Notebook 3: Numerical HJB Solver\\n\\n",
                   "**Author:** Divyansh Atri\\n\\n",
                   "## Overview\\n\\n",
                   "Implementation of finite difference methods for solving the HJB equation.\\n\\n",
                   "**Topics:**\\n",
                   "1. Finite difference discretization\\n",
                   "2. Backward time stepping\\n",
                   "3. Hamiltonian minimization\\n",
                   "4. Stability analysis\\n",
                   "5. Convergence studies"]),
    
    code_cell(["import numpy as np\\n",
               "import matplotlib.pyplot as plt\\n",
               "from mpl_toolkits.mplot3d import Axes3D\\n",
               "import sys\\n",
               "sys.path.append('..')\\n",
               "from utils import *\\n\\n",
               "plt.style.use('seaborn-v0_8-darkgrid')\\n",
               "plt.rcParams['figure.figsize'] = (14, 6)\\n\\n",
               "print('Numerical HJB Solver - Ready')"]),
    
    markdown_cell(["## 1. Finite Difference Discretization\\n\\n",
                   "### Spatial Grid\\n",
                   "$$x_i = x_{\\min} + i\\Delta x, \\quad i = 0, \\ldots, N_x-1$$\\n\\n",
                   "### Time Grid\\n",
                   "$$t_n = n\\Delta t, \\quad n = 0, \\ldots, N_t-1$$\\n\\n",
                   "### Derivatives\\n",
                   "- First: $V_x \\approx \\frac{V_{i+1} - V_{i-1}}{2\\Delta x}$\\n",
                   "- Second: $V_{xx} \\approx \\frac{V_{i+1} - 2V_i + V_{i-1}}{(\\Delta x)^2}$"]),
    
    code_cell(["# Setup: Controlled Brownian Motion\\n",
               "model = ControlledBrownianMotion(sigma=0.5)\\n",
               "cost_fn = QuadraticCost(q=1.0, r=1.0, q_terminal=10.0)\\n\\n",
               "# Grid\\n",
               "x_min, x_max, nx = -3.0, 3.0, 101\\n",
               "T, nt = 2.0, 201\\n\\n",
               "dx = (x_max - x_min) / (nx - 1)\\n",
               "dt = T / (nt - 1)\\n\\n",
               "print(f'Grid: x ∈ [{x_min}, {x_max}], {nx} points, dx = {dx:.4f}')\\n",
               "print(f'Time: t ∈ [0, {T}], {nt} points, dt = {dt:.4f}')\\n",
               "print(f'CFL number: {0.5 * model.sigma**2 * dt / dx**2:.4f}')"]),
    
    markdown_cell(["## 2. HJB Solver Implementation\\n\\n",
                   "Backward time stepping:\\n\\n",
                   "$$V^n_i = V^{n+1}_i - \\Delta t \\cdot \\min_u H(x_i, u, V_x^{n+1}_i, V_{xx}^{n+1}_i)$$"]),
    
    code_cell(["# Create and solve\\n",
               "solver = HJBSolver(x_min, x_max, nx, T, nt, model, cost_fn)\\n\\n",
               "print('Solving HJB equation...')\\n",
               "V, u_opt = solver.solve_backward(u_bounds=(-5, 5), verbose=True)\\n\\n",
               "print(f'\\nSolution shape: V {V.shape}, u_opt {u_opt.shape}')"]),
    
    markdown_cell(["## 3. Visualization"]),
    
    code_cell(["# Plot results\\n",
               "fig = plt.figure(figsize=(16, 5))\\n\\n",
               "# 3D surface\\n",
               "ax1 = fig.add_subplot(131, projection='3d')\\n",
               "T_grid, X_grid = np.meshgrid(solver.t, solver.x, indexing='ij')\\n",
               "surf = ax1.plot_surface(T_grid, X_grid, V, cmap='viridis', alpha=0.9)\\n",
               "ax1.set_xlabel('Time $t$')\\n",
               "ax1.set_ylabel('State $x$')\\n",
               "ax1.set_zlabel('Value $V(t,x)$')\\n",
               "ax1.set_title('Value Function')\\n\\n",
               "# Value at different times\\n",
               "ax2 = fig.add_subplot(132)\\n",
               "times = [0, nt//4, nt//2, 3*nt//4, nt-1]\\n",
               "for idx in times:\\n",
               "    ax2.plot(solver.x, V[idx, :], label=f't={solver.t[idx]:.2f}', linewidth=2)\\n",
               "ax2.set_xlabel('State $x$')\\n",
               "ax2.set_ylabel('Value $V(t,x)$')\\n",
               "ax2.set_title('Value Function Slices')\\n",
               "ax2.legend()\\n",
               "ax2.grid(True, alpha=0.3)\\n\\n",
               "# Optimal control\\n",
               "ax3 = fig.add_subplot(133)\\n",
               "for idx in times:\\n",
               "    ax3.plot(solver.x, u_opt[idx, :], label=f't={solver.t[idx]:.2f}', linewidth=2)\\n",
               "ax3.set_xlabel('State $x$')\\n",
               "ax3.set_ylabel('Control $u^*(t,x)$')\\n",
               "ax3.set_title('Optimal Control Policy')\\n",
               "ax3.legend()\\n",
               "ax3.grid(True, alpha=0.3)\\n",
               "ax3.axhline(0, color='k', linestyle='--', alpha=0.5)\\n\\n",
               "plt.tight_layout()\\n",
               "plt.savefig('../plots/hjb_solution.png', dpi=150, bbox_inches='tight')\\n",
               "plt.show()"]),
    
    markdown_cell(["## 4. Convergence Analysis"]),
    
    code_cell(["# Grid refinement study\\n",
               "print('Testing convergence with grid refinement...')\\n\\n",
               "nx_values = [51, 101, 201]\\n",
               "V_solutions = []\\n\\n",
               "for nx_test in nx_values:\\n",
               "    solver_test = HJBSolver(x_min, x_max, nx_test, T, nt, model, cost_fn)\\n",
               "    V_test, _ = solver_test.solve_backward(u_bounds=(-5, 5), verbose=False)\\n",
               "    V_solutions.append((nx_test, V_test, solver_test.x))\\n",
               "    print(f'  nx = {nx_test}: V(0, 0) = {V_test[0, nx_test//2]:.6f}')\\n\\n",
               "# Compute errors\\n",
               "errors = []\\n",
               "for i in range(len(V_solutions) - 1):\\n",
               "    nx1, V1, x1 = V_solutions[i]\\n",
               "    nx2, V2, x2 = V_solutions[i+1]\\n",
               "    # Interpolate V1 to V2's grid\\n",
               "    from scipy.interpolate import interp1d\\n",
               "    V1_interp = np.zeros_like(V2)\\n",
               "    for n in range(nt):\\n",
               "        f = interp1d(x1, V1[n, :], kind='cubic', fill_value='extrapolate')\\n",
               "        V1_interp[n, :] = f(x2)\\n",
               "    error = np.max(np.abs(V2 - V1_interp))\\n",
               "    errors.append(error)\\n",
               "    print(f'Error (nx={nx1} vs {nx2}): {error:.6e}')\\n\\n",
               "# Plot convergence\\n",
               "if len(errors) > 0:\\n",
               "    plt.figure(figsize=(8, 6))\\n",
               "    plt.semilogy(nx_values[1:], errors, 'bo-', linewidth=2, markersize=8)\\n",
               "    plt.xlabel('Number of spatial points $N_x$')\\n",
               "    plt.ylabel('Max error')\\n",
               "    plt.title('Convergence with Grid Refinement')\\n",
               "    plt.grid(True, alpha=0.3)\\n",
               "    plt.tight_layout()\\n",
               "    plt.savefig('../plots/hjb_convergence.png', dpi=150, bbox_inches='tight')\\n",
               "    plt.show()"]),
    
    markdown_cell(["## Summary\\n\\n",
                   "We implemented a finite difference solver for the HJB equation with:\\n",
                   "- Backward time stepping\\n",
                   "- Pointwise Hamiltonian minimization\\n",
                   "- Convergence analysis\\n\\n",
                   "**Next:** Validate against analytical LQR solution."])
]

nb03 = create_notebook(nb03_cells)
with open(f"{nb_dir}/03_numerical_hjb_solver.ipynb", 'w') as f:
    json.dump(nb03, f, indent=1)
print("✓ Created 03_numerical_hjb_solver.ipynb")

print("\\n" + "=" * 70)
print("Notebooks 02-03 created successfully!")
print("Run: jupyter notebook notebooks/")
print("=" * 70)
