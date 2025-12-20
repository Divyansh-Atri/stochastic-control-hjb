#!/usr/bin/env python3
"""
Quick Demo Script
Demonstrates the complete pipeline: Setup → Solve → Simulate → Verify
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add utils to path
sys.path.insert(0, os.path.dirname(__file__))

from utils import *

print("=" * 70)
print("STOCHASTIC CONTROL - QUICK DEMO")
print("=" * 70)

# ============================================================================
# 1. SETUP
# ============================================================================
print("\n1. SETUP")
print("-" * 70)

# Define problem
model = ControlledBrownianMotion(sigma=0.5)
cost_fn = QuadraticCost(q=1.0, r=1.0, q_terminal=10.0)

print(f"Model: Controlled Brownian Motion")
print(f"  dX_t = u_t dt + {model.sigma} dW_t")
print(f"\nCost: Quadratic")
print(f"  L(x,u) = 0.5({cost_fn.q}x² + {cost_fn.r}u²)")
print(f"  g(x) = 0.5({cost_fn.q_terminal}x²)")

# Grid parameters
x_min, x_max, nx = -3.0, 3.0, 101
T, nt = 2.0, 201

print(f"\nGrid:")
print(f"  Space: [{x_min}, {x_max}] with {nx} points")
print(f"  Time: [0, {T}] with {nt} points")

# ============================================================================
# 2. SOLVE HJB EQUATION
# ============================================================================
print("\n2. SOLVE HJB EQUATION")
print("-" * 70)

solver = HJBSolver(x_min, x_max, nx, T, nt, model, cost_fn)

print("Solving backward in time (using implicit scheme for stability)...")
V, u_opt = solver.solve_implicit(u_bounds=(-5, 5), verbose=False)

print(f"✓ Solution computed")
print(f"  Value at origin: V(0, 0) = {V[0, nx//2]:.6f}")
print(f"  Control at (0, 1): u*(0, 1) = {u_opt[0, nx//2 + 10]:.6f}")

# ============================================================================
# 3. MONTE CARLO SIMULATION
# ============================================================================
print("\n3. MONTE CARLO SIMULATION")
print("-" * 70)

# Define feedback policy
policy_fn = lambda t, x: solver.get_policy(t, x)

# Create simulator
simulator = ClosedLoopSimulator(model, cost_fn, policy_fn)

# Run simulation
x0 = 0.0
n_paths = 1000
dt_sim = 0.01

print(f"Simulating {n_paths} paths from x0 = {x0}...")
results = simulator.simulate(x0, T, dt_sim, n_paths=n_paths, seed=42)

print(f"✓ Simulation complete")
print(f"  Mean empirical cost: {results['mean_cost']:.6f}")
print(f"  Std deviation: {results['std_cost']:.6f}")

# ============================================================================
# 4. VERIFICATION
# ============================================================================
print("\n4. VERIFICATION")
print("-" * 70)

V_theoretical = V[0, nx//2]
V_empirical = results['mean_cost']
error = abs(V_empirical - V_theoretical)
rel_error = error / V_theoretical * 100

print(f"Value function V(0, 0): {V_theoretical:.6f}")
print(f"Empirical mean cost:    {V_empirical:.6f}")
print(f"Absolute error:         {error:.6f}")
print(f"Relative error:         {rel_error:.3f}%")

# Confidence interval
from scipy import stats
ci_95 = stats.t.interval(0.95, n_paths-1, 
                         loc=V_empirical, 
                         scale=results['std_cost']/np.sqrt(n_paths))

print(f"95% CI: [{ci_95[0]:.6f}, {ci_95[1]:.6f}]")

if ci_95[0] <= V_theoretical <= ci_95[1]:
    print("\n✓ VERIFICATION PASSED: Value function within 95% CI")
    verification_passed = True
else:
    print("\n✗ VERIFICATION FAILED: Value function outside 95% CI")
    verification_passed = False

# ============================================================================
# 5. VISUALIZATION
# ============================================================================
print("\n5. VISUALIZATION")
print("-" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Value function
axes[0, 0].plot(solver.x, V[0, :], 'b-', linewidth=2)
axes[0, 0].set_xlabel('State $x$', fontsize=12)
axes[0, 0].set_ylabel('Value $V(0,x)$', fontsize=12)
axes[0, 0].set_title('Value Function at $t=0$', fontsize=13)
axes[0, 0].grid(True, alpha=0.3)

# Optimal control
axes[0, 1].plot(solver.x, u_opt[0, :], 'r-', linewidth=2)
axes[0, 1].axhline(0, color='k', linestyle='--', alpha=0.5)
axes[0, 1].set_xlabel('State $x$', fontsize=12)
axes[0, 1].set_ylabel('Control $u^*(0,x)$', fontsize=12)
axes[0, 1].set_title('Optimal Control at $t=0$', fontsize=13)
axes[0, 1].grid(True, alpha=0.3)

# Sample trajectories
n_plot = min(50, n_paths)
t_sim = results['t']
for i in range(n_plot):
    axes[1, 0].plot(t_sim, results['x_paths'][i, :], 'b-', alpha=0.3, linewidth=0.5)
axes[1, 0].plot(t_sim, results['x_mean'], 'r-', linewidth=3, label='Mean')
axes[1, 0].axhline(0, color='k', linestyle='--', alpha=0.5)
axes[1, 0].set_xlabel('Time $t$', fontsize=12)
axes[1, 0].set_ylabel('State $X_t$', fontsize=12)
axes[1, 0].set_title(f'Sample Trajectories ({n_plot} paths)', fontsize=13)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Cost distribution
axes[1, 1].hist(results['costs'], bins=50, density=True, alpha=0.7, edgecolor='black')
axes[1, 1].axvline(V_empirical, color='r', linestyle='--', linewidth=2, 
                   label=f'Mean: {V_empirical:.3f}')
axes[1, 1].axvline(V_theoretical, color='b', linestyle='--', linewidth=2, 
                   label=f'Theory: {V_theoretical:.3f}')
axes[1, 1].set_xlabel('Cost $J$', fontsize=12)
axes[1, 1].set_ylabel('Density', fontsize=12)
axes[1, 1].set_title('Empirical Cost Distribution', fontsize=13)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/quick_demo.png', dpi=150, bbox_inches='tight')
print("✓ Figure saved to plots/quick_demo.png")

plt.show()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("DEMO COMPLETE")
print("=" * 70)

print("\nResults:")
print(f"  ✓ HJB equation solved")
print(f"  ✓ Optimal control extracted")
print(f"  ✓ {n_paths} Monte Carlo paths simulated")
print(f"  {'✓' if verification_passed else '✗'} Verification {'passed' if verification_passed else 'failed'}")
print(f"  ✓ Visualization generated")

print("\nNext steps:")
print("  1. Explore the Jupyter notebooks for detailed explanations")
print("  2. Run: jupyter notebook notebooks/")
print("  3. Start with 01_dynamic_programming_and_control.ipynb")

print("\n" + "=" * 70)
