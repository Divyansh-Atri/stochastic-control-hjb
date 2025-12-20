# Project Setup and Execution Guide

## Stochastic Control and Hamilton-Jacobi-Bellman Equations

**Author:** Divyansh Atri

---

## Quick Start

### 1. Install Dependencies

```bash
cd stochastic-control-hjb
pip install -r requirements.txt
```

### 2. Launch Jupyter

```bash
jupyter notebook
```

### 3. Execute Notebooks in Order

1. `01_dynamic_programming_and_control.ipynb` - Introduction and theory
2. `02_hjb_derivation.ipynb` - HJB equation derivation
3. `03_numerical_hjb_solver.ipynb` - Numerical implementation
4. `04_lqr_validation.ipynb` - Validation against analytical solution
5. `05_policy_iteration.ipynb` - Alternative solution method
6. `06_controlled_sde_simulation.ipynb` - Monte Carlo verification
7. `07_summary_and_limitations.ipynb` - Summary and discussion

---

## Project Structure

```
stochastic-control-hjb/
├── README.md                          # Comprehensive documentation
├── requirements.txt                   # Python dependencies
├── GETTING_STARTED.md                # This file
│
├── notebooks/                         # Jupyter notebooks (execute in order)
│   ├── 01_dynamic_programming_and_control.ipynb
│   ├── 02_hjb_derivation.ipynb
│   ├── 03_numerical_hjb_solver.ipynb
│   ├── 04_lqr_validation.ipynb
│   ├── 05_policy_iteration.ipynb
│   ├── 06_controlled_sde_simulation.ipynb
│   └── 07_summary_and_limitations.ipynb
│
├── utils/                             # Core implementation
│   ├── __init__.py
│   ├── sde_models.py                 # SDE model definitions
│   ├── cost_functions.py             # Cost functionals
│   ├── hjb_solvers.py                # HJB PDE solvers
│   └── simulation.py                 # Monte Carlo simulation
│
├── tests/                            # Validation
│   └── sanity_checks.ipynb
│
└── plots/                            # Generated figures (created automatically)
```

---

## What This Project Does

### Theory
- Derives the Hamilton-Jacobi-Bellman equation from first principles
- Explains dynamic programming for stochastic systems
- Discusses viscosity solutions

### Implementation
- Finite difference methods for HJB PDEs
- Backward time stepping algorithms
- Hamiltonian minimization for optimal control
- Policy iteration methods

### Validation
- Comparison with analytical LQR solution
- Monte Carlo simulation verification
- Convergence analysis
- Statistical validation

---

## Key Features

### 1. Mathematical Rigor
- Complete derivation from dynamic programming principle
- Proper treatment of stochastic calculus (Itô's formula)
- Discussion of viscosity solutions

### 2. Clean Implementation
- Modular design (separate models, costs, solvers)
- Well-documented code
- Type hints and docstrings

### 3. Comprehensive Testing
- Analytical validation (LQR)
- Numerical convergence studies
- Monte Carlo verification
- Sanity checks

### 4. Research Quality
- Suitable for IEEE submission
- Publication-ready figures
- Detailed mathematical exposition
- Thorough references

---

## Benchmark Problems

### 1. Linear-Quadratic Regulator (LQR)
- **Purpose:** Validation against known analytical solution
- **Dynamics:** Linear with quadratic cost
- **Result:** Numerical solution matches Riccati equation

### 2. Controlled Brownian Motion
- **Purpose:** Intuitive understanding of optimal control
- **Dynamics:** Direct drift control
- **Result:** Stabilization around origin

### 3. Mean-Reverting Process
- **Purpose:** Demonstrate control-reversion tradeoff
- **Dynamics:** Ornstein-Uhlenbeck with control
- **Result:** Optimal balance of natural dynamics and intervention

---

## Expected Results

### Numerical Accuracy
- LQR validation: < 10⁻⁴ relative error
- HJB residual: < 10⁻³ 
- Monte Carlo verification: Within 95% confidence interval

### Convergence
- Spatial: O(Δx²) for smooth solutions
- Temporal: O(Δt) for explicit Euler
- Policy iteration: < 20 iterations

### Computational Cost
- Grid: 101 × 201 (space × time)
- Solve time: ~10 seconds
- Memory: < 100 MB

---

## Troubleshooting

### Issue: Import errors
**Solution:** Ensure you're running notebooks from the `notebooks/` directory and that `sys.path.append('..')` is executed.

### Issue: Slow execution
**Solution:** Reduce grid size (nx, nt) for faster testing. Increase for final results.

### Issue: NaN in solution
**Solution:** Check CFL condition: `dt ≤ C·(dx)²/σ²`. Reduce dt or increase dx.

### Issue: Large HJB residual
**Solution:** Refine grid or check boundary conditions.

---

## Customization

### Add New SDE Model
1. Create class inheriting from `SDEModel` in `utils/sde_models.py`
2. Implement `drift()`, `diffusion()`, and their gradients
3. Use in notebooks

### Add New Cost Function
1. Create class inheriting from `CostFunction` in `utils/cost_functions.py`
2. Implement `running_cost()`, `terminal_cost()`, and gradients
3. Use in notebooks

### Modify Solver Parameters
In notebooks, adjust:
- `x_min`, `x_max`, `nx` - spatial domain and resolution
- `T`, `nt` - time horizon and resolution
- `u_bounds` - control constraints

---

## Performance Tips

### For Faster Execution
- Reduce `nx` and `nt` (e.g., nx=51, nt=51)
- Use fewer Monte Carlo paths (e.g., n_paths=100)
- Skip convergence studies initially

### For Higher Accuracy
- Increase `nx` and `nt` (e.g., nx=201, nt=401)
- Use more Monte Carlo paths (e.g., n_paths=10000)
- Run convergence studies

### For Production
- Use implicit time stepping (`solve_implicit()`)
- Implement parallel Hamiltonian minimization
- Consider sparse grids for higher dimensions

---

## Next Steps

### After Completing Notebooks

1. **Experiment:** Modify parameters and observe effects
2. **Extend:** Add new models or cost functions
3. **Analyze:** Study curse of dimensionality
4. **Apply:** Use for specific applications (finance, robotics, etc.)

### For Research

1. **2D Extension:** Extend to 2D state space
2. **Advanced Numerics:** Implement semi-Lagrangian schemes
3. **Machine Learning:** Combine with neural networks
4. **Applications:** Apply to real-world problems

---

## Citation

If you use this code in your research:

```
Atri, D. (2025). Stochastic Control and Hamilton-Jacobi-Bellman Equations: 
Theory and Numerical Methods. Research Implementation.
```

---

## Support

For questions or issues:
1. Check the README.md for detailed documentation
2. Review the notebook markdown cells for explanations
3. Examine the code comments and docstrings
4. Run `tests/sanity_checks.ipynb` to verify installation

---

## License

This project is released for academic and research purposes.

---

**Ready to begin? Start with `notebooks/01_dynamic_programming_and_control.ipynb`**
