"""
Hamilton-Jacobi-Bellman PDE Solvers

Numerical methods for solving the HJB equation:
∂V/∂t + min_u {L(x,u) + b(x,u)·∇V + (1/2)σ²ΔV} = 0

Implements:
- Finite difference discretization
- Backward time stepping
- Hamiltonian minimization
- Policy extraction
"""

import numpy as np
from typing import Tuple, Callable, Optional
from scipy.optimize import minimize_scalar, minimize
import warnings


class HJBSolver:
    """
    Finite Difference solver for 1D HJB equations
    
    Solves: ∂V/∂t + H(x, V, ∇V, ΔV) = 0
    where H is the Hamiltonian: H = min_u {L(x,u) + b(x,u)∇V + (1/2)σ²ΔV}
    """
    
    def __init__(self, x_min: float, x_max: float, nx: int,
                 T: float, nt: int, model, cost_fn):
        """
        Initialize HJB solver
        
        Args:
            x_min, x_max: Spatial domain
            nx: Number of spatial grid points
            T: Terminal time
            nt: Number of time steps
            model: SDE model (drift, diffusion)
            cost_fn: Cost function (running, terminal)
        """
        self.x_min = x_min
        self.x_max = x_max
        self.nx = nx
        self.T = T
        self.nt = nt
        self.model = model
        self.cost_fn = cost_fn
        
        # Spatial grid
        self.x = np.linspace(x_min, x_max, nx)
        self.dx = (x_max - x_min) / (nx - 1)
        
        # Time grid
        self.t = np.linspace(0, T, nt)
        self.dt = T / (nt - 1)
        
        # Solution storage
        self.V = np.zeros((nt, nx))
        self.u_opt = np.zeros((nt, nx))
        
    def compute_spatial_derivatives(self, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute spatial derivatives using central differences
        
        Returns:
            V_x: First derivative ∂V/∂x
            V_xx: Second derivative ∂²V/∂x²
        """
        V_x = np.zeros_like(V)
        V_xx = np.zeros_like(V)
        
        # Central differences for interior points
        V_x[1:-1] = (V[2:] - V[:-2]) / (2 * self.dx)
        V_xx[1:-1] = (V[2:] - 2*V[1:-1] + V[:-2]) / (self.dx**2)
        
        # Forward/backward differences at boundaries
        V_x[0] = (-3*V[0] + 4*V[1] - V[2]) / (2 * self.dx)
        V_x[-1] = (3*V[-1] - 4*V[-2] + V[-3]) / (2 * self.dx)
        
        V_xx[0] = (2*V[0] - 5*V[1] + 4*V[2] - V[3]) / (self.dx**2)
        V_xx[-1] = (2*V[-1] - 5*V[-2] + 4*V[-3] - V[-4]) / (self.dx**2)
        
        return V_x, V_xx
    
    def minimize_hamiltonian(self, x: float, V_x: float, V_xx: float,
                            u_bounds: Tuple[float, float] = (-10, 10)) -> Tuple[float, float]:
        """
        Minimize Hamiltonian w.r.t. control u
        
        H(x, u, V_x, V_xx) = L(x,u) + b(x,u)·V_x + (1/2)σ²·V_xx
        
        Returns:
            u_opt: Optimal control
            H_min: Minimum Hamiltonian value
        """
        sigma = self.model.diffusion(np.array([x]))[0]
        
        def hamiltonian(u):
            u_arr = np.array([u])
            x_arr = np.array([x])
            
            L = self.cost_fn.running_cost(x_arr, u_arr)[0]
            b = self.model.drift(x_arr, u_arr)[0]
            diffusion_term = 0.5 * sigma**2 * V_xx
            
            return L + b * V_x + diffusion_term
        
        # Use scipy minimize for robustness
        result = minimize_scalar(hamiltonian, bounds=u_bounds, method='bounded')
        
        return result.x, result.fun
    
    def solve_backward(self, u_bounds: Tuple[float, float] = (-10, 10),
                      verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve HJB equation backward in time
        
        Uses explicit time stepping with Hamiltonian minimization at each point.
        
        Returns:
            V: Value function on (t, x) grid
            u_opt: Optimal control policy on (t, x) grid
        """
        # Terminal condition
        self.V[-1, :] = self.cost_fn.terminal_cost(self.x)
        
        # Backward time stepping
        for n in range(self.nt - 2, -1, -1):
            if verbose and n % max(1, self.nt // 10) == 0:
                progress = 100 * (self.nt - 1 - n) / (self.nt - 1)
                print(f"Progress: {progress:.1f}% (time step {self.nt - 1 - n}/{self.nt - 1})")
            
            # Compute spatial derivatives at current time
            V_x, V_xx = self.compute_spatial_derivatives(self.V[n+1, :])
            
            # Minimize Hamiltonian at each spatial point
            for i in range(self.nx):
                u_opt, H_min = self.minimize_hamiltonian(
                    self.x[i], V_x[i], V_xx[i], u_bounds
                )
                
                self.u_opt[n, i] = u_opt
                
                # Explicit Euler with stability check: V^n = V^{n+1} - dt * H_min
                update = self.V[n+1, i] - self.dt * H_min
                
                # Clamp to prevent numerical overflow
                max_val = 1e10
                if np.abs(update) > max_val:
                    update = np.sign(update) * max_val
                
                self.V[n, i] = update
        
        if verbose:
            print("HJB solution complete.")
        
        return self.V, self.u_opt
    
    def solve_implicit(self, u_bounds: Tuple[float, float] = (-10, 10),
                      max_iter: int = 100, tol: float = 1e-6,
                      verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve HJB using implicit time stepping (more stable)
        
        Uses fixed-point iteration at each time step.
        """
        # Terminal condition
        self.V[-1, :] = self.cost_fn.terminal_cost(self.x)
        
        # Backward time stepping
        for n in range(self.nt - 2, -1, -1):
            if verbose and n % max(1, self.nt // 10) == 0:
                progress = 100 * (self.nt - 1 - n) / (self.nt - 1)
                print(f"Progress: {progress:.1f}% (time step {self.nt - 1 - n}/{self.nt - 1})")
            
            # Initial guess: use previous time step
            V_new = self.V[n+1, :].copy()
            
            # Fixed-point iteration
            for iter_count in range(max_iter):
                V_old = V_new.copy()
                
                # Compute derivatives
                V_x, V_xx = self.compute_spatial_derivatives(V_new)
                
                # Update each point
                for i in range(self.nx):
                    u_opt, H_min = self.minimize_hamiltonian(
                        self.x[i], V_x[i], V_xx[i], u_bounds
                    )
                    
                    self.u_opt[n, i] = u_opt
                    
                    # Implicit update: V^n = V^{n+1} - dt * H(V^n)
                    V_new[i] = self.V[n+1, i] - self.dt * H_min
                
                # Check convergence
                error = np.max(np.abs(V_new - V_old))
                if error < tol:
                    break
            
            self.V[n, :] = V_new
        
        if verbose:
            print("HJB solution complete (implicit).")
        
        return self.V, self.u_opt
    
    def get_policy(self, t: float, x: float) -> float:
        """
        Extract optimal control policy at given (t, x)
        
        Uses bilinear interpolation on computed policy grid.
        """
        # Find time index
        t_idx = np.searchsorted(self.t, t)
        t_idx = min(t_idx, self.nt - 1)
        
        # Find spatial index
        x_idx = np.searchsorted(self.x, x)
        x_idx = min(max(x_idx, 1), self.nx - 1)
        
        # Linear interpolation
        if t_idx == 0:
            u = self.u_opt[0, x_idx]
        else:
            # Time interpolation
            t_weight = (t - self.t[t_idx-1]) / (self.t[t_idx] - self.t[t_idx-1])
            u = (1 - t_weight) * self.u_opt[t_idx-1, x_idx] + \
                t_weight * self.u_opt[t_idx, x_idx]
        
        return u


class PolicyIterationSolver:
    """
    Policy Iteration for HJB equations
    
    Alternates between:
    1. Policy evaluation: Solve linear PDE for fixed policy
    2. Policy improvement: Update policy via Hamiltonian minimization
    """
    
    def __init__(self, x_min: float, x_max: float, nx: int,
                 T: float, nt: int, model, cost_fn):
        self.x_min = x_min
        self.x_max = x_max
        self.nx = nx
        self.T = T
        self.nt = nt
        self.model = model
        self.cost_fn = cost_fn
        
        self.x = np.linspace(x_min, x_max, nx)
        self.dx = (x_max - x_min) / (nx - 1)
        self.t = np.linspace(0, T, nt)
        self.dt = T / (nt - 1)
        
        self.V = np.zeros((nt, nx))
        self.u_policy = np.zeros((nt, nx))
    
    def policy_evaluation(self, u_policy: np.ndarray) -> np.ndarray:
        """
        Evaluate value function for fixed policy
        
        Solves: ∂V/∂t + L(x,u) + b(x,u)·∇V + (1/2)σ²·ΔV = 0
        """
        V = np.zeros((self.nt, self.nx))
        V[-1, :] = self.cost_fn.terminal_cost(self.x)
        
        # Backward time stepping
        for n in range(self.nt - 2, -1, -1):
            for i in range(1, self.nx - 1):
                x = self.x[i]
                u = u_policy[n, i]
                
                # Spatial derivatives (central difference)
                V_x = (V[n+1, i+1] - V[n+1, i-1]) / (2 * self.dx)
                V_xx = (V[n+1, i+1] - 2*V[n+1, i] + V[n+1, i-1]) / (self.dx**2)
                
                # Drift and diffusion
                b = self.model.drift(np.array([x]), np.array([u]))[0]
                sigma = self.model.diffusion(np.array([x]))[0]
                L = self.cost_fn.running_cost(np.array([x]), np.array([u]))[0]
                
                # Explicit Euler
                V[n, i] = V[n+1, i] - self.dt * (L + b * V_x + 0.5 * sigma**2 * V_xx)
            
            # Boundary conditions (zero gradient)
            V[n, 0] = V[n, 1]
            V[n, -1] = V[n, -2]
        
        return V
    
    def policy_improvement(self, V: np.ndarray, u_bounds: Tuple[float, float]) -> np.ndarray:
        """
        Improve policy via Hamiltonian minimization
        """
        u_new = np.zeros((self.nt, self.nx))
        
        for n in range(self.nt - 1):
            for i in range(self.nx):
                x = self.x[i]
                
                # Compute derivatives
                if i == 0:
                    V_x = (V[n, 1] - V[n, 0]) / self.dx
                elif i == self.nx - 1:
                    V_x = (V[n, -1] - V[n, -2]) / self.dx
                else:
                    V_x = (V[n, i+1] - V[n, i-1]) / (2 * self.dx)
                
                # Minimize Hamiltonian
                sigma = self.model.diffusion(np.array([x]))[0]
                
                def hamiltonian(u):
                    u_arr = np.array([u])
                    x_arr = np.array([x])
                    L = self.cost_fn.running_cost(x_arr, u_arr)[0]
                    b = self.model.drift(x_arr, u_arr)[0]
                    return L + b * V_x
                
                result = minimize_scalar(hamiltonian, bounds=u_bounds, method='bounded')
                u_new[n, i] = result.x
        
        return u_new
    
    def solve(self, u_bounds: Tuple[float, float] = (-10, 10),
             max_iter: int = 50, tol: float = 1e-4, verbose: bool = True):
        """
        Solve via policy iteration
        """
        # Initialize policy (zero control)
        self.u_policy = np.zeros((self.nt, self.nx))
        
        for iteration in range(max_iter):
            # Policy evaluation
            V_new = self.policy_evaluation(self.u_policy)
            
            # Policy improvement
            u_new = self.policy_improvement(V_new, u_bounds)
            
            # Check convergence
            policy_change = np.max(np.abs(u_new - self.u_policy))
            value_change = np.max(np.abs(V_new - self.V))
            
            if verbose:
                print(f"Iteration {iteration+1}: policy change = {policy_change:.6f}, "
                      f"value change = {value_change:.6f}")
            
            self.V = V_new
            self.u_policy = u_new
            
            if policy_change < tol and value_change < tol:
                if verbose:
                    print(f"Converged in {iteration+1} iterations")
                break
        
        return self.V, self.u_policy
