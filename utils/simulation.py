"""
Stochastic Simulation Tools

Implements numerical integration schemes for controlled SDEs:
- Euler-Maruyama method
- Closed-loop simulation with feedback control
- Monte Carlo trajectory generation
"""

import numpy as np
from typing import Callable, Tuple, Optional


def euler_maruyama(x0: float, t_span: Tuple[float, float], dt: float,
                   drift: Callable, diffusion: Callable, control: Callable,
                   seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Euler-Maruyama scheme for controlled SDE
    
    Simulates: dX_t = b(X_t, u_t) dt + σ(X_t) dW_t
    
    Args:
        x0: Initial condition
        t_span: (t_start, t_end)
        dt: Time step
        drift: Function b(t, x, u)
        diffusion: Function σ(t, x)
        control: Function u(t, x) - feedback control law
        seed: Random seed for reproducibility
    
    Returns:
        t: Time array
        x: State trajectory
        u: Control trajectory
    """
    if seed is not None:
        np.random.seed(seed)
    
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t)
    
    x = np.zeros(n_steps)
    u = np.zeros(n_steps)
    x[0] = x0
    
    # Generate Brownian increments
    dW = np.sqrt(dt) * np.random.randn(n_steps - 1)
    
    for i in range(n_steps - 1):
        # Compute control
        u[i] = control(t[i], x[i])
        
        # Euler-Maruyama step
        b = drift(t[i], x[i], u[i])
        sigma = diffusion(t[i], x[i])
        
        x[i+1] = x[i] + b * dt + sigma * dW[i]
    
    # Final control
    u[-1] = control(t[-1], x[-1])
    
    return t, x, u


def simulate_controlled_sde(x0: float, T: float, dt: float, model, policy_fn: Callable,
                           n_paths: int = 1, seed: Optional[int] = None) -> dict:
    """
    Simulate multiple paths of controlled SDE
    
    Args:
        x0: Initial state
        T: Terminal time
        dt: Time step
        model: SDE model with drift and diffusion methods
        policy_fn: Feedback control u(t, x)
        n_paths: Number of Monte Carlo paths
        seed: Random seed
    
    Returns:
        Dictionary with keys: 't', 'x_paths', 'u_paths', 'x_mean', 'x_std'
    """
    if seed is not None:
        np.random.seed(seed)
    
    t = np.arange(0, T + dt, dt)
    n_steps = len(t)
    
    x_paths = np.zeros((n_paths, n_steps))
    u_paths = np.zeros((n_paths, n_steps))
    
    for path_idx in range(n_paths):
        # Generate Brownian increments
        dW = np.sqrt(dt) * np.random.randn(n_steps - 1)
        
        x_paths[path_idx, 0] = x0
        
        for i in range(n_steps - 1):
            x_current = x_paths[path_idx, i]
            t_current = t[i]
            
            # Feedback control
            u_current = policy_fn(t_current, x_current)
            u_paths[path_idx, i] = u_current
            
            # SDE step
            b = model.drift(np.array([x_current]), np.array([u_current]))[0]
            sigma = model.diffusion(np.array([x_current]))[0]
            
            x_paths[path_idx, i+1] = x_current + b * dt + sigma * dW[i]
        
        # Final control
        u_paths[path_idx, -1] = policy_fn(t[-1], x_paths[path_idx, -1])
    
    # Compute statistics
    x_mean = np.mean(x_paths, axis=0)
    x_std = np.std(x_paths, axis=0)
    
    return {
        't': t,
        'x_paths': x_paths,
        'u_paths': u_paths,
        'x_mean': x_mean,
        'x_std': x_std
    }


def compute_empirical_cost(t: np.ndarray, x_paths: np.ndarray, u_paths: np.ndarray,
                          cost_fn, dt: float) -> Tuple[np.ndarray, float, float]:
    """
    Compute empirical cost from simulated trajectories
    
    J = E[∫₀ᵀ L(X_t, u_t) dt + g(X_T)]
    
    Args:
        t: Time array
        x_paths: State trajectories (n_paths × n_steps)
        u_paths: Control trajectories (n_paths × n_steps)
        cost_fn: Cost function object
        dt: Time step
    
    Returns:
        costs: Cost for each path
        mean_cost: Mean cost
        std_cost: Standard deviation of cost
    """
    n_paths = x_paths.shape[0]
    costs = np.zeros(n_paths)
    
    for i in range(n_paths):
        # Running cost (trapezoidal integration)
        running_costs = cost_fn.running_cost(x_paths[i, :], u_paths[i, :])
        integral = np.trapz(running_costs, dx=dt)
        
        # Terminal cost
        terminal = cost_fn.terminal_cost(np.array([x_paths[i, -1]]))[0]
        
        costs[i] = integral + terminal
    
    return costs, np.mean(costs), np.std(costs)


def verify_hjb_solution(V_numerical: np.ndarray, t_grid: np.ndarray, x_grid: np.ndarray,
                       model, cost_fn, u_opt: np.ndarray, dx: float, dt: float) -> dict:
    """
    Verify HJB solution by checking residual
    
    Computes: R = ∂V/∂t + min_u {L + b·∇V + (1/2)σ²·ΔV}
    
    Should be approximately zero if solution is correct.
    
    Returns:
        Dictionary with residual statistics
    """
    nt, nx = V_numerical.shape
    residual = np.zeros((nt - 1, nx - 2))  # Exclude boundaries and terminal time
    
    for n in range(nt - 1):
        for i in range(1, nx - 1):
            x = x_grid[i]
            u = u_opt[n, i]
            
            # Time derivative (forward difference)
            V_t = (V_numerical[n+1, i] - V_numerical[n, i]) / dt
            
            # Spatial derivatives
            V_x = (V_numerical[n, i+1] - V_numerical[n, i-1]) / (2 * dx)
            V_xx = (V_numerical[n, i+1] - 2*V_numerical[n, i] + V_numerical[n, i-1]) / (dx**2)
            
            # Hamiltonian
            L = cost_fn.running_cost(np.array([x]), np.array([u]))[0]
            b = model.drift(np.array([x]), np.array([u]))[0]
            sigma = model.diffusion(np.array([x]))[0]
            
            H = L + b * V_x + 0.5 * sigma**2 * V_xx
            
            # HJB residual
            residual[n, i-1] = V_t + H
    
    return {
        'residual': residual,
        'max_abs_residual': np.max(np.abs(residual)),
        'mean_abs_residual': np.mean(np.abs(residual)),
        'rms_residual': np.sqrt(np.mean(residual**2))
    }


class ClosedLoopSimulator:
    """
    Closed-loop simulator for controlled SDEs with feedback policy
    """
    
    def __init__(self, model, cost_fn, policy_fn: Callable):
        """
        Args:
            model: SDE model
            cost_fn: Cost function
            policy_fn: Feedback control law u(t, x)
        """
        self.model = model
        self.cost_fn = cost_fn
        self.policy_fn = policy_fn
    
    def simulate(self, x0: float, T: float, dt: float, n_paths: int = 100,
                seed: Optional[int] = None) -> dict:
        """
        Run closed-loop Monte Carlo simulation
        
        Returns comprehensive statistics and trajectories
        """
        results = simulate_controlled_sde(x0, T, dt, self.model, self.policy_fn,
                                         n_paths, seed)
        
        # Compute costs
        costs, mean_cost, std_cost = compute_empirical_cost(
            results['t'], results['x_paths'], results['u_paths'],
            self.cost_fn, dt
        )
        
        results['costs'] = costs
        results['mean_cost'] = mean_cost
        results['std_cost'] = std_cost
        
        # Compute control statistics
        results['u_mean'] = np.mean(results['u_paths'], axis=0)
        results['u_std'] = np.std(results['u_paths'], axis=0)
        
        return results
    
    def compare_with_analytical(self, x0: float, T: float, dt: float,
                               V_analytical: Callable, n_paths: int = 100,
                               seed: Optional[int] = None) -> dict:
        """
        Compare numerical solution with analytical solution (if available)
        """
        results = self.simulate(x0, T, dt, n_paths, seed)
        
        # Evaluate analytical value function at initial point
        V_analytical_value = V_analytical(0, x0)
        
        # Compare with empirical cost
        error = results['mean_cost'] - V_analytical_value
        relative_error = error / abs(V_analytical_value) if V_analytical_value != 0 else np.inf
        
        results['V_analytical'] = V_analytical_value
        results['error'] = error
        results['relative_error'] = relative_error
        
        return results
