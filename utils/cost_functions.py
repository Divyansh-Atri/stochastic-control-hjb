"""
Cost Functions for Stochastic Control

Defines running costs L(x,u) and terminal costs g(x) for various control problems.
Also includes analytical solutions where available (e.g., LQR).
"""

import numpy as np
from typing import Callable, Optional


class CostFunction:
    """Base class for cost functionals in stochastic control"""
    
    def running_cost(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Running cost L(x,u)"""
        raise NotImplementedError
    
    def terminal_cost(self, x: np.ndarray) -> np.ndarray:
        """Terminal cost g(x)"""
        raise NotImplementedError
    
    def running_cost_gradient_x(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Gradient of L w.r.t. x"""
        raise NotImplementedError
    
    def running_cost_gradient_u(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Gradient of L w.r.t. u"""
        raise NotImplementedError
    
    def running_cost_hessian_u(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Hessian of L w.r.t. u (for control minimization)"""
        raise NotImplementedError
    
    def terminal_cost_gradient(self, x: np.ndarray) -> np.ndarray:
        """Gradient of g w.r.t. x"""
        raise NotImplementedError


class QuadraticCost(CostFunction):
    """
    Quadratic Cost Function (LQR)
    
    L(x,u) = (1/2) q x² + (1/2) r u²
    g(x) = (1/2) q_T x²
    
    This is the standard LQR cost with known analytical solution.
    """
    
    def __init__(self, q: float = 1.0, r: float = 1.0, q_terminal: float = 1.0):
        self.q = q              # State cost weight
        self.r = r              # Control cost weight
        self.q_terminal = q_terminal  # Terminal state cost weight
    
    def running_cost(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """L(x,u) = (1/2)(q x² + r u²)"""
        return 0.5 * (self.q * x**2 + self.r * u**2)
    
    def terminal_cost(self, x: np.ndarray) -> np.ndarray:
        """g(x) = (1/2) q_T x²"""
        return 0.5 * self.q_terminal * x**2
    
    def running_cost_gradient_x(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """∂L/∂x = q x"""
        return self.q * x
    
    def running_cost_gradient_u(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """∂L/∂u = r u"""
        return self.r * u
    
    def running_cost_hessian_u(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """∂²L/∂u² = r"""
        return self.r * np.ones_like(u)
    
    def terminal_cost_gradient(self, x: np.ndarray) -> np.ndarray:
        """∂g/∂x = q_T x"""
        return self.q_terminal * x
    
    def optimal_control(self, x: np.ndarray, V_x: np.ndarray, b_u: float) -> np.ndarray:
        """
        Analytical optimal control from Hamiltonian minimization
        
        For quadratic cost: u* = -(1/r) b_u V_x
        where b_u is the control coefficient in drift
        """
        return -(b_u / self.r) * V_x


class TargetTrackingCost(CostFunction):
    """
    Target Tracking Cost
    
    L(x,u) = (1/2) q (x - x_target)² + (1/2) r u²
    g(x) = (1/2) q_T (x - x_target)²
    
    Penalizes deviation from a target state.
    """
    
    def __init__(self, x_target: float = 0.0, q: float = 1.0, r: float = 1.0, 
                 q_terminal: float = 10.0):
        self.x_target = x_target
        self.q = q
        self.r = r
        self.q_terminal = q_terminal
    
    def running_cost(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """L(x,u) = (1/2)(q(x-x_t)² + r u²)"""
        return 0.5 * (self.q * (x - self.x_target)**2 + self.r * u**2)
    
    def terminal_cost(self, x: np.ndarray) -> np.ndarray:
        """g(x) = (1/2) q_T (x - x_t)²"""
        return 0.5 * self.q_terminal * (x - self.x_target)**2
    
    def running_cost_gradient_x(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """∂L/∂x = q(x - x_t)"""
        return self.q * (x - self.x_target)
    
    def running_cost_gradient_u(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """∂L/∂u = r u"""
        return self.r * u
    
    def running_cost_hessian_u(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """∂²L/∂u² = r"""
        return self.r * np.ones_like(u)
    
    def terminal_cost_gradient(self, x: np.ndarray) -> np.ndarray:
        """∂g/∂x = q_T(x - x_t)"""
        return self.q_terminal * (x - self.x_target)


class MixedCost(CostFunction):
    """
    Mixed Cost with State and Control Penalties
    
    L(x,u) = q₁|x|^p + q₂|u|^r
    g(x) = q_T|x|^p
    
    Allows for non-quadratic penalties (e.g., L1 regularization).
    """
    
    def __init__(self, q1: float = 1.0, q2: float = 1.0, q_terminal: float = 1.0,
                 p: float = 2.0, r: float = 2.0):
        self.q1 = q1
        self.q2 = q2
        self.q_terminal = q_terminal
        self.p = p  # State penalty power
        self.r = r  # Control penalty power
    
    def running_cost(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """L(x,u) = q₁|x|^p + q₂|u|^r"""
        return self.q1 * np.abs(x)**self.p + self.q2 * np.abs(u)**self.r
    
    def terminal_cost(self, x: np.ndarray) -> np.ndarray:
        """g(x) = q_T|x|^p"""
        return self.q_terminal * np.abs(x)**self.p
    
    def running_cost_gradient_x(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """∂L/∂x = q₁ p |x|^(p-1) sign(x)"""
        return self.q1 * self.p * np.abs(x)**(self.p - 1) * np.sign(x)
    
    def running_cost_gradient_u(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """∂L/∂u = q₂ r |u|^(r-1) sign(u)"""
        return self.q2 * self.r * np.abs(u)**(self.r - 1) * np.sign(u)
    
    def running_cost_hessian_u(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """∂²L/∂u² = q₂ r(r-1) |u|^(r-2)"""
        return self.q2 * self.r * (self.r - 1) * np.abs(u)**(self.r - 2)
    
    def terminal_cost_gradient(self, x: np.ndarray) -> np.ndarray:
        """∂g/∂x = q_T p |x|^(p-1) sign(x)"""
        return self.q_terminal * self.p * np.abs(x)**(self.p - 1) * np.sign(x)


def lqr_analytical_solution(A: float, B: float, sigma: float, q: float, r: float,
                            q_terminal: float, T: float, t: np.ndarray, 
                            x: np.ndarray) -> np.ndarray:
    """
    Analytical solution for LQR value function
    
    For linear dynamics dX = (AX + Bu)dt + σ dW and quadratic cost,
    the value function is V(t,x) = (1/2) P(t) x² + ψ(t)
    
    where P(t) solves the Riccati ODE:
    dP/dt = -q + 2AP - (B²/r)P²,  P(T) = q_T
    
    Returns:
        V(t,x): Value function on grid (t, x)
    """
    # Solve Riccati equation backward in time
    dt = T / len(t)
    P = np.zeros_like(t)
    P[-1] = q_terminal
    
    # Backward integration
    for i in range(len(t) - 2, -1, -1):
        tau = T - t[i]  # Time to maturity
        dP = -q + 2 * A * P[i+1] - (B**2 / r) * P[i+1]**2
        P[i] = P[i+1] - dP * dt
    
    # Compute ψ(t) from P(t)
    psi = np.zeros_like(t)
    for i in range(len(t) - 2, -1, -1):
        dpsi = -0.5 * sigma**2 * P[i+1]
        psi[i] = psi[i+1] - dpsi * dt
    
    # Value function V(t,x) = (1/2) P(t) x² + ψ(t)
    T_grid, X_grid = np.meshgrid(t, x, indexing='ij')
    P_grid = P[:, np.newaxis] * np.ones_like(X_grid)
    psi_grid = psi[:, np.newaxis] * np.ones_like(X_grid)
    
    V = 0.5 * P_grid * X_grid**2 + psi_grid
    
    return V, P
