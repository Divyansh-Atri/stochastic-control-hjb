"""
Stochastic Differential Equation Models

This module defines controlled SDE models for stochastic control problems.
Each model specifies drift b(x,u), diffusion σ(x), and their derivatives.
"""

import numpy as np
from typing import Callable, Tuple


class SDEModel:
    """Base class for controlled SDE models: dX_t = b(X_t, u_t)dt + σ(X_t)dW_t"""
    
    def __init__(self, dim: int = 1):
        self.dim = dim
    
    def drift(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Drift coefficient b(x,u)"""
        raise NotImplementedError
    
    def diffusion(self, x: np.ndarray) -> np.ndarray:
        """Diffusion coefficient σ(x)"""
        raise NotImplementedError
    
    def drift_gradient(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Gradient of drift w.r.t. x: ∇_x b(x,u)"""
        raise NotImplementedError
    
    def diffusion_gradient(self, x: np.ndarray) -> np.ndarray:
        """Gradient of diffusion w.r.t. x: ∇_x σ(x)"""
        raise NotImplementedError


class LinearQuadraticModel(SDEModel):
    """
    Linear-Quadratic Regulator (LQR)
    
    Dynamics: dX_t = (A X_t + B u_t) dt + σ dW_t
    
    This is the canonical test case with known analytical solution.
    """
    
    def __init__(self, A: float = -1.0, B: float = 1.0, sigma: float = 0.5):
        super().__init__(dim=1)
        self.A = A
        self.B = B
        self.sigma = sigma
    
    def drift(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """b(x,u) = A*x + B*u"""
        return self.A * x + self.B * u
    
    def diffusion(self, x: np.ndarray) -> np.ndarray:
        """σ(x) = σ (constant)"""
        return self.sigma * np.ones_like(x)
    
    def drift_gradient(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """∇_x b = A"""
        return self.A * np.ones_like(x)
    
    def diffusion_gradient(self, x: np.ndarray) -> np.ndarray:
        """∇_x σ = 0"""
        return np.zeros_like(x)


class ControlledBrownianMotion(SDEModel):
    """
    Controlled Brownian Motion
    
    Dynamics: dX_t = u_t dt + σ dW_t
    
    The drift is directly controlled. Provides intuitive understanding
    of optimal control as balancing control cost vs state deviation.
    """
    
    def __init__(self, sigma: float = 1.0):
        super().__init__(dim=1)
        self.sigma = sigma
    
    def drift(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """b(x,u) = u"""
        return u
    
    def diffusion(self, x: np.ndarray) -> np.ndarray:
        """σ(x) = σ (constant)"""
        return self.sigma * np.ones_like(x)
    
    def drift_gradient(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """∇_x b = 0"""
        return np.zeros_like(x)
    
    def diffusion_gradient(self, x: np.ndarray) -> np.ndarray:
        """∇_x σ = 0"""
        return np.zeros_like(x)


class MeanRevertingModel(SDEModel):
    """
    Mean-Reverting Process with Control (Ornstein-Uhlenbeck type)
    
    Dynamics: dX_t = (θ(μ - X_t) + u_t) dt + σ dW_t
    
    Demonstrates tradeoff between:
    - Natural mean reversion (θ)
    - Control intervention (u)
    - Stochastic fluctuations (σ)
    """
    
    def __init__(self, theta: float = 2.0, mu: float = 0.0, sigma: float = 0.5):
        super().__init__(dim=1)
        self.theta = theta  # Mean reversion speed
        self.mu = mu        # Long-term mean
        self.sigma = sigma  # Volatility
    
    def drift(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """b(x,u) = θ(μ - x) + u"""
        return self.theta * (self.mu - x) + u
    
    def diffusion(self, x: np.ndarray) -> np.ndarray:
        """σ(x) = σ (constant)"""
        return self.sigma * np.ones_like(x)
    
    def drift_gradient(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """∇_x b = -θ"""
        return -self.theta * np.ones_like(x)
    
    def diffusion_gradient(self, x: np.ndarray) -> np.ndarray:
        """∇_x σ = 0"""
        return np.zeros_like(x)


class GeometricBrownianMotion(SDEModel):
    """
    Controlled Geometric Brownian Motion
    
    Dynamics: dX_t = (μ X_t + u_t) dt + σ X_t dW_t
    
    State-dependent diffusion makes this nonlinear.
    Relevant for financial applications.
    """
    
    def __init__(self, mu: float = 0.05, sigma: float = 0.2):
        super().__init__(dim=1)
        self.mu = mu
        self.sigma = sigma
    
    def drift(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """b(x,u) = μ*x + u"""
        return self.mu * x + u
    
    def diffusion(self, x: np.ndarray) -> np.ndarray:
        """σ(x) = σ*x"""
        return self.sigma * x
    
    def drift_gradient(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """∇_x b = μ"""
        return self.mu * np.ones_like(x)
    
    def diffusion_gradient(self, x: np.ndarray) -> np.ndarray:
        """∇_x σ = σ"""
        return self.sigma * np.ones_like(x)
