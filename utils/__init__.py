"""
Stochastic Control and HJB Equations - Utility Package

This package provides core functionality for solving stochastic control problems
via the Hamilton-Jacobi-Bellman equation.
"""

from .sde_models import (
    SDEModel,
    LinearQuadraticModel,
    ControlledBrownianMotion,
    MeanRevertingModel,
    GeometricBrownianMotion
)

from .cost_functions import (
    CostFunction,
    QuadraticCost,
    TargetTrackingCost,
    MixedCost,
    lqr_analytical_solution
)

from .hjb_solvers import (
    HJBSolver,
    PolicyIterationSolver
)

from .simulation import (
    euler_maruyama,
    simulate_controlled_sde,
    compute_empirical_cost,
    verify_hjb_solution,
    ClosedLoopSimulator
)

__all__ = [
    # SDE Models
    'SDEModel',
    'LinearQuadraticModel',
    'ControlledBrownianMotion',
    'MeanRevertingModel',
    'GeometricBrownianMotion',
    
    # Cost Functions
    'CostFunction',
    'QuadraticCost',
    'TargetTrackingCost',
    'MixedCost',
    'lqr_analytical_solution',
    
    # HJB Solvers
    'HJBSolver',
    'PolicyIterationSolver',
    
    # Simulation
    'euler_maruyama',
    'simulate_controlled_sde',
    'compute_empirical_cost',
    'verify_hjb_solution',
    'ClosedLoopSimulator',
]
