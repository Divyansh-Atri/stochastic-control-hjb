#!/usr/bin/env python3
"""
Project Verification Script
Checks that all components are properly installed and functional
"""

import os
import sys

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"✓ {description}")
        return True
    else:
        print(f"✗ {description} - MISSING")
        return False

def check_imports():
    """Check if all required modules can be imported"""
    print("\nChecking Python imports...")
    try:
        import numpy
        print("✓ NumPy")
    except ImportError:
        print("✗ NumPy - run: pip install numpy")
        return False
    
    try:
        import scipy
        print("✓ SciPy")
    except ImportError:
        print("✗ SciPy - run: pip install scipy")
        return False
    
    try:
        import matplotlib
        print("✓ Matplotlib")
    except ImportError:
        print("✗ Matplotlib - run: pip install matplotlib")
        return False
    
    try:
        import jupyter
        print("✓ Jupyter")
    except ImportError:
        print("✗ Jupyter - run: pip install jupyter")
        return False
    
    return True

def check_utils():
    """Check if utils package is properly set up"""
    print("\nChecking utils package...")
    sys.path.insert(0, os.path.dirname(__file__))
    
    try:
        from utils import SDEModel, LinearQuadraticModel
        print("✓ SDE models")
    except ImportError as e:
        print(f"✗ SDE models - {e}")
        return False
    
    try:
        from utils import QuadraticCost, lqr_analytical_solution
        print("✓ Cost functions")
    except ImportError as e:
        print(f"✗ Cost functions - {e}")
        return False
    
    try:
        from utils import HJBSolver, PolicyIterationSolver
        print("✓ HJB solvers")
    except ImportError as e:
        print(f"✗ HJB solvers - {e}")
        return False
    
    try:
        from utils import simulate_controlled_sde, ClosedLoopSimulator
        print("✓ Simulation tools")
    except ImportError as e:
        print(f"✗ Simulation tools - {e}")
        return False
    
    return True

def main():
    print("=" * 70)
    print("STOCHASTIC CONTROL PROJECT - VERIFICATION")
    print("=" * 70)
    
    base_dir = os.path.dirname(__file__)
    
    # Check documentation
    print("\nChecking documentation...")
    all_good = True
    all_good &= check_file_exists(os.path.join(base_dir, "README.md"), "README.md")
    all_good &= check_file_exists(os.path.join(base_dir, "GETTING_STARTED.md"), "GETTING_STARTED.md")
    all_good &= check_file_exists(os.path.join(base_dir, "requirements.txt"), "requirements.txt")
    
    # Check utils
    print("\nChecking utils modules...")
    all_good &= check_file_exists(os.path.join(base_dir, "utils/__init__.py"), "utils/__init__.py")
    all_good &= check_file_exists(os.path.join(base_dir, "utils/sde_models.py"), "utils/sde_models.py")
    all_good &= check_file_exists(os.path.join(base_dir, "utils/cost_functions.py"), "utils/cost_functions.py")
    all_good &= check_file_exists(os.path.join(base_dir, "utils/hjb_solvers.py"), "utils/hjb_solvers.py")
    all_good &= check_file_exists(os.path.join(base_dir, "utils/simulation.py"), "utils/simulation.py")
    
    # Check notebooks
    print("\nChecking notebooks...")
    notebooks = [
        "01_dynamic_programming_and_control.ipynb",
        "02_hjb_derivation.ipynb",
        "03_numerical_hjb_solver.ipynb",
        "04_lqr_validation.ipynb",
        "05_policy_iteration.ipynb",
        "06_controlled_sde_simulation.ipynb",
        "07_summary_and_limitations.ipynb"
    ]
    
    for nb in notebooks:
        all_good &= check_file_exists(os.path.join(base_dir, "notebooks", nb), nb)
    
    # Check tests
    print("\nChecking tests...")
    all_good &= check_file_exists(os.path.join(base_dir, "tests/sanity_checks.ipynb"), "sanity_checks.ipynb")
    
    # Check Python dependencies
    all_good &= check_imports()
    
    # Check utils imports
    all_good &= check_utils()
    
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(base_dir, "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        print(f"\n✓ Created plots directory")
    
    # Summary
    print("\n" + "=" * 70)
    if all_good:
        print("✓ ALL CHECKS PASSED!")
        print("\nProject is ready to use.")
        print("\nNext steps:")
        print("  1. Read GETTING_STARTED.md")
        print("  2. Run: jupyter notebook")
        print("  3. Open notebooks/01_dynamic_programming_and_control.ipynb")
    else:
        print("✗ SOME CHECKS FAILED")
        print("\nPlease fix the issues above before proceeding.")
        print("Run: pip install -r requirements.txt")
    print("=" * 70)

if __name__ == "__main__":
    main()
