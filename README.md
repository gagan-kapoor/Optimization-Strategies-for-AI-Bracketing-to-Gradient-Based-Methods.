# Optimization from Scratch – Python

**Course Project – Optimization in Engineering**  
Designed and implemented a modular optimization framework from scratch in Python, capable of solving single-variable, multivariable, and constrained optimization problems. Includes convergence visualizations and benchmark testing.

---

## Project Overview

This project implements a set of core optimization algorithms and applies them to classical benchmark problems. It includes:

- Single-variable optimization using Bounding Phase and Interval Halving methods
- Multivariable optimization using the Conjugate Gradient Method with line search
- Constrained optimization using the Bracket Operator Penalty Method
- Visualization tools to analyze convergence, feasibility, and search trajectories

---

## Key Features

### 1. Single Variable Optimizer
**File:** `single_var_optimizer.py`

Implements:
- Bounding Phase Method
- Interval Halving Method

Features:
- Objective function evaluation tracking
- Iteration-wise function value plotting

### 2. Multi Variable Optimizer
**File:** `multi_var_optimizer.py`

Implements:
- Fletcher-Reeves Conjugate Gradient Method
- Numerical gradient calculation
- Line search via single-variable optimizer

Features:
- 2D surface plots for bivariate functions
- Progress plots of function value per iteration

### 3. Constrained Optimization
**File:** `constrained_multivar.py` or `constrained_multivar.ipynb`

Implements:
- Bracket Operator Penalty Method for inequality constraints

Features:
- Handles variable bounds and nonlinear constraints
- Visualizes feasible region and optimization trajectory (for 2D problems)

---

## Benchmark Problems

Tested on the following classical optimization functions:

- Sum of Squares
- Rosenbrock Function
- Dixon-Price Function
- Trid Function
- Zakharov Function
- Custom constrained problems (2D and 8D)

---

## Visual Outputs
- Plot: Objective function value vs iteration
- Plot: Function evaluations vs iteration
- 2D contour plots with feasible region overlays (for 2D problems)
- Optimization path annotated on surface

## Relevance to Machine Learning
This project builds intuition and practical experience with:

- Loss minimization
- Constrained optimization (similar to L1, L2 regularization and SVM constraints)
- Numerical optimization techniques used in training ML models
- Gradient-based search and line search strategies
- Visualization of non-convex loss surfaces
## References
- S. S. Rao – Engineering Optimization: Theory and Practice
- Nocedal & Wright – Numerical Optimization
- Optimization problem definitions from academic benchmarks
## Author
- GAGAN KAPOOR
