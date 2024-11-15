#%%
#Libraries 
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.optimize import fsolve
from scipy.integrate import quad


#%% 
# PROBLEM 1
# Define the functions for x(t) and y(t)
def x(t):
    return 0.5 + 0.3 * t + 3.9 * t**2 - 4.7 * t**3

def y(t):
    return 1.5 + 0.3 * t + 0.9 * t**2 - 2.7 * t**3

# Define the derivatives of x(t) and y(t)
def dx_dt(t):
    return 0.3 + 2 * 3.9 * t - 3 * 4.7 * t**2

def dy_dt(t):
    return 0.3 + 2 * 0.9 * t - 3 * 2.7 * t**2

# Define the integrand for the arc length
def integrand(t):
    return np.sqrt(dx_dt(t)**2 + dy_dt(t)**2)

# Function to compute the arc length from t=0 to t=s
def compute_arc_length(s):
    arc_length, _ = quad(integrand, 0, s)
    return arc_length

# Example usage
s = 1  # Example value for s
arc_length = compute_arc_length(s)
print(f"Arc length from t=0 to t={s}: {arc_length:.3f}")


#%%
# PROBLEM 2

# Total arc length from t=0 to t=1
total_arc_length = compute_arc_length(1)

# Function to find the root, representing t^*(s)
def f(t, s):
    return compute_arc_length(t) / total_arc_length - s

# Bisection method to find t^*(s)
def find_t_star(s, tol=1e-3):
    a, b = 0, 1  # Initial interval for bisection
    while (b - a) / 2 > tol:
        midpoint = (a + b) / 2
        if f(midpoint, s) == 0:
            return midpoint  # Exact solution found
        elif f(a, s) * f(midpoint, s) < 0:
            b = midpoint
        else:
            a = midpoint
    return (a + b) / 2  # Midpoint as approximate solution

# Example usage:
s = 0.5  # Example value for s
t_star = find_t_star(s)
print(f"t^*({s}) ≈ {t_star:.3f}")
# %%
# PROBLEM 3

