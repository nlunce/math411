#%% Import Libraries
import numpy as np  
from scipy.optimize import fsolve  
import sympy as sp  


#%%
# -----------------------------------------------------
# PROBLEM 1: Numerical Root-Finding for GPS Positioning
# -----------------------------------------------------

# Given constants and satellite data
c = 299792.458  # Speed of light in km/s
A = [15600, 18760, 17610, 19170]  # Satellite x-coordinates in km
B = [7540, 2750, 14630, 610]      # Satellite y-coordinates in km
C = [20140, 18610, 13480, 18390]  # Satellite z-coordinates in km
t = [0.07074, 0.07220, 0.07690, 0.07242]  # Signal travel times in seconds

# Function to define the residuals for the nonlinear system
def residuals(vars):
    """
    Residual function for GPS equations:
    sqrt((x - A_i)^2 + (y - B_i)^2 + (z - C_i)^2) - c * (t_i - d)
    """
    x, y, z, d = vars  # Unpack the unknowns
    res = []
    for i in range(4):  # Loop through the 4 satellites
        dist = np.sqrt((x - A[i])**2 + (y - B[i])**2 + (z - C[i])**2)
        res.append(dist - c * (t[i] - d))  # Append each residual
    return res

# Initial guess for (x, y, z, d)
initial_guess = [0, 0, 6370.0, 0]  # Receiver near Earth's surface and d = 0

# Solve the nonlinear system using fsolve
sol = fsolve(residuals, initial_guess)

# Print the solution for (x, y, z, d)
print("----- PROBLEM 1: Numerical Solution -----")
print(f"x = {sol[0]:.6f} km")
print(f"y = {sol[1]:.6f} km")
print(f"z = {sol[2]:.6f} km")
print(f"d = {sol[3]:.6e} seconds")
print("-----------------------------------------\n")

#%%
# ------------------------------------------------------------
# PROBLEM 2: Determinant-Based Analytical Approach for GPS
# ------------------------------------------------------------

# Define symbolic variables
x, y, z, d = sp.symbols('x y z d', real=True)

# Formulate the nonlinear equations
eqs = []
for i in range(4):
    eq = (x - A[i])**2 + (y - B[i])**2 + (z - C[i])**2 - (c * (t[i] - d))**2
    eqs.append(eq)

# Linearize the system
# Subtract eqs[1], eqs[2], and eqs[3] from eqs[0] to eliminate x^2, y^2, z^2 terms
lin_eqs = [sp.simplify(eqs[0] - eqs[i]) for i in range(1, 4)]

# Extract the coefficients of the linear equations
A_matrix, b_vector = sp.linear_eq_to_matrix(lin_eqs, [x, y, z, d])

# Split the coefficient matrix into components:
A_xyz = A_matrix[:, :3]  # Coefficients of x, y, z
A_d = A_matrix[:, 3]     # Coefficient of d

# Solve for x, y, z in terms of d
xyz_solution = A_xyz.LUsolve(b_vector - A_d * d)

# Simplify solutions for x, y, z as functions of d
x_d = sp.simplify(xyz_solution[0])
y_d = sp.simplify(xyz_solution[1])
z_d = sp.simplify(xyz_solution[2])

# Substitute x(d), y(d), z(d) into the first original equation
quadratic_eq_d = sp.simplify(eqs[0].subs({x: x_d, y: y_d, z: z_d}))

# Solve the resulting quadratic equation for d
coeffs_d = sp.Poly(quadratic_eq_d, d).all_coeffs()
d_solutions = sp.solve(quadratic_eq_d, d)

# Select the physically meaningful solution for d (real and close to zero)
d_final = None
for candidate in d_solutions:
    if candidate.is_real:
        d_final = candidate.evalf()
        break

# Compute final (x, y, z) by substituting d into x_d, y_d, z_d
x_final = x_d.subs(d, d_final).evalf()
y_final = y_d.subs(d, d_final).evalf()
z_final = z_d.subs(d, d_final).evalf()

# Print the final analytical solution
print("----- PROBLEM 2: Analytical Solution -----")
print(f"x = {x_final:.6f} km")
print(f"y = {y_final:.6f} km")
print(f"z = {z_final:.6f} km")
print(f"d = {d_final:.6e} seconds")
print("-----------------------------------------")
