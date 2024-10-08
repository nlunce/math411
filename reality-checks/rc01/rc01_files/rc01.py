import numpy as np
import matplotlib.pyplot as plt

# Constants
L1 = 2
L2 = np.sqrt(2)
L3 = np.sqrt(2)
GAMA = np.pi / 2
P1 = np.sqrt(5)
P2 = np.sqrt(5)
P3 = np.sqrt(5)
X0 = 0
Y0 = 0
X1 = 4
Y1 = 0
X2 = 0
Y2 = 4

def f(theta):
    """
    Calculates a value based on the given angle theta.

    Parameters:
    theta (float): The angle in radians.

    Returns:
    float: The calculated result.

    Raises:
    ZeroDivisionError: If the denominator D is zero.
    """
    # Calculate intermediate values A2, B2, A3, B3
    A2 = (L3 * np.cos(theta)) - X1
    B2 = L3 * np.sin(theta)
    
    A3 = L2 * ((np.cos(theta) * np.cos(GAMA)) - (np.sin(theta) * np.sin(GAMA))) - X2
    B3 = L2 * ((np.cos(theta) * np.sin(GAMA)) - (np.sin(theta) * np.cos(GAMA))) - Y2
    
    # Calculate N1, N2, and D
    N1 = (B3 * ((P2 ** 2) - (P1 ** 2) - (A2 ** 2) - (B2 ** 2))) - (B2 * ((P3 ** 2) - (P1 ** 2) - (A3 ** 2) - (B3 ** 2)))
    N2 = ((-1 * A3) * ((P2 ** 2) - (P1 ** 2) - (A2 ** 2) - (B2 ** 2))) + (A2 * ((P3 ** 2) - (P1 ** 2) - (A3 ** 2) - (B3 ** 2)))
    
    D = 2 * ((A2 * B3) - (B2 * A3))
    
    # Check for division by zero
    if D == 0:
        raise ZeroDivisionError("D cannot be zero, division by zero error.")
    
    # Continue with calculations involving D (if applicable)
    result = (N1 ** 2) + (N2 ** 2) - ((P1 ** 2) * (D ** 2))
    
    return result

theta = np.pi / 4

result = f(theta)
print(result)

# Plotting f(theta) on [-pi, pi]
theta_values = np.linspace(-np.pi, np.pi, 1000)
results = []
for theta in theta_values:
    try:
        results.append(f(theta))
    except ZeroDivisionError:
        results.append(np.nan)

plt.figure(figsize=(10, 6))
plt.plot(theta_values, results, label=r'$f(\theta)$')  # Default linewidth
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)  # Default linewidth
plt.axvline(-np.pi/4, color='red', linestyle=':', linewidth=2, label=r'$\theta = -\pi/4$')  # Increased linewidth
plt.axvline(np.pi/4, color='red', linestyle=':', linewidth=2, label=r'$\theta = \pi/4$')  # Increased linewidth
plt.xlabel(r'$\theta$', fontsize=14)
plt.ylabel(r'$f(\theta)$', fontsize=14)
plt.title(r'Function of $f(\theta)$ for Stewart Platform', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()