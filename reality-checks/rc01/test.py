import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.optimize import brentq

# Define a data class to hold all the constants needed for the function f(theta)
@dataclass
class Constants:
    l1: float
    l2: float
    l3: float
    gamma: float
    x1: float
    x2: float
    y2: float
    p1: float
    p2: float
    p3: float

#### 1. Define the Function \( f(\theta) \)

def f(theta, constants):
    """
    Calculates the value based on the given angle theta and constants object.

    Parameters:
    theta (float): The angle in radians.
    constants (Constants): An object containing the necessary constants.

    Returns:
    float: The calculated result.
    """
    l1, l2, l3 = constants.l1, constants.l2, constants.l3
    gamma = constants.gamma
    x1, x2, y2 = constants.x1, constants.x2, constants.y2
    p1, p2, p3 = constants.p1, constants.p2, constants.p3

    a2 = l3 * np.cos(theta) - x1 
    b2 = l3 * np.sin(theta)
    a3 = l2 * np.cos(theta + gamma) - x2
    b3 = l2 * np.sin(theta + gamma) - y2
    d = 2 * (a2 * b3 - b2 * a3)
    
    if d == 0:
        return np.inf  # Avoid division by zero
    
    n1 = b3 * (p2**2 - p1**2 - a2**2 - b2**2) - b2 * (p3**2 - p1**2 - a3**2 - b3**2)
    n2 = -a3 * (p2**2 - p1**2 - a2**2 - b2**2) + a2 * (p3**2 - p1**2 - a3**2 - b3**2)
    
    return n1**2 + n2**2 - p1**2 * d**2

#### 2. Define Helper Functions for Root Finding

def count_roots(constants, theta_min=-np.pi, theta_max=np.pi, num_points=1000):
    """
    Counts the number of roots of f(theta) = 0 within the interval [theta_min, theta_max].

    Parameters:
    constants (Constants): The constants defining the Stewart platform.
    theta_min (float): The minimum theta value.
    theta_max (float): The maximum theta value.
    num_points (int): Number of points to sample in the interval.

    Returns:
    int: Number of unique roots found.
    list: List of roots.
    """
    theta_vals = np.linspace(theta_min, theta_max, num_points)
    
    # Compute f(theta) for each scalar theta using a list comprehension
    f_vals = np.array([f(theta, constants) for theta in theta_vals])
    
    roots = []
    
    for i in range(len(theta_vals)-1):
        if np.sign(f_vals[i]) != np.sign(f_vals[i+1]):
            try:
                root = brentq(f, theta_vals[i], theta_vals[i+1], args=(constants,))
                # Adjust root to be within [theta_min, theta_max]
                if theta_min <= root <= theta_max:
                    roots.append(root)
            except ValueError:
                # brentq failed to find a root in this interval
                pass
    
    # Remove duplicate roots within a tolerance
    roots = np.array(roots)
    unique_roots = []
    for r in roots:
        if not any(np.isclose(r, ur, atol=1e-5) for ur in unique_roots):
            unique_roots.append(r)
    
    return len(unique_roots), unique_roots

#### 3. Define the Function to Find \( p_2 \) Intervals

def find_p2_intervals(constants, p2_min, p2_max, p2_step):
    """
    Iterates over p2 values and records the number of roots for each.

    Parameters:
    constants (Constants): The constants defining the Stewart platform.
    p2_min (float): Minimum p2 value.
    p2_max (float): Maximum p2 value.
    p2_step (float): Step size for p2.

    Returns:
    dict: Dictionary with number of roots as keys and list of p2 values as values.
    """
    p2_values = np.arange(p2_min, p2_max + p2_step, p2_step)
    root_counts = {0: [], 2: [], 4: [], 6: []}
    
    for p2 in p2_values:
        constants.p2 = p2
        num_roots, roots = count_roots(constants)
        
        if num_roots in root_counts:
            root_counts[num_roots].append(p2)
        # Optionally, handle cases with roots not in {0,2,4,6}
        # else:
        #     pass
    
    return root_counts

#### 4. Implement Problem 7

# Define the constants as per Problem 4
constants = Constants(
    l1=3, 
    l2=3 * np.sqrt(2), 
    l3=3, 
    gamma=np.pi / 4, 
    x1=5, 
    x2=0, 
    y2=6,  
    p1=5, 
    p2=5,  # Initial p2; will be varied
    p3=3
)

# Define the range and step for p2
p2_min = 3.0
p2_max = 8.0
p2_step = 0.01

# Find the intervals
root_counts = find_p2_intervals(constants, p2_min, p2_max, p2_step)

# Plot the number of roots vs p2
plt.figure(figsize=(12, 6))
colors = {0: 'blue', 2: 'green', 4: 'orange', 6: 'red'}

for num_roots, p2_list in root_counts.items():
    plt.scatter(p2_list, [num_roots]*len(p2_list), label=f'{num_roots} roots', s=10, color=colors.get(num_roots, 'grey'))

plt.xlabel('$p_2$', fontsize=14)
plt.ylabel('Number of Poses (Roots)', fontsize=14)
plt.title('Number of Poses vs $p_2$', fontsize=16)
plt.legend()
plt.grid(True)
plt.show()

# Determine the intervals
# Initialize a dictionary to store intervals for each number of roots
intervals_dict = {0: [], 2: [], 4: [], 6: []}

for num_roots, p2_list in root_counts.items():
    if p2_list:
        p2_sorted = np.sort(p2_list)
        # Find continuous intervals
        diffs = np.diff(p2_sorted)
        gap = p2_step / 2
        split_indices = np.where(diffs > gap)[0] + 1
        intervals = np.split(p2_sorted, split_indices)
        
        # Append each interval to the corresponding list in intervals_dict
        for interval in intervals:
            p2_start = interval[0]
            p2_end = interval[-1]
            intervals_dict[num_roots].append((p2_start, p2_end))

# After all intervals are collected, print them
for num_roots, intervals in intervals_dict.items():
    if intervals:
        print(f"\nIntervals with {num_roots} poses:")
        for interval in intervals:
            p2_start, p2_end = interval
            print(f"  p2 from {p2_start:.2f} to {p2_end:.2f}")
