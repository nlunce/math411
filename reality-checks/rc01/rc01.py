import numpy as np
import matplotlib.pyplot as plt



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
    
    x = N1 / D
    y = N2 / D

    # Continue with calculations involving D (if applicable)
    result = (N1 ** 2) + (N2 ** 2) - ((P1 ** 2) * (D ** 2))
    
    return result, x, y

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

theta = np.pi / 4

result, x, y = f(theta)
print(f'f(θ) = {result}\nx = {x}\ny = {y}')


theta_values = np.linspace(-np.pi, np.pi, 1000)
results = []
for theta in theta_values:
    try:
        res, _, _ = f(theta)
        results.append(res)
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


def get_points(x, y, theta, gama):
    L1_point = (x,y)
    
    L2_x = x + (L3 * np.cos(theta))
    L2_y = y + (L3 * np.sin(theta))
    
    L2_point = (np.round(L2_x), np.round(L2_y))
    
    L3_x = x + (L2 * np.cos(theta + gama))
    L3_y = y + (L2 * np.sin(theta + gama))
    
    L3_point = (np.round(L3_x), np.round(L3_y))
    
    return [L1_point, L2_point, L3_point]

def plot_triangle(points, base_points, save_path='triangle_plot.png'):
    """
    Takes an input of three points (a list of 3 tuples or a 3x2 numpy array)
    and plots a triangle with small open circles at each of the points.
    The triangle is rendered with lines connecting each point.
    
    Parameters:
    points (list of tuples or numpy array): Points representing the vertices of the triangle.
    save_path (str): File path to save the plotted figure.
    """
    
    # Ensure points is a numpy array
    points = np.array(points)
    
    # Check if the input is in the correct shape (3x2)
    if points.shape != (3, 2):
        raise ValueError("Input should be a list of 3 points, each as a pair of (x, y) coordinates.")
    
    # Extract the x and y coordinates
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    
    # Close the triangle by repeating the first point at the end
    x_closed = np.append(x_coords, x_coords[0])
    y_closed = np.append(y_coords, y_coords[0])
    
    # Create the plot
    plt.figure(figsize=(6, 6))
    
    # Plot the triangle with open circles at each vertex
    plt.plot(x_closed, y_closed, 'b-', marker='o', markerfacecolor='none', 
             markeredgecolor='r', markersize=10, label='Triangle')
    
    
      # Plot lines from base points to corresponding triangle vertices

    for i, base in enumerate(base_points):
        plt.plot([base[0], points[i, 0]], [base[1], points[i, 1]], 'g--', linewidth=1.5, label=f'Base to Vertex {i+1}')
    
    
    
    # Set labels and title
    plt.title("Triangle with Given Vertices")
    plt.xlabel("x")
    plt.ylabel("y")
    
    # Set axis limits for better visualization
    plt.xlim(min(x_closed) - 1, max(x_closed) + 1)
    plt.ylim(min(y_closed) - 1, max(y_closed) + 1)
    
    # Add grid for better visualization
    plt.grid(True)
    
    
    # Save the figure
    plt.savefig(save_path, dpi=300)
    
    
theta = -np.pi / 4
gama = np.pi / 2

result, x, y = f(theta)

points = get_points(x, y, theta, gama)
print(points)

plot_triangle(points, base_points, save_path='triangle_plot.png')

theta = np.pi / 4
gama = np.pi / 2

result, x, y = f(theta)
base_points = [(0,0), (4,0), (0,4)]
points = get_points(x, y, theta, gama)
print(points)

plot_triangle(points, base_points, save_path='triangle_plot.png')