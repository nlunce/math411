import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import quad
import time

# Path functions x(t) and y(t)
def x(t):
    return 0.5 + 0.3 * t + 3.9 * t**2 - 4.7 * t**3

def y(t):
    return 1.5 + 0.3 * t + 0.9 * t**2 - 2.7 * t**3

# Derivatives of x(t) and y(t) for arc length calculation
def dx_dt(t):
    return 0.3 + 2 * 3.9 * t - 3 * 4.7 * t**2

def dy_dt(t):
    return 0.3 + 2 * 0.9 * t - 3 * 2.7 * t**2

# Arc length integrand
def integrand(t):
    return np.sqrt(dx_dt(t)**2 + dy_dt(t)**2)

# Compute arc length from t=0 to t=s
def compute_arc_length(s):
    arc_length, _ = quad(integrand, 0, s)
    return arc_length

# Bisection method to find t for a target arc length fraction
def bisection_find_t(target_length, tol=1e-8):
    a, b = 0, 1
    while (b - a) / 2 > tol:
        midpoint = (a + b) / 2
        if compute_arc_length(midpoint) == target_length:
            return midpoint
        elif compute_arc_length(midpoint) < target_length:
            a = midpoint
        else:
            b = midpoint
    return (a + b) / 2

# Newton's Method to find t for a target arc length
def newton_find_t(target_length, initial_guess, tol=1e-8, max_iter=100):
    t = initial_guess
    for _ in range(max_iter):
        f_t = compute_arc_length(t) - target_length
        f_prime_t = integrand(t)
        if abs(f_t) < tol:
            return t
        t -= f_t / f_prime_t  # Update t
    return t

# Equipartition function for constant-speed points
def equipartition(n):
    partition_points = [0]
    total_length = compute_arc_length(1)
    segment_length = total_length / n
    for i in range(1, n):
        target_length = i * segment_length
        t_i = bisection_find_t(target_length)
        partition_points.append(t_i)
    partition_points.append(1)
    return partition_points

# Compare performance of Bisection and Newton's methods
def compare_performance(target_length):
    start_time_bisection = time.time()
    bisection_result = bisection_find_t(target_length)
    bisection_time = time.time() - start_time_bisection

    start_time_newton = time.time()
    newton_result = newton_find_t(target_length, initial_guess=0.5)
    newton_time = time.time() - start_time_newton

    print(f"Bisection Method Result: {bisection_result:.9f} Time: {bisection_time:.9f} seconds")
    print(f"Newton's Method Result: {newton_result:.9f} Time: {newton_time:.9f} seconds")

# Plot function for equipartitioned curve
def plot_styled_curve(n):
    plt.figure(figsize=(8, 8), facecolor='white')
    
    t_vals = np.linspace(0, 1, 500)
    x_vals = x(t_vals)
    y_vals = y(t_vals)

    key_points_t = equipartition(n)
    key_points_x = [x(t) for t in key_points_t]
    key_points_y = [y(t) for t in key_points_t]

    # Plot the curve with enhanced styling
    plt.plot(x_vals, y_vals, color="#2196F3", linewidth=2.5, zorder=3)
    plt.scatter(key_points_x, key_points_y, color="#1565C0", s=40, zorder=4)
    
    # Add grid with softer appearance
    plt.grid(True, linestyle='-', alpha=0.2, color='gray')
    plt.xticks(np.arange(-1, 1.5, 0.5))
    plt.yticks(np.arange(0, 2.5, 0.5))
    
    
    # Enhanced axis lines
    ax = plt.gca()
    ax.set_xticklabels(['' if x == 0 else str(x) for x in ax.get_xticks()])
    ax.set_yticklabels(['' if y == 0 else str(y) for y in ax.get_yticks()])
    
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Enhance tick appearance
    plt.tick_params(axis='both', which='major', length=6, width=1, colors='black', direction='out')
    plt.tick_params(axis='both', which='minor', length=3, width=1, colors='black', direction='out')
    
    # Label positioning and styling
    ax.set_ylabel('y', rotation=0, labelpad=15, y=1.02, fontsize=12)
    ax.set_xlabel('x', x=1.02, fontsize=12)
    
    plt.xlim(-1.5, 1.5)
    plt.ylim(-0.5, 2)
    plt.gca().set_aspect('equal')
    
    plt.show()

def animate_path():
    fig = plt.figure(figsize=(16, 8), facecolor='white')
    
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    t_values_original_speed = np.linspace(0, 1, 25)
    x_vals_original_speed = x(t_values_original_speed)
    y_vals_original_speed = y(t_values_original_speed)

    t_values_constant_speed = equipartition(25)
    x_vals_constant_speed = [x(t) for t in t_values_constant_speed]
    y_vals_constant_speed = [y(t) for t in t_values_constant_speed]

    # Enhanced styling function for subplots
    def style_subplot(ax, title):
        ax.grid(True, linestyle='-', alpha=0.2, color='gray')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-0.5, 2)
        ax.set_xticks(np.arange(-1, 1.5, 0.5))
        ax.set_yticks(np.arange(0, 2.5, 0.5))
        ax.set_aspect('equal')
        ax.set_xticklabels(['' if x == 0 else str(x) for x in ax.get_xticks()])
        ax.set_yticklabels(['' if y == 0 else str(y) for y in ax.get_yticks()])
        
        
        # Enhanced axis lines
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        # Enhanced ticks
        ax.tick_params(axis='both', which='major', length=6, width=1, colors='black', direction='out')
        ax.tick_params(axis='both', which='minor', length=3, width=1, colors='black', direction='out')
        
        ax.set_title(title, pad=20, fontsize=12, fontweight='bold')

    # Configure first subplot
    ax1.plot(x_vals_original_speed, y_vals_original_speed, color="#2196F3", linewidth=2.5, zorder=3)
    original_point, = ax1.plot([], [], 'o', color="#1565C0", markersize=8, zorder=4)
    style_subplot(ax1, "Original Speed")

    # Configure second subplot
    ax2.plot(x_vals_constant_speed, y_vals_constant_speed, color="#2196F3", linewidth=2.5, zorder=3)
    constant_point, = ax2.plot([], [], 'go',  markersize=8, zorder=4)
    style_subplot(ax2, "Constant Speed")

    plt.tight_layout()

    def update_original(fnum):
        original_point.set_data(x_vals_original_speed[:fnum], y_vals_original_speed[:fnum])
        return original_point,

    def update_constant(fnum):
        constant_point.set_data(x_vals_constant_speed[:fnum], y_vals_constant_speed[:fnum])
        return constant_point,

    num_frames = len(x_vals_original_speed)
    ani = animation.FuncAnimation(fig, lambda fnum: update_original(fnum) + update_constant(fnum),
                                frames=num_frames, interval=200, blit=True)
    ani.save('combined_animation.mp4', writer='ffmpeg')
    plt.show()


total_length = compute_arc_length(1)
print("Arc Length from t=0 to t=1:", total_length)

print("\nComparing Performance of Bisection and Newton's Methods:")
compare_performance(total_length / 2)  # Example for halfway arc length

plot_styled_curve(4)
plot_styled_curve(20)
animate_path()


# Parameters for the curve
A = 0.4
a = 3
f = np.pi / 2
c = 0.5
B = 0.3
b = 4
D = 0.5

# Maximum value of t for one full loop
t_max = 2 * np.pi 

# Define the functions for x(t) and y(t)
def x(t):
    return A * np.sin(a * t + f) + c

def y(t):
    return B * np.sin(b * t) + D

# Derivatives of x(t) and y(t) for arc length calculation
def dx_dt(t):
    return A * a * np.cos(a * t + f)

def dy_dt(t):
    return B * b * np.cos(b * t)

# Integrand for arc length calculation
def integrand(t):
    return np.sqrt(dx_dt(t)**2 + dy_dt(t)**2)

# Compute arc length using numerical integration
def compute_arc_length(s):
    arc_length, _ = quad(integrand, 0, s)
    return arc_length

# Equipartition function to divide path into equal arc-length segments
def equipartition(n):
    total_length = compute_arc_length(2 * np.pi)
    segment_length = total_length / n
    partition_points = [0]
    for i in range(1, n):
        target_length = i * segment_length
        partition_points.append(find_t_for_length(target_length, partition_points[-1]))
    partition_points.append(2 * np.pi)
    return partition_points

# Find parameter t for a given arc length using Newton's Method
def find_t_for_length(target_length, initial_guess=0, tol=1e-8, max_iter=100):
    t = initial_guess
    for _ in range(max_iter):
        f_t = compute_arc_length(t) - target_length
        f_prime_t = integrand(t)
        if abs(f_t) < tol:
            return t
        t -= f_t / f_prime_t
        t = max(0, min(2 * np.pi, t))
    return t


# Data for animations
n_points = 200
t_values_original = np.linspace(0, 2 * np.pi, n_points)
x_original = x(t_values_original)
y_original = y(t_values_original)

t_values_constant = equipartition(n_points)
x_constant = [x(t) for t in t_values_constant]
y_constant = [y(t) for t in t_values_constant]

# Set up the figure for side-by-side animation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Original speed plot
ax1.plot(x_original, y_original, color="#2196F3", linewidth=2)
point1, = ax1.plot([], [], 'o', color="#1565C0")
ax1.set_title("Original Speed")
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_aspect('equal')  # Ensure square grid
ax1.grid(True, linestyle='-', alpha=0.2, color='gray')

# Constant speed plot
ax2.plot(x_constant, y_constant, color="#2196F3", linewidth=2)
point2, = ax2.plot([], [], 'go')
ax2.set_title("Constant Speed")
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_aspect('equal')  # Ensure square grid
ax2.grid(True, linestyle='-', alpha=0.2, color='gray')

# Update functions for each animation
def update_original(frame):
    point1.set_data(x_original[:frame], y_original[:frame])
    return point1,

def update_constant(frame):
    point2.set_data(x_constant[:frame], y_constant[:frame])
    return point2,

# Combine animations into one
num_frames = len(x_original)
ani = animation.FuncAnimation(
    fig,
    lambda frame: update_original(frame) + update_constant(frame),
    frames=num_frames,
    interval=100,
    blit=True
)


plt.show()
# Save animation as MP4
ani.save("custom_path_animation.mp4", writer="ffmpeg")
