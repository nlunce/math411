import numpy as np
import matplotlib.pyplot as plt

class StewartPlatform:
    def __init__(self, L1, L2, L3, GAMA,
                 P1, P2, P3,
                 X0, Y0, X1, Y1, X2, Y2,
                 theta):
        """
        Initializes the StewartPlatform with given constants and initial theta.

        Parameters:
        L1 (float): Length parameter L1.
        L2 (float): Length parameter L2.
        L3 (float): Length parameter L3.
        GAMA (float): Angle GAMA in radians.
        P1 (float): Parameter P1.
        P2 (float): Parameter P2.
        P3 (float): Parameter P3.
        X0 (float): X-coordinate X0.
        Y0 (float): Y-coordinate Y0.
        X1 (float): X-coordinate X1.
        Y1 (float): Y-coordinate Y1.
        X2 (float): X-coordinate X2.
        Y2 (float): Y-coordinate Y2.
        theta (float): Initial angle theta in radians.
        """
        # Initialize constants
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.GAMA = GAMA
        self.P1 = P1
        self.P2 = P2
        self.P3 = P3
        self.X0 = X0
        self.Y0 = Y0
        self.X1 = X1
        self.Y1 = Y1
        self.X2 = X2
        self.Y2 = Y2
        self.theta = theta

    def set_theta(self, theta):
        """
        Sets a new value for theta.

        Parameters:
        theta (float): The new angle in radians.
        """
        self.theta = theta
        print(f"Theta has been updated to {self.theta} radians.")

    def f(self):
        """
        Calculates a value based on the instance's theta.

        Returns:
        tuple: (result, x, y)

        Raises:
        ZeroDivisionError: If the denominator D is zero.
        """
        theta = self.theta
        # Calculate intermediate values A2, B2, A3, B3
        A2 = (self.L3 * np.cos(theta)) - self.X1
        B2 = self.L3 * np.sin(theta) - self.Y1

        A3 = self.L2 * (np.cos(theta) * np.cos(self.GAMA) - np.sin(theta) * np.sin(self.GAMA)) - self.X2
        B3 = self.L2 * (np.cos(theta) * np.sin(self.GAMA) + np.sin(theta) * np.cos(self.GAMA)) - self.Y2

        # Calculate N1, N2, and D
        N1 = (B3 * (self.P2**2 - self.P1**2 - A2**2 - B2**2)) - \
             (B2 * (self.P3**2 - self.P1**2 - A3**2 - B3**2))
        N2 = (-A3 * (self.P2**2 - self.P1**2 - A2**2 - B2**2)) + \
             (A2 * (self.P3**2 - self.P1**2 - A3**2 - B3**2))

        D = 2 * (A2 * B3 - B2 * A3)

        # Check for division by zero
        if D == 0:
            raise ZeroDivisionError("D cannot be zero, division by zero error.")

        x = N1 / D
        y = N2 / D

        # Continue with calculations involving D (if applicable)
        result = (N1**2) + (N2**2) - ((self.P1**2) * (D**2))

        return result, x, y

    def test_f_theta(self):
        """
        Tests the function f(theta) by printing its output based on the instance's theta.
        """
        try:
            result, x, y = self.f()
            print(f'f(theta={self.theta}) = {result}\nx = {x}\ny = {y}')
        except ZeroDivisionError as e:
            print(f'Error for theta={self.theta}: {e}')

    def plot_f_theta(self, theta_min=-np.pi, theta_max=np.pi, num_points=1000):
        """
        Plots the function f(theta) over a specified range.

        Parameters:
        theta_min (float): The minimum theta value in radians.
        theta_max (float): The maximum theta value in radians.
        num_points (int): Number of points to compute between theta_min and theta_max.
        """
        # Generate theta values and compute f(theta)
        theta_values = np.linspace(theta_min, theta_max, num_points)
        results = []
        for theta in theta_values:
            try:
                # Temporarily set theta to compute f(theta)
                original_theta = self.theta
                self.theta = theta
                res, _, _ = self.f()
                results.append(res)
                self.theta = original_theta  # Restore original theta
            except ZeroDivisionError:
                results.append(np.nan)

        plt.figure(figsize=(10, 6))
        plt.plot(theta_values, results, label=r'$f(\theta)$')
        plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
        plt.axvline(-np.pi/4, color='red', linestyle=':', linewidth=2, label=r'$\theta = -\pi/4$')
        plt.axvline(np.pi/4, color='red', linestyle=':', linewidth=2, label=r'$\theta = \pi/4$')
        plt.xlabel(r'$\theta$', fontsize=14)
        plt.ylabel(r'$f(\theta)$', fontsize=14)
        plt.title(r'Function of $f(\theta)$ for Stewart Platform', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.show()

    def get_points(self):
        """
        Computes the points of the Stewart Platform based on the instance's theta.

        Returns:
        list: List of tuples representing the points [(L1_point), (L2_point), (L3_point)].
        """
        theta = self.theta
        gama = self.GAMA

        try:
            _, x, y = self.f()
            L1_point = (x, y)

            L2_x = x + (self.L3 * np.cos(theta))
            L2_y = y + (self.L3 * np.sin(theta))
            L2_point = (np.round(L2_x, 2), np.round(L2_y, 2))

            L3_x = x + (self.L2 * np.cos(theta + gama))
            L3_y = y + (self.L2 * np.sin(theta + gama))
            L3_point = (np.round(L3_x, 2), np.round(L3_y, 2))

            return [L1_point, L2_point, L3_point]
        except ZeroDivisionError:
            return [(np.nan, np.nan), (np.nan, np.nan), (np.nan, np.nan)]

    def get_anchor_points(self):
        """
        Generates the anchor points of the Stewart Platform.

        Returns:
        list: List of tuples representing the anchor points [(X0,Y0), (X1,Y1), (X2,Y2)].
        """
        return [(self.X0, self.Y0), (self.X1, self.Y1), (self.X2, self.Y2)]

    def plot_triangle(self, ax, points, anchor_points, title=None):
        """
        Plots a triangle representing the Stewart Platform configuration on a given axis.

        Parameters:
        ax (matplotlib.axes.Axes): The axes on which to plot.
        points (list): List of tuples representing the triangle points.
        anchor_points (list): List of tuples representing the anchor points.
        title (str, optional): Title for the subplot.
        """
        if not any(np.isnan(p[0]) for p in points):
            # Ensure points and anchor_points are numpy arrays
            points = np.array(points)
            anchor_points = np.array(anchor_points)

            # Extract the x and y coordinates for the triangle vertices
            x_coords = points[:, 0]
            y_coords = points[:, 1]

            # Close the triangle by repeating the first point at the end
            x_closed = np.append(x_coords, x_coords[0])
            y_closed = np.append(y_coords, y_coords[0])

            # Plot the triangle with thicker lines
            ax.plot(x_closed, y_closed, 'r-', linewidth=3.5, label='Platform')

            # Plot blue dots at the triangle vertices
            ax.plot(x_coords, y_coords, 'bo', markersize=8, label='Vertices')

            # Plot lines from base points to corresponding triangle vertices
            for i, anchor in enumerate(anchor_points):
                ax.plot([anchor[0], points[i, 0]], [anchor[1], points[i, 1]], 'b-', linewidth=1.5)

            # Plot small green circles at the strut anchor points
            ax.plot(anchor_points[:, 0], anchor_points[:, 1], 'go', markersize=8, label='Anchor Points')

            # Set labels 
            ax.set_xlabel("x", fontsize=12)
            ax.set_ylabel("y", fontsize=12)

            # Set the domain and range for both axes
            ax.set_xlim(min(anchor_points[:,0].min(), x_coords.min()) - 1, 
                        max(anchor_points[:,0].max(), x_coords.max()) + 1)
            ax.set_ylim(min(anchor_points[:,1].min(), y_coords.min()) - 1, 
                        max(anchor_points[:,1].max(), y_coords.max()) + 1)

            # Add grid for better visualization
            ax.grid(True)
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Invalid Configuration', horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes, fontsize=12, color='red')
            ax.set_xlabel("x", fontsize=12)
            ax.set_ylabel("y", fontsize=12)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.grid(True)

        if title:
            ax.set_title(title, fontsize=14)

    def reproduce_figure(self, thetas, gama=None):
        """
        Reproduces a figure with multiple Stewart Platform configurations.

        Parameters:
        thetas (list or tuple): List of theta values in radians for which configurations are to be plotted.
        gama (float, optional): Angle gama in radians. If None, uses the instance's GAMA.
        """
        if gama is None:
            gama = self.GAMA

        anchor_points = self.get_anchor_points()
        num_configs = len(thetas)

        # Create subplots based on the number of configurations
        fig, axes = plt.subplots(1, num_configs, figsize=(7*num_configs, 6))
        if num_configs == 1:
            axes = [axes]  # Make it iterable

        for ax, theta in zip(axes, thetas):
            original_theta = self.theta  # Store original theta
            self.theta = theta  # Set new theta
            try:
                points = self.get_points()
            except ZeroDivisionError:
                points = [(np.nan, np.nan)] * 3
            self.theta = original_theta  # Restore original theta

            title = rf'Configuration at $\theta = {theta:.2f}$ rad'
            self.plot_triangle(ax, points, anchor_points, title=title)

        # Adjust layout to avoid overlap
        plt.tight_layout()

        # Display the concatenated plots
        plt.show()


# Define constants for the Stewart Platform
constants = {
    'L1': 2,
    'L2': np.sqrt(2),
    'L3': np.sqrt(2),
    'GAMA': np.pi / 2,
    'P1': np.sqrt(5),
    'P2': np.sqrt(5),
    'P3': np.sqrt(5),
    'X0': 0,
    'Y0': 0,
    'X1': 4,
    'Y1': 0,
    'X2': 0,
    'Y2': 4,
    'theta': np.pi / 4  # Initial theta
}

# Create a StewartPlatform instance with the specified constants and initial theta
platform = StewartPlatform(**constants)

# Test the function f(theta) with the initial theta
print("Testing f(theta) with initial theta (pi/4):")
platform.test_f_theta()

# Plot f(theta) over the range [-pi, pi]
print("\nPlotting f(theta) over [-pi, pi]:")
platform.plot_f_theta()

# Reproduce Figure with multiple configurations
print("\nReproducing Figure with multiple configurations:")
thetas = [np.pi / 4, -np.pi / 4, np.pi / 6, -np.pi / 6]  # Example theta values
platform.reproduce_figure(thetas)

# Example with different constants and initial theta
custom_constants = {
    'L1': 3,
    'L2': 2,
    'L3': 2.5,
    'GAMA': np.pi / 3,
    'P1': 3,
    'P2': 3,
    'P3': 3,
    'X0': 1,
    'Y0': 1,
    'X1': 5,
    'Y1': 1,
    'X2': 1,
    'Y2': 5,
    'theta': np.pi / 6  # Initial theta
}

# Create another StewartPlatform instance with custom constants and initial theta
custom_platform = StewartPlatform(**custom_constants)

# Test the function f(theta) with the initial theta
print("\nTesting f(theta) with custom initial theta (pi/6):")
custom_platform.test_f_theta()

# Plot f(theta) over the range [-pi, pi] for custom constants
print("\nPlotting f(theta) over [-pi, pi] with custom constants:")
custom_platform.plot_f_theta()

# Reproduce Figure with multiple configurations for custom constants
print("\nReproducing Figure with multiple configurations (custom constants):")
custom_thetas = [np.pi / 6, -np.pi / 6, np.pi / 3, -np.pi / 3]  # Example theta values
custom_platform.reproduce_figure(custom_thetas)

# Update theta and test again
print("\nUpdating theta to pi/3 and testing:")
platform.set_theta(np.pi / 3)
platform.test_f_theta()

# Reproduce Figure with updated theta
print("\nReproducing Figure after updating theta to pi/3:")
updated_thetas = [platform.theta, -platform.theta]
platform.reproduce_figure(updated_thetas)
