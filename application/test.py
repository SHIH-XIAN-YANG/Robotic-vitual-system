import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def lagrange_basis(x, x_points, k):
    """
    Compute the k-th Lagrange basis polynomial at x.
    
    Parameters:
    x: float
        The x coordinate at which to evaluate the basis polynomial.
    x_points: array_like
        The x coordinates of the data points.
    k: int
        The index of the basis polynomial.
        
    Returns:
    float
        The value of the k-th basis polynomial at x.
    """
    L_k = 1.0
    n = len(x_points)
    for i in range(n):
        if i != k:
            L_k *= (x - x_points[i]) / (x_points[k] - x_points[i])
    return L_k

def lagrange_interpolation_2d(x, y, x_points, y_points, f_points):
    """
    Perform 2D Lagrange interpolation at a point (x, y).
    
    Parameters:
    x: float
        The x coordinate at which to interpolate.
    y: float
        The y coordinate at which to interpolate.
    x_points: array_like
        The x coordinates of the data points.
    y_points: array_like
        The y coordinates of the data points.
    f_points: 2D array_like
        The function values at the data points.
        
    Returns:
    float
        The interpolated value at (x, y).
    """
    n = len(x_points)
    m = len(y_points)
    
    # Compute the Lagrange basis polynomials for x and y
    L_x = np.array([lagrange_basis(x, x_points, i) for i in range(n)])
    L_y = np.array([lagrange_basis(y, y_points, j) for j in range(m)])
    
    # Compute the interpolated value
    f = 0.0
    for i in range(n):
        for j in range(m):
            f += f_points[i, j] * L_x[i] * L_y[j]
    
    return f

# Generate sample data points
x_points = np.array([-1, 0, 1])
y_points = np.array([-1, 0, 1])
f_points = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])

# Create a grid of points at which to evaluate the interpolation
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
x_grid, y_grid = np.meshgrid(x, y)
f_grid = np.zeros_like(x_grid)

# Perform the interpolation
for i in range(x_grid.shape[0]):
    for j in range(y_grid.shape[1]):
        f_grid[i, j] = lagrange_interpolation_2d(x_grid[i, j], y_grid[i, j], x_points, y_points, f_points)

# Plot the result
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x_grid, y_grid, f_grid, cmap='viridis', edgecolor='none')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Interpolated Function')
fig.colorbar(surf)
plt.show()
