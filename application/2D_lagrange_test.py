import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def lagrange_basis(x, y, xi, yi):
    n = len(xi)
    m = len(yi)
    L = np.ones((n, m))
    
    for i in range(n):
        for j in range(m):
            for k in range(n):
                if k != i:
                    L[i, j] *= (x - xi[k]) / (xi[i] - xi[k])
            for l in range(m):
                if l != j:
                    L[i, j] *= (y - yi[l]) / (yi[j] - yi[l])
                    
    return L

def lagrange_interpolation(x, y, xi, yi, zi):
    n = len(xi)
    m = len(yi)
    L = lagrange_basis(x, y, xi, yi)
    
    z = 0
    for i in range(n):
        for j in range(m):
            z += L[i, j] * zi[i, j]
    
    return z

# Define the points
xi = [0, 1, 2]
yi = [0, 1, 2]
zi = np.array([[1, -10, 1], [3, 4, 10], [1, 2, 3]])

# Generate the grid for plotting
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)

# Initialize Z with the same shape as X and Y
Z = np.zeros_like(X)

# Compute interpolated values
for i in range(len(x)):
    for j in range(len(y)):
        Z[j, i] = lagrange_interpolation(x[i], y[j], xi, yi, zi)

# Plotting the 3D graph
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

# Plot the original data points
xi, yi = np.meshgrid(xi, yi)
ax.scatter(xi, yi, zi, color='r', s=50, label='Data Points')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Lagrange Interpolation')
ax.legend()

plt.show()
