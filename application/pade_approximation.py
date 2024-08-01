import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# Given points
points = np.array([(1, 10), (2, 8), (4, 9), (5, 13), (7, 16)])
x_data = points[:, 0]
y_data = points[:, 1]

# Define the rational function form
def rational_func(params, x):
    a0, a1, a2, a3, b1, b2 = params
    numerator = a0 + a1 * x + a2 * x**2 + a3 * x**3
    denominator = 1 + b1 * x + b2 * x**2
    return numerator / denominator

# Objective function to minimize
def objective_func(params, x, y):
    return rational_func(params, x) - y

# Initial guess for the parameters
initial_guess = np.zeros(6)

# Solve the least squares problem
result = least_squares(objective_func, initial_guess, args=(x_data, y_data))

# Extract the parameters
a0, a1, a2, a3, b1, b2 = result.x

# Print the parameters
print(f"Parameters found:\na0 = {a0}\na1 = {a1}\na2 = {a2}\na3 = {a3}\nb1 = {b1}\nb2 = {b2}")

# Fit a polynomial of degree 2 (quadratic) to the points
coefficients = np.polyfit(x_data, y_data, len(x_data)-1)
polynomial = np.poly1d(coefficients)


# Generate points for plotting the fitted polynomial
x_fit = np.linspace(min(x_data), max(x_data), 1000)
y_fit = rational_func(result.x, x_fit)

# Plot the original points and the fitted polynomial
plt.scatter(x_data, y_data, color='red', label='Data Points')
plt.plot(x_fit, y_fit, label='Fitted Padé Approximant')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Padé Approximant Fit to Given Points')
plt.grid(True)
plt.show()

# Generate x values for plotting the fitted polynomial
x_fit = np.linspace(min(x_data), max(x_data), 100)
y_fit = polynomial(x_fit)

# Plot the original points
plt.scatter(x_data, y_data, color='red', label='Original Points')

# Plot the fitted polynomial
plt.plot(x_fit, y_fit, color='blue', label=f'Fitted Polynomial: {polynomial}')

# Add labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Polynomial Fit to Given Points')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()