import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-np.array(x)))  # Convert x to a NumPy array

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    x = np.array(x)  # Convert x to a NumPy array
    return np.where(x > 0, x, alpha * x)

def tanh(x):
    return np.tanh(x)

# Given data
random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]

# Calculate activations for given data
sigmoid_y = sigmoid(random_values)
relu_y = relu(random_values)
leaky_relu_y = leaky_relu(random_values)
tanh_y = tanh(random_values)

# Plot activations for given data
plt.figure(figsize=(10, 6))
# plt.scatter(random_values, sigmoid_y, label='Sigmoid')
plt.scatter(random_values, relu_y, label='ReLU')
plt.scatter(random_values, leaky_relu_y, label='Leaky ReLU')
plt.scatter(random_values, tanh_y, label='Tanh')
plt.legend()
plt.xlabel('x')
plt.ylabel('Activation')
plt.title('Activation Functions for Given Data')
plt.grid(True)
plt.show()