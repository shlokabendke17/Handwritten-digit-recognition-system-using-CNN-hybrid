import numpy as np

# ReLU Activation
def relu(x):
    return np.maximum(0, x)


# ReLU Derivative (for backprop later)
def relu_derivative(x):
    return (x > 0).astype(float)


# Softmax Activation
def softmax(x):
    # Stability trick (VERY IMPORTANT)
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)