# preprocess raw image data by normalizing pixel values, reshaping the data into a CNN-compatible format, and converting labels into one-hot encoded vectors.

import numpy as np

# Normalize pixel values (0–255 → 0–1)
def normalize(X):
    # Grayscale pixels - integers from 0 (black) to 255 (white). -> poor performance
    # Dividing pixels value by 255.0 scales it down to floating-point numbers between 0 and 1 -> faster performance
    return X / 255.0


# Reshape data (n, 28, 28) → (n, 1, 28, 28)
def reshape(X):
    # CNNs need 4D input format: (Batch Size, Channels, Height, Width)
    return X.reshape(-1, 1, 28, 28)
    # -1 -> tells NumPy to automatically calculate batch size based on the original shape of X.
    # 1-> (grayscale- 1 channel, while RGB- 3 channels).


# Convert integer labels into binary one-hot encoded vectors of length 10
def one_hot_encode(y, num_classes=10):
    # # Removes false mathematical rank (e.g., model thinking 5 > 3)
    return np.eye(num_classes)[y]
    # creates a 10x10 identity matrix, and indexing it with [y] plucks out the perfectly formatted row for that digit.
    # y = 3 becomes [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] (the index at 3 is flipped to active).



# Combine all preprocessing steps into a single function
def preprocess_data(X, y):
    X = normalize(X)
    X = reshape(X)
    y = one_hot_encode(y)
    
    return X, y