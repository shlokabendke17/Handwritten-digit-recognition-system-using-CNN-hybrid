import numpy as np

def normalize(X):
    return X / 255.0


def reshape(X):
    # Convert (n, 28, 28) → (n, 1, 28, 28)
    return X.reshape(-1, 1, 28, 28)


def one_hot_encode(y, num_classes=10):
    return np.eye(num_classes)[y]


def preprocess_data(X, y):
    X = normalize(X)
    X = reshape(X)
    y = one_hot_encode(y)
    
    return X, y

# def shuffle_data(X, y):
#     indices = np.arange(X.shape[0])
#     np.random.shuffle(indices)
#     return X[indices], y[indices]