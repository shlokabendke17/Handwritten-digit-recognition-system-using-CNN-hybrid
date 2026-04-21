import numpy as np
from tensorflow.keras.datasets import mnist

def load_data():
    # Load dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Convert to numpy arrays (safety, though already numpy)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return X_train, y_train, X_test, y_test