# load the MNIST dataset and returning training and testing data in a structured format for further processing

import numpy as np
from tensorflow.keras.datasets import mnist

def load_data():
    # Load dataset (pre-split into train and test sets)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # X_train: (60000, 28, 28), y_train: (60000,), X_test: (10000, 28, 28), y_test: (10000,)

    # Convert to numpy arrays (safety, Keras already gives arrays)
    X_train = np.array(X_train) 
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Return training and testing datasets
    return X_train, y_train, X_test, y_test