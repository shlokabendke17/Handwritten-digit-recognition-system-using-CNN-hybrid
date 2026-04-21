# # To test the PHASE-2:data loading and preprocessing 
# from data.mnist_loader import load_data
# from utils.preprocessing import preprocess_data

# X_train, y_train, X_test, y_test = load_data()

# X_train, y_train = preprocess_data(X_train, y_train)
# X_test, y_test = preprocess_data(X_test, y_test)

# print("X_train shape:", X_train.shape)
# print("y_train shape:", y_train.shape)
# print("Sample label (one-hot):", y_train[0])
# print("Pixel range:", X_train.min(), "to", X_train.max())


# # To test the PHASE-3: activations (ReLU and Softmax)
# import numpy as np
# from models.activations import relu, softmax

# x = np.array([[1, -2, 3]])

# print("ReLU:", relu(x))

# y = np.array([[2.0, 1.0, 0.1]])
# print("Softmax:", softmax(y))


# # To test the PHASE-4: layers (Dense layer)
# import numpy as np
# from models.layers import Dense

# # Example input (batch_size=1, features=4)
# x = np.array([[1, 2, 3, 4]])

# dense = Dense(4, 3)  # 4 input → 3 output

# output = dense.forward(x)

# print("Output:", output)
# print("Shape:", output.shape)


# # To test the PHASE-4: layers (Convolution layer)
# import numpy as np
# from models.layers import Conv2D

# # Dummy input: batch=1, channel=1, 28x28 image
# x = np.random.randn(1, 1, 28, 28)

# conv = Conv2D(num_filters=8, filter_size=3)

# output = conv.forward(x)

# print("Output shape:", output.shape)


# # To test the PHASE-4: layers (Pooling layer)
# import numpy as np
# from models.layers import MaxPool2D

# x = np.random.randn(1, 8, 26, 26)  # simulate conv output

# pool = MaxPool2D(size=2)

# output = pool.forward(x)

# print("Output shape:", output.shape)


# # To test the PHASE-4: layers (Flatten layer)
# import numpy as np
# from models.layers import Flatten

# x = np.random.randn(1, 8, 13, 13)

# flatten = Flatten()
# output = flatten.forward(x)

# print("Output shape:", output.shape)


# # To test the PHASE-4: model integration (SimpleCNN)
# from data.mnist_loader import load_data
# from utils.preprocessing import preprocess_data
# from models.cnn import SimpleCNN

# # Load + preprocess
# X_train, y_train, X_test, y_test = load_data()
# X_train, y_train = preprocess_data(X_train, y_train)

# # Take one sample
# x_sample = X_train[0:1]

# # Initialize model
# model = SimpleCNN()

# # Forward pass
# output = model.forward(x_sample)

# print("Output:", output)
# print("Shape:", output.shape)
# print("Sum:", output.sum())


# from training.train import train

# model = SimpleCNN()

# # Use smaller subset (IMPORTANT — speed)
# X_small = X_train[:5000]
# y_small = y_train[:5000]

# train(model, X_small, y_small, epochs=5, lr=0.01)

# preds = model.forward(X_test[:1000])
# from training.train import accuracy
# print("Test Accuracy:", accuracy(preds, y_test[:1000]))


# from data.mnist_loader import load_data
# from utils.preprocessing import preprocess_data
# from models.cnn import SimpleCNN
# from training.train import train

# # Load + preprocess
# X_train, y_train, X_test, y_test = load_data()
# X_train, y_train = preprocess_data(X_train, y_train)
# X_test, y_test = preprocess_data(X_test, y_test)

# # Initialize model
# model = SimpleCNN()

# # Train
# train(model, X_train[:5000], y_train[:5000], epochs=5, lr=0.01)

# # ===== SAVE MODEL HERE =====
# import pickle

# with open("saved_models/cnn_model.pkl", "wb") as f:
#     pickle.dump(model, f)

# print("Model saved successfully!")



# from data.mnist_loader import load_data
# from utils.preprocessing import preprocess_data
# from models.cnn import SimpleCNN
# from training.train import train, accuracy
# import numpy as np

# # ===== Load + preprocess =====
# X_train, y_train, X_test, y_test = load_data()

# X_train, y_train = preprocess_data(X_train, y_train)
# X_test, y_test = preprocess_data(X_test, y_test)

# # ===== SHUFFLE TRAINING DATA (VERY IMPORTANT) =====
# indices = np.arange(len(X_train))
# np.random.shuffle(indices)

# X_train = X_train[indices]
# y_train = y_train[indices]

# # ===== TAKE SUBSET =====
# X_small = X_train[:5000]
# y_small = y_train[:5000]

# # ===== Initialize model =====
# model = SimpleCNN()

# # ===== Train =====
# train(model, X_small, y_small, epochs=5, lr=0.01)

# # ===== TEST ACCURACY (QUICK CHECK) =====
# preds = []

# for i in range(1000):  # use small test subset for speed
#     out = model.forward(X_test[i:i+1])
#     preds.append(out)

# preds = np.vstack(preds)

# test_acc = accuracy(preds, y_test[:1000])
# print("Test Accuracy:", test_acc)



# from data.mnist_loader import load_data
# from utils.preprocessing import preprocess_data
# from models.cnn import SimpleCNN
# from training.train import train, accuracy
# import numpy as np
# import pickle

# # ===== Load + preprocess =====
# X_train, y_train, X_test, y_test = load_data()

# X_train, y_train = preprocess_data(X_train, y_train)
# X_test, y_test = preprocess_data(X_test, y_test)

# # ===== Shuffle training data (VERY IMPORTANT) =====
# indices = np.arange(len(X_train))
# np.random.shuffle(indices)

# X_train = X_train[indices]
# y_train = y_train[indices]

# # ===== Take subset for faster training =====
# subset_size = 50000

# X_small = X_train[:subset_size]
# y_small = y_train[:subset_size]

# # ===== Initialize model =====
# model = SimpleCNN()

# # ===== Train =====
# train(model, X_small, y_small, epochs=10, lr=0.01)

# # ===== Test Accuracy =====
# preds = []

# for i in range(1000):  # small batch for speed
#     out = model.forward(X_test[i:i+1])
#     preds.append(out)

# preds = np.vstack(preds)

# test_acc = accuracy(preds, y_test[:1000])
# print("Test Accuracy:", test_acc)

# # ===== SAVE MODEL (FINAL STEP) =====
# with open("saved_models/cnn_model.pkl", "wb") as f:
#     pickle.dump(model, f)

# print("Model saved successfully!")

