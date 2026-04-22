from data.mnist_loader import load_data
from utils.preprocessing import preprocess_data
from models.cnn import SimpleCNN
from training.train import train, accuracy
import numpy as np
import pickle

# ===== Load + preprocess =====
X_train, y_train, X_test, y_test = load_data()

X_train, y_train = preprocess_data(X_train, y_train)
X_test, y_test = preprocess_data(X_test, y_test)

# ===== Shuffle training data (VERY IMPORTANT) =====
indices = np.arange(len(X_train))
np.random.shuffle(indices)

X_train = X_train[indices]
y_train = y_train[indices]

# ===== Take subset for faster training =====
subset_size = 50000

X_small = X_train[:subset_size]
y_small = y_train[:subset_size]

# ===== Initialize model =====
model = SimpleCNN()

# ===== Train =====
train(model, X_small, y_small, epochs=10, lr=0.01)

# ===== Test Accuracy =====
preds = []

for i in range(1000):  # small batch for speed
    out = model.forward(X_test[i:i+1])
    preds.append(out)

preds = np.vstack(preds)

test_acc = accuracy(preds, y_test[:1000])
print("Test Accuracy:", test_acc)

# ===== SAVE MODEL (FINAL STEP) =====
with open("saved_models/cnn_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")

# import numpy as np
# import pickle
# import matplotlib.pyplot as plt
# import seaborn as sns

# from data.mnist_loader import load_data
# from utils.preprocessing import preprocess_data
# from models.cnn import SimpleCNN
# from training.train import train, accuracy


# def prepare_data():
#     print(" Loading dataset...")
#     X_train, y_train, X_test, y_test = load_data()

#     print(" Preprocessing data...")
#     X_train, y_train = preprocess_data(X_train, y_train)
#     X_test, y_test = preprocess_data(X_test, y_test)

#     # Shuffle training data
#     indices = np.arange(len(X_train))
#     np.random.shuffle(indices)

#     X_train = X_train[indices]
#     y_train = y_train[indices]

#     return X_train, y_train, X_test, y_test


# def train_model(X_train, y_train, subset_size=50000, epochs=10, lr=0.01):
#     print(" Initializing model...")
#     model = SimpleCNN()

#     print(f" Training on {subset_size} samples...")
#     X_subset = X_train[:subset_size]
#     y_subset = y_train[:subset_size]

#     train(model, X_subset, y_subset, epochs=epochs, lr=lr)

#     return model


# def evaluate_model(model, X_test, y_test, test_samples=1000):
#     print("📊 Evaluating model...")

#     preds = []
#     true_labels = []

#     for i in range(test_samples):
#         out = model.forward(X_test[i:i+1])
#         preds.append(np.argmax(out))
#         true_labels.append(np.argmax(y_test[i]))

#     preds = np.array(preds)
#     true_labels = np.array(true_labels)

#     # ===== Accuracy =====
#     acc = np.mean(preds == true_labels)
#     print(f"✅ Test Accuracy: {acc * 100:.2f}%")

#     # ===== Confusion Matrix =====
#     cm = np.zeros((10, 10), dtype=int)

#     for t, p in zip(true_labels, preds):
#         cm[t][p] += 1

#     cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

#     # ===== Plot =====
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues")
#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     plt.title("Confusion Matrix")
#     plt.show()

#     return acc


# def save_model(model, path="saved_models/cnn_model1.pkl"):
#     print(" Saving model...")
#     with open(path, "wb") as f:
#         pickle.dump(model, f)
#     print(" Model saved successfully!")


# def main():
#     # ===== STEP 1: DATA =====
#     X_train, y_train, X_test, y_test = prepare_data()

#     # ===== STEP 2: TRAIN =====
#     model = train_model(
#         X_train,
#         y_train,
#         subset_size=50000,   # adjust if slow
#         epochs=10,
#         lr=0.01
#     )

#     # ===== STEP 3: EVALUATE =====
#     evaluate_model(model, X_test, y_test)

#     # ===== STEP 4: SAVE =====
#     save_model(model)


# if __name__ == "__main__":
#     main()