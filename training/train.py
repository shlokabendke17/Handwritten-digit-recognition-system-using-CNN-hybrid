import numpy as np
from models.activations import relu_derivative

# ----- Loss -----
def cross_entropy_loss(y_pred, y_true):
    epsilon = 1e-9
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred))


# ----- Accuracy -----
def accuracy(y_pred, y_true):
    pred_labels = np.argmax(y_pred, axis=1)
    true_labels = np.argmax(y_true, axis=1)
    return np.mean(pred_labels == true_labels)


# ----- Training -----
def train(model, X, y, epochs=5, lr=0.01):
    
    for epoch in range(epochs):
        total_loss = 0
        
        for i in range(len(X)):
            x_sample = X[i:i+1]
            y_sample = y[i:i+1]
            
            # ===== FORWARD =====
            output = model.forward(x_sample)
            
            # ===== LOSS =====
            loss = cross_entropy_loss(output, y_sample)
            total_loss += loss
            
            # ===== BACKPROP =====
            
            # Output gradient
            grad_output = output - y_sample  # (1,10)
            
            # ----- FC2 -----
            dW2 = np.dot(model.fc1_act.T, grad_output)
            db2 = grad_output
            
            # Backprop to FC1
            grad_fc1 = np.dot(grad_output, model.fc2.weights.T)
            grad_fc1 *= relu_derivative(model.fc1_out)
            
            # ----- FC1 -----
            dW1 = np.dot(model.flatten_out.T, grad_fc1)
            db1 = grad_fc1
            
            # ===== UPDATE =====
            model.fc2.weights -= lr * dW2
            model.fc2.bias -= lr * db2
            
            model.fc1.weights -= lr * dW1
            model.fc1.bias -= lr * db1
        
        # ===== EPOCH SUMMARY =====
        avg_loss = total_loss / len(X)
        
        # quick accuracy check (subset for speed)
        preds = model.forward(X[:500])
        acc = accuracy(preds, y[:500])
        
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Accuracy: {acc:.4f}")