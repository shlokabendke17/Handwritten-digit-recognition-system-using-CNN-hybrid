import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import pickle

# Load trained model
with open("saved_models/cnn_model.pkl", "rb") as f:
    model = pickle.load(f)


# ===== Resize with aspect ratio preserved =====
def resize_and_pad(digit):
    h, w = digit.shape

    # Scale longest side to 20
    if h > w:
        new_h = 20
        new_w = int(w * (20 / h))
    else:
        new_w = 20
        new_h = int(h * (20 / w))

    digit = Image.fromarray((digit * 255).astype(np.uint8))
    digit = digit.resize((new_w, new_h))

    digit = np.array(digit) / 255.0

    # Create 28x28 canvas
    # new_img = np.zeros((28, 28))
    new_img = np.zeros((28, 28))
    digit = digit * 1.2
    digit = np.clip(digit, 0, 1)

    # Center it
    y_offset = (28 - new_h) // 2
    x_offset = (28 - new_w) // 2

    new_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit

    return new_img


# ===== Preprocessing =====
def preprocess_image(img):
    img = img.convert("L")
    img = np.array(img)

    # Normalize
    img = img / 255.0

    # Threshold (clean noise)
    img = (img > 0.1).astype(float)

    # Find bounding box
    coords = np.argwhere(img > 0)

    if coords.size == 0:
        return np.zeros((1,1,28,28))

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    digit = img[y_min:y_max, x_min:x_max]

    # 🔥 FIX: preserve shape
    digit = resize_and_pad(digit)

    return digit.reshape(1, 1, 28, 28)


# ===== Predict =====
def predict():
    processed = preprocess_image(image1)

    # Debug visualization
    import matplotlib.pyplot as plt
    plt.imshow(processed[0][0], cmap="gray")
    plt.title("Processed Input")
    plt.show()

    output = model.forward(processed)
    digit = np.argmax(output)
    confidence = np.max(output)

    label.config(text=f"Prediction: {digit} ({confidence*100:.1f}%)")

    print("Max:", processed.max(), "Min:", processed.min())


# ===== Clear Canvas =====
def clear():
    canvas.delete("all")
    draw.rectangle([0, 0, 280, 280], fill="black")
    label.config(text="Draw a digit")


# ===== Drawing =====
def draw_lines(event):
    x, y = event.x, event.y
    
    r = 10  # 🔥 slightly thicker brush (important improvement)
    
    canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
    draw.ellipse([x-r, y-r, x+r, y+r], fill="white")


# ===== UI Setup =====
root = tk.Tk()
root.title("Digit Recognition")

canvas = tk.Canvas(root, width=280, height=280, bg="black")
canvas.pack()

# PIL image (for processing)
image1 = Image.new("RGB", (280, 280), "black")
draw = ImageDraw.Draw(image1)

canvas.bind("<B1-Motion>", draw_lines)

# Buttons
btn_predict = tk.Button(root, text="Predict", command=predict)
btn_predict.pack()

btn_clear = tk.Button(root, text="Clear", command=clear)
btn_clear.pack()

label = tk.Label(root, text="Draw a digit", font=("Helvetica", 16))
label.pack()

root.mainloop()