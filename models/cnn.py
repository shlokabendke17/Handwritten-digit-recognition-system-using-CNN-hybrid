import numpy as np
from models.layers import Conv2D, MaxPool2D, Flatten, Dense
from models.activations import relu, softmax

# Conv → ReLU → Pool → Flatten → Dense → Softmax
class SimpleCNN:
    def __init__(self):
        # Layers
        self.conv1 = Conv2D(num_filters=8, filter_size=3)
        self.pool1 = MaxPool2D(size=2)
        
        self.flatten = Flatten()
        
        # 8 * 13 * 13 = 1352
        self.fc1 = Dense(1352, 128)
        self.fc2 = Dense(128, 10)

    def forward(self, x):
        # ----- Conv Layer -----
        x = self.conv1.forward(x)
        x = relu(x)

        # ----- Pooling -----
        x = self.pool1.forward(x)

        # ----- Flatten -----
        self.flatten_out = self.flatten.forward(x)

        # ----- Fully Connected 1 -----
        self.fc1_out = self.fc1.forward(self.flatten_out)
        self.fc1_act = relu(self.fc1_out)

        # ----- Fully Connected 2 -----
        self.fc2_out = self.fc2.forward(self.fc1_act)

        # ----- Output -----
        output = softmax(self.fc2_out)

        return output