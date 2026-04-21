import numpy as np

# Dense layer
class Dense:
    def __init__(self, input_size, output_size):
        # Initialize weights (small random values)
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))

    def forward(self, input_data):
        self.input = input_data  # Save for backprop
        return np.dot(self.input, self.weights) + self.bias
    

# Convolutional layer (basic implementation)
class Conv2D:
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        
        # Initialize filters
        self.filters = np.random.randn(num_filters, filter_size, filter_size) * 0.1

    def forward(self, input_data):
        self.input = input_data  # shape: (batch, channels, height, width)
        batch_size, channels, height, width = input_data.shape
        
        output_size = height - self.filter_size + 1
        
        output = np.zeros((batch_size, self.num_filters, output_size, output_size))

        for b in range(batch_size):
            for f in range(self.num_filters):
                for i in range(output_size):
                    for j in range(output_size):
                        region = input_data[b, 0, i:i+self.filter_size, j:j+self.filter_size]
                        output[b, f, i, j] = np.sum(region * self.filters[f])

        return output
    

# Pooling layer
class MaxPool2D:
    def __init__(self, size=2):
        self.size = size

    def forward(self, input_data):
        self.input = input_data
        batch_size, channels, height, width = input_data.shape
        
        output_height = height // self.size
        output_width = width // self.size
        
        output = np.zeros((batch_size, channels, output_height, output_width))

        for b in range(batch_size):
            for c in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        region = input_data[
                            b,
                            c,
                            i*self.size:(i+1)*self.size,
                            j*self.size:(j+1)*self.size
                        ]
                        output[b, c, i, j] = np.max(region)

        return output
    

# Flatten layer
class Flatten:
    def forward(self, input_data):
        self.input_shape = input_data.shape
        batch_size = input_data.shape[0]
        
        return input_data.reshape(batch_size, -1)