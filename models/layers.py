import numpy as np

# Dense layer - combines all extracted features to produce final predictions.
class Dense:
    def __init__(self, input_size, output_size):
        # Initialize weights (small random values)
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))

    def forward(self, input_data): # passes data through layer
        self.input = input_data  # Save for backprop
        # output = input · weights(what model learns) + bias(shifts output)
        return np.dot(self.input, self.weights) + self.bias
    # Take input, Multiply with weights, Add bias, Output result

# Convolutional layer - Extract features from image using filters (kernels) and learns Edges, Curves, Patterns
#(basic implementation) - uses only one channel (grayscale) and no padding or stride.
class Conv2D:
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        
        # Initialize filters
        self.filters = np.random.randn(num_filters, filter_size, filter_size) * 0.1

    def forward(self, input_data):
        self.input = input_data  # shape: (batch, channels, height, width)
        batch_size, channels, height, width = input_data.shape
        
        # output_size = input_size - filter_size + 1
        output_size = height - self.filter_size + 1
        # 28 - 3 + 1 = 26
        
        output = np.zeros((batch_size, self.num_filters, output_size, output_size))

        for b in range(batch_size):
            for f in range(self.num_filters):
                for i in range(output_size):
                    for j in range(output_size):
                        # Extracts small patch of image (region) and applies filter to it
                        region = input_data[b, 0, i:i+self.filter_size, j:j+self.filter_size]
                        # Convolution multiplies the filter with a small region of the image and sums the result to detect features
                        output[b, f, i, j] = np.sum(region * self.filters[f])

        return output
    

# Pooling layer - Downsampling layer that reduces spatial dimensions of feature maps
class MaxPool2D:
    def __init__(self, size=2):
        self.size = size

    def forward(self, input_data):
        self.input = input_data
        batch_size, channels, height, width = input_data.shape
        
        output_height = height // self.size
        # Reduces dimensions by taking the maximum value in each non-overlapping region defined by the pooling size
        output_width = width // self.size
        
        output = np.zeros((batch_size, channels, output_height, output_width))

        for b in range(batch_size):
            for c in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        # Extracts region
                        region = input_data[
                            b,
                            c,
                            i*self.size:(i+1)*self.size,
                            j*self.size:(j+1)*self.size
                        ]
                        output[b, c, i, j] = np.max(region)
                        # Takes max value

        return output
        # Reduce image size, Keep important features, Remove noise
        # 26 × 26 → 13 × 13 
    

# Flatten layer - Converts 2D → 1D for fully connected layers
class Flatten:
    def forward(self, input_data):
        self.input_shape = input_data.shape
        batch_size = input_data.shape[0]
        
        return input_data.reshape(batch_size, -1)