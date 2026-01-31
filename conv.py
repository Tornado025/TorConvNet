import numpy as np
class Conv3x3:
# A convolutional layer using 3x3 filters.

    def __init__(self, num_filters, padding=0, stride=1):
        self.num_filters = num_filters
        self.padding = padding
        self.stride = stride
        # Initialize filters with random values , 9 is used to normalize as per Xavier initialization
        self.filters = np.random.randn(num_filters, 3, 3) / 9

    def iterate_regions(self, image):
        # Generates 3x3 image regions with specified padding and stride.
        h, w = image.shape
        # Apply padding if needed
        if self.padding > 0:
            image = np.pad(image, self.padding, mode='constant', constant_values=0)
            h, w = image.shape
        
        # Iterate with stride
        for i in range(0, h - 2, self.stride):
            for j in range(0, w - 2, self.stride):
                im_region = image[i:(i + 3), j:(j + 3)]
                yield im_region, i, j

    def forward(self, input):
        #Forward propagation i.e. performing the convolution operation which is an element-wise multiplication
        self.last_input = input
        h, w = input.shape
        
        # Apply padding if needed
        if self.padding > 0:
            input = np.pad(input, self.padding, mode='constant', constant_values=0)
            h, w = input.shape
        
        # Calculate output dimensions based on padding and stride
        outh = (h - 3 + 2*self.padding) // self.stride + 1
        outw = (w - 3 + 2*self.padding) // self.stride + 1
        output = np.zeros((outh, outw, self.num_filters))

        for im_region, i, j in self.iterate_regions(input):
            # Adjust indices for stride
            outi = i // self.stride
            outj = j // self.stride
            for f in range(self.num_filters):
                output[outi, outj, f] = np.sum(im_region * self.filters[f])

        return output

    def backprop(self, dldout, learning_rate):
        dldfilters = np.zeros(self.filters.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            # Adjust indices for stride
            outi = i // self.stride
            outj = j // self.stride
            for f in range(self.num_filters):
                dldfilters[f] += dldout[outi, outj, f] * im_region

        # Update filters
        self.filters -= learning_rate * dldfilters

        return None  