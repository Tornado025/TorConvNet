import numpy as np
class Pool2:

    def __init__(self, size=2, stride=2):
        self.size = size
        self.stride = stride

    def iterate_regions(self, image):
        # Generates non-overlapping image regions to pool over.
        h, w, numfilters = image.shape

        for i in range(0, h - self.size + 1, self.stride):
            for j in range(0, w - self.size + 1, self.stride):
                imregion = image[i:(i + self.size), j:(j + self.size)]
                yield imregion, i, j

    def forward(self, input):
        # Forward pass for max pooling operation.
        self.lastinput = input
        h, w, numfilters = input.shape

        outh = (h - self.size) // self.stride + 1
        outw = (w - self.size) // self.stride + 1
        output = np.zeros((outh, outw, numfilters))

        for imregion, i, j in self.iterate_regions(input):
            outi = i // self.stride
            outj = j // self.stride
            output[outi, outj] = np.amax(imregion, axis=(0, 1))

        return output

    def backprop(self, dlout):
        dlinput = np.zeros(self.lastinput.shape)

        for imregion, i, j in self.iterate_regions(self.lastinput):
            outi = i // self.stride
            outj = j // self.stride

            h, w, f = imregion.shape
            maxval = np.amax(imregion, axis=(0, 1))

            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        # If this pixel was the max value, copy the gradient to it others remain filled with 0
                        if imregion[i2, j2, f2] == maxval[f2]:
                            dlinput[i + i2, j + j2, f2] += dlout[outi, outj, f2]

        return dlinput