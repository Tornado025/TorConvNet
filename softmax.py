import numpy as np
class softmax:

    def init(self, len, nodes):
        # Initialize weights and biases
        self.weights = np.random.randn(len, nodes) / len
        self.biases = np.zeros(nodes)

    def forward(self, input):
        # Flatten the input
        self.lastinputshape = input.shape
        input = input.flatten()
        self.lastinput = input

        # Calculate totals
        totals = np.dot(input, self.weights) + self.biases
        self.lasttotals = totals

        # Softmax activation
        exptotals = np.exp(totals - np.max(totals))  # for numerical stability
        return exptotals / np.sum(exptotals, axis=0)

    def backprop(self, dldout, learningrate):
        # Gradient of loss with respect to totals
        for i, gradient in enumerate(dldout):
            if gradient == 0:
                continue

            # e^totals
            texp = np.exp(self.lasttotals - np.max(self.lasttotals))

            # Sum of all e^totals
            s = np.sum(texp)

            # Gradients of totals against softmax outputs
            doutdt = -texp[i] * texp / (s ** 2)
            doutdt[i] = texp[i] * (s - texp[i]) / (s ** 2)

            # Gradients of loss against totals
            dldt = gradient * doutdt

            # Gradients of loss against weights/biases/input
            dldw = self.lastinput[:, np.newaxis] @ dldt[np.newaxis, :]
            dldb = dldt
            dldinput = self.weights @ dldt

            # Update weights and biases
            self.weights -= learningrate * dldw
            self.biases -= learningrate * dldb

        return dldinput.reshape(self.lastinputshape)