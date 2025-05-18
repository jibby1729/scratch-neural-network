import numpy as np # only for math functions
import matplotlib # only to plot loss

# our layers will be : 784 (input) -> 28 (hidden) -> 28 (hidden) -> 10 (output) for a total of 784 x 28 + 28^2 + 28 x 10 weights and 28 + 28 + 10 biases 
# We will use ReLU activation for the hidden layers (for a problem as simple as this, this is all we relly need; the issue of dead neurons doesn't really make a difference here) and softmax for the output. our loss function will be a cross-entropy loss (since this is a classification problem, also it works out well with the cross-entropy formula to give a simpler gradient)

class ReLU:
    def apply(self, raw):
        self.raw = raw
        return np.maximum(0, raw)

class SoftMax:
    def apply(self, raw):
        self.raw = raw
        pass # TODO: fill out the calculation here



class Layer:

    def __init__(self, input_size, output_size, activation = None):
        # define the hidden and output layer object, this EXCLUDES the input layer. This takes in an input_size (the number of neurons in the previous layer) and an output_size (the number of neurons in this layer) alongside an activation function (defaulted to just nothing, but this won't be allowed). Note that for the output layer the output_size would be 10 and the input_size 28, and for the first hidden layer input_size is 784 whereas output_size is 28 (it has 28 neurons)

        self.weights = np.random.randn(input_size, output_size)
        # we'll be processing multiple batches at once so the "weight matrix" is actually transposed as we'll be feeding forward multiple inputs (so multiple images) at the same time into input layer

        self.biases = np.zeroes((1, output_size))
        # set biases to 0 first
        
        if activation is None:
            raise ValueError("You need to pick an activation function!")
        self.activation = activation
        

    def forward(self, inputs):
        # carry out the calculation to evaluate the neuron values in the current layer which are gonna be fed forward. 'inputs' is a matrix of shape (batch_size, input_size)
        self.inputs = inputs
        self.lintransform = np.dot(self.inputs, self.weights) + self.biases

        if self.activation is ReLU:
            return ReLU.apply(self.lintransform)
        elif self.activation is SoftMax:
            return SoftMax.apply(self.lintransform)
        else:
            return self.lintransform






