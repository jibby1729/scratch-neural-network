import numpy as np # only for math functions
import matplotlib # only to plot loss

# our layers will be : 784 (input) -> 28 (hidden) -> 28 (hidden) -> 10 (output) for a total of 784 x 28 + 28^2 + 28 x 10 weights and 28 + 28 + 10 biases 
# We will use ReLU activation for the hidden layers (for a problem as simple as this, this is all we relly need; the issue of dead neurons doesn't really make a difference here) and softmax for the output. our loss function will be a cross-entropy loss (since this is a classification problem, also it works out well with the cross-entropy formula to give a simpler gradient)



class ReLU:
    def apply(self, raw):
        #raw will be the raw values of the neurons after the lintransform, it should be np.array of shape (batch_size, output_size)
        self.raw = raw
        return np.maximum(0, raw)
    
    def backward(self, dL_dout):
        return dL_dout * (self.raw > 0)

class SoftMax:
    def apply(self, raw):
        #raw will be the raw values of the neurons after the lintransform, it should be np.array of shape (batch_size, output_size)
        self.raw = raw
        # we will actually implement stable version, using softmax(v + c) = softmax(v) as calculating the exponentials might cause overflow issues
        normalized_exp = np.exp(raw - np.max(raw, axis = 1, keepdims=True))
        return normalized_exp / (np.sum(normalized_exp, axis = 1, keepdims=True))
    
    def backward(self, dL_dout):
        # not needed tbh but to make backward work for both activation classes this is more convenient, useless function that doesn't do anything 
        return dL_dout
        

class CrossEntropy:
    def calc(self, predicted, actual):
        # calculates the entropy loss. predicted is (batch_size, output_size) where output_size is the one for the output layer, and actual will be the same shape
        usable = np.clip(predicted, 1e-10, 1e10)
        return -np.sum((actual * np.log(usable))) / (len(actual))
    
    def backward(self, predicted, actual):
        # note that we are using the softmax function with the cross-entropy, so this (a chain rule calculation, do it yourself) simplifies a lot if you remember that actual is a bunch of one-hot vectors 
        return predicted - actual


class Layer:

    def __init__(self, input_size, output_size, activation = None):
        # define the hidden and output layer object, this EXCLUDES the input layer. This takes in an input_size (the number of neurons in the previous layer) and an output_size (the number of neurons in this layer) alongside an activation function (defaulted to just nothing, but this won't be allowed). Note that for the output layer the output_size would be 10 and the input_size 28, and for the first hidden layer input_size is 784 whereas output_size is 28 (it has 28 neurons)

        self.weights = 0.1 * np.random.randn(input_size, output_size)
        # we'll be processing multiple batches at once so the "weight matrix" is actually transposed as we'll be feeding forward multiple inputs (so multiple images) at the same time into input layer

        self.biases = np.zeros((1, output_size))
        # set biases to 0 first
        
        if activation is None:
            raise ValueError("You need to pick an activation function!")
        self.activation = activation
    

    def forward(self, inputs):
        # carry out the calculation to evaluate the neuron values in the current layer which are gonna be fed forward. 'inputs' is a matrix of shape (batch_size, input_size)
        self.inputs = inputs
        self.lintransform = np.dot(self.inputs, self.weights) + self.biases
        return self.activation.apply(self.lintransform)
    
    def backward(self, dL_dout, learning_rate=0.01):
        #backpropagation throguh the layer
        dL_dout = self.activation.backward(dL_dout)

        # Gradients for weights and biases, this is just the backpropagation calculus, note that for db we sum the rows (updating for the whole batch at once)
        dW = np.dot(self.inputs.T, dL_dout)
        db = np.sum(dL_dout, axis=0, keepdims=True)

        # Update weights and biases, this is gradient descent 
        self.weights -= learning_rate * dW
        self.biases -= learning_rate * db
        
        # Gradient for the previous layer
        return np.dot(dL_dout, self.weights.T)

    
class nn:
    def __init__(self, layers):
    # layers will be a List[Layer] object (an array of Layer objects), our loss is the cross-entropy loss (though if you want you can put in sth else but then you'd have to define its class with its own calc and backward methods)
        self.layers = layers
        self.loss = CrossEntropy()
    
    def forward(self, inputs):
    # push the input forward through the layers
        outputs = inputs
        for layer in self.layers:
            outputs = layer.forward(outputs)
        return outputs
    
    def backward(self, predicted, actual, learning_rate=0.01):
        # Start with the loss gradient (which is v simple in our case, but i am building this to be more flexible in case u wanna add more loss/activation functions)
        dL_dout = self.loss.backward(predicted, actual)
        
        # Backpropagate through each layer, since we start from the output layer we are going backwards
        for layer in reversed(self.layers):
            dL_dout = layer.backward(dL_dout, learning_rate)    
        
    

        







