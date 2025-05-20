# scratch-neural-network
This is an implementation of a feed-forward neural network (aka multi-layer perceptron) by hand, ie without referring to in-built neural network functions in Pytorch. This is mostly so that I learn, but also serves as a good way to compare in-built implementations to theoretical implementations. The motivation for this is mostly Feynman's quote: "What I cannot create, I do not understand" which should be applied rather conservatively (no, you do not have to recreate everything you work with), but makes the most sense when applied to foundational ideas (such as neural networks). 

As of right now, this implementation supports ReLU activation functions for the layers, and only SoftMax on the output layer, because the backpropagation method for the final layer right now assumes we will be using Cross-Entropy (which simmplifies the derivative a lot) but the way the code is written you can add in any activation function you want, you just have to add an `activation.apply()` and `activation.backward()` method to the class. 

It also does batch gradient descent, but again you can very easily change the `nn.train()` method to implement whatever optimizer you want. 

For now, the only loss function supported is the CrossEntropy loss, so this nn implementation is limited to classification problems I guess. The training data is in the `training.csv` file whereas the test data is in the `test.csv` file (taken from the kaggle for MNIST). The code is all in `neuralnetwork.py`. I did this to learn how Pytorch actually works deep down.  
