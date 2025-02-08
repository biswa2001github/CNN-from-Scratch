import numpy as np
from layer import Layer

class Activation(Layer):
    def __init__(self, activation, activation_derivative):
        super().__init__()
        self.activation = activation
        self.activation_derivative = activation_derivative
        
    def forward(self, input):
        self.input = input
        return self.activation(self.input)
    
    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_derivative(self.input))
    
class sigmoid(Activation):
    def __init__(self):
        sigmoid = lambda x: 1/(1 + np.exp(-x))
        sigmoid_derivative = lambda x: sigmoid(x) * (1 - sigmoid(x))
        super().__init__(sigmoid, sigmoid_derivative)
        
class linear(Activation):
    def __init__(self):
        linear = lambda x: x
        linear_derivative = lambda x: 1
        super().__init__(linear, linear_derivative)