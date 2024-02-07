import numpy as np

class activation_functions():
    def __init__(self):
        pass

    #list of activation functions (Non Linear and Linear)
    def sigmoid(self, X):
        function_val = 1/(1 + np.exp(-1* X))
        

        return function_val

    def Relu(self, X):
        function_val = np.maximum(0, X)
        
        
        return function_val

    def tanh(self, X):
        z1, z2 = np.exp(X), np.exp(-1*X)
        function_val = (z1 - z2)/(z1 + z2)

        return function_val
    
    def Linear(self, X, weights, bias):
        function_val = ((weights.T) @ X) + bias
        return function_val

class derivative_activations():
    def __init__(self):
        pass

    def sigmoid(self, X):
        derivative = (np.exp(-1* X))/np.square((1 - np.exp(-X)))

        return derivative
    
    def Relu(self, X):
        derivative = np.where(X>0, 1, 0)
        
        return derivative
    
    def tanh(self, X):
        z1, z2 = np.exp(X), np.exp(-1*X)
        function_val = (z1 - z2)/(z1 + z2)
        derivative = 1 - np.square(function_val, 2)
        
        return derivative