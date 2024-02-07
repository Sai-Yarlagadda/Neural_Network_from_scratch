import numpy as np
class init_weights():
    def __init__(self, width_layers):
        self.width_layers = width_layers
        self.weights = 0
        self.bias = np.ones(len(width_layers))

    def He(self):
        weights = []
        for i in range(len(self.width_layers)-1):
                k = 2/ (self.width_layers[i])#He Initialization formula
                weight = np.sqrt(k)* np.random.randn(self.width_layers[i], self.width_layers[i+1])
                weights.append(weight)
        return weights
    
    def Xa(self):
        weights = []
        for i in range(len(self.width_layers)-1):
                k = 1/ (self.width_layers[i] )#He Initialization formula
                weight = np.sqrt(k) * np.random.randn(self.width_layers[i], self.width_layers[i+1])
                weights.append(weight)
        return weights
    