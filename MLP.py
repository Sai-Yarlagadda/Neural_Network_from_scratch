import numpy as np
from activations import activation_functions, derivative_activations

class MLP():
    def __init__(self, n_layers, width_layers, activations, initialization):
        self.n_layers = n_layers
        self.width_layers = width_layers

        weights = []
        
        if initialization == 'He':
            for i in range(len(width_layers)-1):
                k = 2/ (width_layers[i])#He Initialization formula
                weight = np.sqrt(k)* np.random.randn(width_layers[i], width_layers[i+1])
                weights.append(weight)
        
        if initialization == 'Xe':
            np.random.seed(0)
            for i in range(len(width_layers)-1):
                k = 1/ (width_layers[i] )#He Initialization formula
                weight = np.sqrt(k) * np.random.randn(width_layers[i], width_layers[i+1])
                weights.append(weight)
        print(weights)
        self.bias = np.ones(n_layers-1)

        derivative_vals = []
        for i in range(len(width_layers)-1):
                derivative = np.zeros((width_layers[i], width_layers[i+1]))
                derivative_vals.append(derivative)
        self.weights = weights
        self.derivatives = derivative_vals
        self.activation = activations #list of activations


    def forward(self, X):
        forward_list = []
        forward_list.append(X)
        activations_list = []
        activation_ = activation_functions()
        for i in range(len(self.weights)):
            passed_input = activation_.Linear(X, weights = self.weights[i], bias= self.bias[i])
            activations_list.append(passed_input)
            activation = self.activation[i]
            try:
                act = getattr(activation_, activation)
                passed_input = act(passed_input)
            except AttributeError:
                passed_input = passed_input
            X = passed_input             
            forward_list.append(passed_input)
        self.forward_list = forward_list
        self.activations_list = activations_list

        return forward_list
        
    
    
    def backward(self, error): #cost_fnc is the error
    
        vals_derivatives = self.derivatives
        activations = self.activation
        activations_list = self.activations_list
        derivative_act = derivative_activations()
        
        for i in reversed(range(len(self.weights))):
            if activations[i] == 'None':
                f_der_act = 1
            else:
                derivative = getattr(derivative_act, activations[i])
                f_der_act = derivative(activations_list[i])
                f_der_act = f_der_act.reshape(f_der_act.shape[0],1)

            delta = np.array(error) * f_der_act
            m = np.array(forward_results[i])
            delta = delta.reshape(delta.shape[0],1)
            m = m.reshape(m.shape[0],1)
            vals_derivatives[i] = np.dot(m, delta.T)
            error = self.weights[i] @ delta
        self.derivatives = vals_derivatives

        return 0
    
X = np.array([2,1,3])
activations = ['sigmoid', 'None']
width = [3,4,2]
n_layers = 3

mlp = MLP(n_layers = n_layers,
          width_layers= width,
          activations= activations,
          initialization = 'Xe')

forward_results = mlp.forward(X)
backprop_test = mlp.backward([0.5,0.5])

