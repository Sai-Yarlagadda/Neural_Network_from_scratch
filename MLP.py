import numpy as np
from activations import activation_functions, derivative_activations

class MLP():
    def __init__(self, n_layers, width_layers, activations, initialization,
                 lr = 0.01, beta1 = 0.1, beta2 = 0.1, momentum = 0.1, epsilon = 0.000001):
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
        self.bias = np.zeros(n_layers-1)

        derivative_vals = []
        for i in range(len(width_layers)-1):
                derivative = np.zeros((width_layers[i], width_layers[i+1]))
                derivative_vals.append(derivative)
        self.derivatives = derivative_vals

        self.weights = weights
        self.activation = activations #list of activations

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.momentum = momentum
        self.epsilon = epsilon
        self.t = 0

        self.moment1 = derivative_vals #its same as initial_derivative_vals(All zeros)
        self.moment2 = derivative_vals
        #print(f"weights before optimizing: {self.weights}")
    
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
        
        deltas = []
        for i in reversed(range(len(self.weights))):
            if activations[i] == 'None':
                f_der_act = np.ones(self.width_layers[i+1])
            else:
                derivative = getattr(derivative_act, activations[i])
                f_der_act = derivative(activations_list[i])
                f_der_act = f_der_act.reshape(f_der_act.shape[0],1)

            delta = np.array(error) * f_der_act
            m = np.array(self.forward_list[i])
            delta = delta.reshape(delta.shape[0],1)
            m = m.reshape(m.shape[0],1)
            deltas.append(delta)
            vals_derivatives[i] = np.dot(m, delta.T)
            error = self.weights[i] @ delta
        self.derivatives = vals_derivatives 
        #print(self.derivatives)
    
    def gradient_descent(self, optimizer):
        if optimizer == 'vanilla_gradient':
            for i in range(len(self.derivatives)):
                self.weights[i] = np.subtract(self.weights[i], self.lr * self.derivatives[i])
            
        elif optimizer == 'momentum':
            momentum = self.momentum
            gradients = self.derivatives
            for i in range(len(gradients)):
                self.weights[i] = np.add((momentum*self.weights[i]), ((1 - momentum)*gradients[i]))
        elif optimizer == 'adam':
            self.t = self.t + 1
            gradients = self.derivatives
            beta1 = self.beta1
            beta2 = self.beta2
            epsilon = self.epsilon

            for i in range(len(self.weights)):
                moment1 = beta1 * self.moment1[i] + (1-beta1)*gradients[i]
                moment2 = beta2 * self.moment2[i] + (1 - beta2) * (gradients[i] * gradients[i])

                m_hat = moment1/(1 - (beta1 ** (self.t)))
                v_hat = moment2/(1 - (beta2 ** (self.t)))
                self.weights[i] = self.weights[i] - (self.lr*m_hat)/(np.sqrt(v_hat) + epsilon)
        #print(f"Weights after optimizing: {self.weights}")

    def zero_grad(self):
        derivative_vals = []
        for i in range(len(self.width_layers)-1):
                derivative = np.zeros((self.width_layers[i], self.width_layers[i+1]))
                derivative_vals.append(derivative)
        self.derivatives = derivative_vals


'''X = np.array([2,1,3])
activations = ['sigmoid', 'None']
width = [3,4,2]
n_layers = 3

mlp = MLP(n_layers = n_layers,
          width_layers= width,
          activations= activations,
          initialization = 'Xe')

forward_results = mlp.forward(X)
backprop_derivatives, backprop_deltas = mlp.backward([0.8,0.2])
mlp.gradient_descent('adam')'''
