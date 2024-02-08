import numpy as np
class gradient_descent():
    def __init__(self, weights, gradient, lr, beta1, beta2, momentum, epsilon, t):
        self.weights = weights
        self.gradients = gradient_descent
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.momentum = momentum
        self.epsilon = epsilon
        self.t = 0

        moment1 = []
        moment2 = []
        for i in range(len(self.weights)):
            m = np.array(self.weights[i])
            m = np.zeros_like(m)
            moment1.append(m)
            moment2.append(m)
        moment1 = np.array(moment1)
        moment2 = np.array(moment2)

        self.moment1 = moment1
        self.moment2 = moment2

    def vanilla_gradient(self):
        weights = self.weights
        derivatives = self.gradients
        lr = self.lr
        weights = np.subtract(weights, lr*derivatives)

        return weights
    
    def momentum(self):
        momentum = self.momentum 
        gradients = self.gradients
        weights = np.add((momentum * (weights)),((1- momentum)*gradients))

        return weights
    
    def adam(self):
        # wt = weights or parameters of the model
        # lr = learning rate of the model
        # beta1, beta2 = decay rates of first and second moments
        # s = first moment estimate - mean of gradients
        # v = second moment estimate - variance of gradients
        
        self.t = self.t + 1
        weights = self.weights
        gradients = self.gradients
        lr = self.lr
        beta1 = self.beta1
        beta2 = self.beta2
        moment1 = self.moment1
        moment2 = self.moment2
        epsilon = self.epsilon

        for i in range(len(weights)):
            self.moment1 = beta1 * moment1[i] + (1- beta1)* gradients[i]
            self.moment2 = beta2 * moment2[i] + (1- beta2) * (gradients[i] * gradients[i])

            m_hat = self.moment1/ (1 - (beta1 ** (self.t)))
            v_hat = self.moment2/ (1 - (beta2 ** (self.t)))

            weights[i] =  weights[i] - lr/(np.sqrt(v_hat) + epsilon)

        return weights
