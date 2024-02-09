from numpyNN import *
from MLP import MLP
from loss_functions import loss_functions

def main():
    X, y = linearData()
    n_layers = 3
    width = [2, 3, 1]
    activations = ['sigmoid', 'None']
    n_epochs = 100
    mlp = MLP(n_layers=n_layers,
              width_layers=width,
              activations=activations,
              initialization='Xe')

    for epoch in range(n_epochs):
        total_loss = 0
        for i in range(X.shape[0]):
            forward = mlp.forward(X[i])
            final_layer_output = forward[-1]
            loss_fnc = loss_functions(final_layer_output, y[i])
            loss = loss_fnc.cross_entropy_loss()
            total_loss += loss
            mlp.zero_grad
            mlp.backward(loss)
        for i in range(len(mlp.derivatives)):
            mlp.derivatives[i] /= X.shape[0]
        mlp.gradient_descent('vanilla_gradient')
        avg_loss = total_loss / X.shape[0]
        print(f"Epoch {epoch}, Average Loss: {avg_loss}")

if __name__ == "__main__":
    main()
