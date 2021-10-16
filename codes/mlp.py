import sys

import numpy as np
import matplotlib.pyplot as plt
from random import random


class MLP(object):
    def __init__(self, num_inputs=3, hidden_layers=None, num_outputs=2):
        if hidden_layers is None:
            hidden_layers = [3, 3]
        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        layers = [num_inputs] + hidden_layers + [num_outputs]

        # create random connection weights for the layers
        weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            weights.append(w)
        self.weights = weights

        # activations are the outputs of every layer, activations[0] = x
        activations = []
        for i in range(len(layers)):
            a = np.zeros(shape=(layers[i]))
            activations.append(a)
        self.activations = activations

        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros(shape=(layers[i], layers[i + 1]))
            derivatives.append(d)
        self.derivatives = derivatives

    def forward_propagate(self, inputs):

        activations = inputs
        self.activations[0] = inputs

        for i, w in enumerate(self.weights):
            net_inputs = np.dot(activations, w)
            activations = self._sigmoid(net_inputs)
            self.activations[i + 1] = activations

        # return output layer activation
        return activations

    def back_propagate(self, error, verbose=False):
        """
        loop, start from the last one
        dE/dW_i = (y - a_[i+1])

        :param verbose:
        :param error:
        :return:
        """

        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i + 1]
            delta = error * self._sigmoid_derivative(activations)
            delta_reshaped = delta.reshape(delta.shape[0], -1).T
            current_activations = self.activations[i]
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)
            self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)
            error = np.dot(delta, self.weights[i].T)

            if verbose:
                print("Derivatives for W{}: {}".format(i, self.derivatives[i]))

        return error

    def gradient_descent(self, learning_rate):

        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learning_rate

    def train(self, inputs, targets, epochs, learning_rate):

        for i in range(epochs):
            sum_error = 0
            for input, target in zip(inputs, targets):
                # forward propagation
                output = self.forward_propagate(inputs=input)
                # calculate the error
                error = target - output
                # backward
                self.back_propagate(error=error)
                # apply gradient descent
                self.gradient_descent(learning_rate)
                sum_error = sum_error + self._mse(target, output)
            # report error
            print("Error: {} at epoch {}".format((sum_error / len(inputs)), i))

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)

    def _mse(self, target, output):
        return np.average((target - output) ** 2)


def main():
    mlp = MLP(num_inputs=1, hidden_layers=[3], num_outputs=1)

    inputs = np.linspace(-5, 5, 1000)
    inputs = np.expand_dims(inputs, axis=1)
    targets = 1 / (1 + np.exp(-inputs))
    plt.plot(inputs, targets)
    # plt.show()

    input_ = [1.2]
    label_ = 1 / (1 + np.exp(-1.2))
    print(label_)

    mlp.train(inputs=inputs, targets=targets, epochs=50, learning_rate=0.1)

    output = mlp.forward_propagate(input_)
    print(output)

    #
    y = []
    for x in inputs:
        output = mlp.forward_propagate(x)
        y.append(output)
    plt.plot(inputs, y)
    plt.show()

    '''
    '''

    sys.exit()
    # create an mlp
    mlp = MLP(num_inputs=2, hidden_layers=[2, 2], num_outputs=1)

    # create dataset for training
    inputs = np.array([[random() / 2 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in inputs])

    # train our mlp
    mlp.train(inputs=inputs, targets=targets, epochs=50, learning_rate=0.1)

    # create dummy data
    inputs = np.array([0.16, 0.63])
    target = np.array([0.4])

    output = mlp.forward_propagate(inputs)
    print("Our network believes that {} + {} is equal to {}".format(inputs[0], inputs[1], output[0]))


if __name__ == '__main__':
    main()
