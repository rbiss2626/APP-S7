import numpy as np

from dnn_framework.layer import Layer


class FullyConnectedLayer(Layer):
    """
    This class implements a fully connected layer.
    """

    def __init__(self, input_count, output_count):
        self.W = np.random.normal(loc=0.0,
                             scale=np.sqrt(2 / (input_count + output_count)),
                             size=(output_count, input_count))
        
        self.b = np.random.normal(loc=0.0,
                             scale=np.sqrt(2 / output_count),
                             size=(output_count,))

    def get_parameters(self):
        return {"w": self.W, "b": self.b}

    def get_buffers(self):
        raise NotImplementedError()

    def forward(self, x):
        Y = self.W @ x.T
        Y = Y.T + self.b

        return (Y, x)

    def backward(self, output_grad, cache):
        gradX = output_grad @ self.W
        gradW = output_grad.T @ cache
        gradB = np.sum(output_grad, axis=0) #output_grad[0,:] + output_grad[1,:]

        return (gradX, {"w" : gradW, "b": gradB})


class BatchNormalization(Layer):
    """
    This class implements a batch normalization layer.
    """

    def __init__(self, input_count, alpha=0.1):
        self.M = input_count
        self.alpha = alpha

        self.gamma = np.ones(input_count)
        self.beta = np.zeros(input_count)

        self.global_mean = np.zeros((input_count,))
        self.global_variance = np.zeros((input_count,))

    def get_parameters(self):
        return {"gamma": self.gamma, "beta": self.beta}

    def get_buffers(self):
        return {"global_mean": self.global_mean, "global_variance": self.global_variance}

    def forward(self, x):
        mu = np.mean(x)
        sigma = (1/self.M) * np.sum(np.square(x - mu))
        xi = (x - mu) / np.sqrt(sigma + 0.000001)
        return ((self.gamma * xi + self.beta), x)


    def _forward_training(self, x):
        raise NotImplementedError()

    def _forward_evaluation(self, x):
        raise NotImplementedError()

    def backward(self, output_grad, cache):
        raise NotImplementedError()


class Sigmoid(Layer):
    """
    This class implements a sigmoid activation function.
    """

    def get_parameters(self):
        raise NotImplementedError()

    def get_buffers(self):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, output_grad, cache):
        raise NotImplementedError()


class ReLU(Layer):
    """
    This class implements a ReLU activation function.
    """

    def get_parameters(self):
        raise NotImplementedError()

    def get_buffers(self):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, output_grad, cache):
        raise NotImplementedError()
