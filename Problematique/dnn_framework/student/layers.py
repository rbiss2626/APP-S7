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
        gradx = output_grad @ self.W
        gradW = output_grad.T @ cache
        gradB = np.sum(output_grad, axis=0) #output_grad[0,:] + output_grad[1,:]

        return (gradx, {"w" : gradW, "b": gradB})


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
        N = x.shape[0]

        if x.ndim == 2:
            I = x.shape[1]
        else:
            I = 0

        Y = np.zeros(x.shape)

        for j in range(0,N):
            if I > 0:
                for i in range(0,I):
                    Y[j,i] = 1/(1+np.exp(-x[j,i]))
            else:
                Y[j] = 1/(1+np.exp(-x[j]))

        return (Y, x)

    def backward(self, output_grad, cache):
        Y, _ = self.forward(cache)
        dLdsig = output_grad * Y * (1 - Y)
        return (dLdsig, None)


class ReLU(Layer):
    """
    This class implements a ReLU activation function.
    """

    def get_parameters(self):
        raise NotImplementedError()

    def get_buffers(self):
        raise NotImplementedError()

    def forward(self, x):
        N = x.shape[0]

        if x.ndim == 2:
            I = x.shape[1]
        else:
            I = 0

        Y = np.zeros(x.shape)

        for j in range(0,N):
            if I > 0:
                for i in range(0,I):
                    value = x[j,i] 
                    if value >= 0:
                        Y[j,i] = value 
            else:
                value = x[j] 
                if value >= 0:
                    Y[j] = value 

        return (Y,x)

    def backward(self, output_grad, cache):
        N = cache.shape[0]

        if cache.ndim == 2:
            I = cache.shape[1]
        else:
            I = 0

        dLdrelu = np.zeros(cache.shape)

        for j in range(0,N):
            if I > 0:
                for i in range(0,I): 
                    if cache[j,i] >= 0:
                        dLdrelu[j,i] = 1 
            else:
                if cache[j]  >= 0:
                    dLdrelu[j] = 1 

        return (dLdrelu*output_grad, None)
