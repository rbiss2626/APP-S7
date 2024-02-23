import numpy as np

from dnn_framework.loss import Loss


class CrossEntropyLoss(Loss):
    """
    This class combines a softmax activation function and a cross entropy loss.
    """

    def calculate(self, x, target):
        """
        :param x: The input tensor (shape: (N, C))
        :param target: The target classes (shape: (N,))
        :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """
        loss = []
        grad = []

        for i in range(x.shape[0]):
            sx = softmax(x[i])
            y = np.zeros(x.shape[1])
            y[target[i]] = 1
            loss.append(-np.sum(y * np.log(sx + 0.00000001)))
            grad.append((sx - y)/x.shape[0])
        
        global_loss = np.mean(loss)
        return (global_loss, grad)


def softmax(x):
    """
    :param x: The input tensor (shape: (N, C))
    :return The softmax of x
    """
    e_x = np.exp(x)
    return e_x / e_x.sum()


class MeanSquaredErrorLoss(Loss):
    """
    This class implements a mean squared error loss.
    """

    def calculate(self, x, target):
        """
        :param x: The input tensor (shape: any)
        :param target: The target tensor (shape: same as x)
        :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """
        return (np.mean((x - target)**2), (2*(x - target))/x.size)
