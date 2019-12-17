from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]
    # D = X.shape[1]
    C = W.shape[1]
    scores = X @ W
    scores -= np.min(scores, axis=1).reshape(scores.shape[0], 1)
    scores = np.exp(scores)
    sums = np.sum(scores, axis=1)
    for i in range(N):
        loss += -np.log(scores[i, y[i]] / sums[i])
        for j in range(C):
            dW[:, j] += X[i] * (scores[i, j] / sums[i])
            if j == y[i]:
                dW[:, j] -= X[i]
    loss /= N
    dW /= N

    loss += reg * np.sum(W ** 2)
    dW += 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]
    scores = X @ W
    scores -= np.max(scores, axis=1).reshape(scores.shape[0], 1)
    scores = np.exp(scores)
    sums = np.sum(scores, axis=1)
    probs = scores[np.arange(N), y] / sums
    allProbs = scores / sums.reshape(sums.size, 1)
    allProbs[np.arange(N), y] -= 1

    loss += np.sum(-np.log(probs))
    loss /= N
    loss += reg * np.sum(W ** 2)

    dW += X.T @ allProbs
    dW /= N;
    dW += 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
