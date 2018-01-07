import numpy as np
from random import shuffle

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
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in xrange(num_train):
    scores = X[i].dot(W)
    scores -= np.max(scores)
    p = np.exp(scores) / np.sum(np.exp(scores))
    loss += - np.log(p[y[i]])
    
    for j in xrange(num_classes):
      if j == y[i]:
        dW[:, j] -= X[i]
      
      dW[:, j] += p[j] * X[i]
  
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  num_classes = W.shape[1]
  num_train = X.shape[0]

  scores = np.dot(X, W) # [N x C]
  
  # Scores is now an C X N matrix
  scores = scores.T
  scores -= np.max(scores, axis = 0) # Subtracting the max for overflow
  scores = scores.T
  
  exp_scores = np.exp(scores) 
  p = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N X C]
  
  #computing the loss
  loss += -np.sum(np.log(p[range(num_train), y])) / num_train
  loss += 0.5 * reg * np.sum(W * W)

  # gradient on scores
  dscores = p.copy()
  dscores[range(num_train), y] -= 1
  dscores /= num_train
  
  # backprop the gradient to parameters
  dW = np.dot(X.T, dscores)
  dW += reg * W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

