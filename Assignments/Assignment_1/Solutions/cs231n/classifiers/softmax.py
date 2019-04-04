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
  
  for i in range(num_train):
    fx = np.dot(X[i], W)
#     print(fx.shape)
    
    
    lC = -np.amax(fx)
    
    dexp = np.zeros(fx.shape)
    dfx =  np .zeros(fx.shape)
    for j in range(num_classes):
        dexp[j] = np.exp(fx[j]+lC) # Numerical Stability
        dfx[j] = dexp[j]
    dfx = dfx/np.sum(dfx)
    dfx[y[i]] += -1
    
    dW = dW + np.dot(X[np.newaxis, i].T, dfx[np.newaxis,:])
     
    loss += -fx[y[i]] - lC + np.log(np.sum(dexp))
  
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
    
  dW = dW/num_train + reg*2*W
            
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
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.dot(X, W)
 
  scores = (scores.T - np.amax(scores, axis=1)).T
   
  scores_exp = np.exp(scores)
    
  
  den = np.sum(scores_exp, axis=1)
    
  
  loss = -1*scores[np.arange(num_train), y] + np.log(den)
    
  dscores = (scores_exp.T/den).T
    
  dscores[np.arange(num_train),y] += -1
  
  loss = np.sum(loss)/num_train + reg * np.sum(W*W)
  
  dW = np.dot(X.T, dscores)
  dW = dW/num_train + 2 * reg*W
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

