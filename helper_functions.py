import numpy as np

def sigmoid(Z):
    """
    Implementation of sigmoid activation in numpy

    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(Z), same shape as Z
    cache -- returns Z as well, to be used when back propogating
    
    """

    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache

def relu(Z):
    """
    Implementation of RELU activation function.
    
    Args:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-Activation parameters, same shape as Z
    cache -- a python dictionary containing "A"; stored for computing the back prop step
    """

    A = np.maximum(0,Z)
    assert(A.shape == Z.shape)

    cache = Z
    return A, cache

def relu_backward(dA, cache):
    """
    Implementation of the back prop for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing back prop efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = cache
    dZ = np.array(dA, copy = True)

    dZ[Z <= 0] = 0 #taking care of z <= 0 case

    assert (dZ.shape == Z.shape)

    return dZ

def sigmoid_backward(dA, cache):
    """
    Implementation of the back prop of a single sigmoid unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)

    return dZ
