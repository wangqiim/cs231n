from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = x.shape[0]
    ndim = int(np.prod(x.shape) // num_train)
    out_dim = int(np.prod(w.shape) // ndim)
    # print(f"ndim = {ndim}, out_dim = {out_dim}")
    
    x_reshape = np.reshape(x, (num_train, ndim))
    w_reshape = np.reshape(w, (ndim, out_dim))
    out = np.dot(x_reshape, w_reshape) + b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = dout.shape[0]
    out_dim = dout.shape[1]
    x_reshaped = x.reshape(num_train, -1)
    w_reshaped = w.reshape(-1, out_dim)
    
    # dx = dout.dot(w.T)  # Shape (N, D)
    # dx = dx.reshape(x.shape)  # Reshape back to original input shape
    
    dx = dout.dot(w_reshaped.T)
    dx.resize(x.shape)
    
    dw = x_reshaped.T.dot(dout)  # Shape (D, M)

    db = np.sum(dout, axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = x * (x > 0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = dout * (x > 0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from A1.
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = x.shape[0]
    assert num_train != 0, 'Problem num_train is zero'
    scores = x - np.max(x, axis=1, keepdims=True)  # 减去最大值防止指数溢出
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # P = e^s / sum(e^s)

    # 裁剪概率，确保 log(p) 不爆炸
    # probs = np.clip(probs, 1e-10, 1.0)  # 下限设为1e-10
    loss = np.sum(-np.log(probs[np.arange(num_train), y])) / num_train

    dprobs = probs.copy()
    dprobs[np.arange(num_train), y] -= 1  # 正确类别的梯度 -= 1
    dx = dprobs / num_train
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx

def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Step 1: Compute mean
        sample_mean = np.mean(x, axis=0)
        
        # Step 2: Compute variance
        sample_var = np.var(x, axis=0)
        
        # Step 3: Normalize the data
        x_normalized = (x - sample_mean) / np.sqrt(sample_var + eps)
        
        # Step 4: Scale and shift
        out = gamma * x_normalized + beta
        
        # Update running mean and variance
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        
        # Store values needed for backward pass
        cache = (x, x_normalized, sample_mean, sample_var, gamma, beta, eps)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Normalize using running statistics
        x_normalized = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_normalized + beta
        
        # No need to store cache for test mode
        cache = None

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache

def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    (x, x_normalized, sample_mean, sample_var, gamma, beta, eps) = cache
    num_train = x.shape[0]
    
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_normalized, axis=0)
    
    # 0. 计算中间梯度
    dx_normalized = dout * gamma
    std_inv = 1.0 / np.sqrt(sample_var + eps)
    x_minus_mean = x - sample_mean
    # 1. 计算方差梯度（隐含耦合项）
    dvar = np.sum(dx_normalized * x_minus_mean * (-0.5) * std_inv**3, axis=0)
    # 2. 计算均值梯度（隐含耦合项）
    dmean = np.sum(dx_normalized * (-std_inv), axis=0) + dvar * np.mean(-2 * x_minus_mean, axis=0)
    # 3. 合并梯度
    dx = dx_normalized * std_inv + dvar * (2/num_train) * x_minus_mean + dmean / num_train
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N = dout.shape[0]
    (x, x_normalized, sample_mean, sample_var, gamma, beta, eps) = cache
    
    dx = (1. / N) * gamma * (sample_var + eps)**(-1./2) * (
        N * dout - np.sum(dout, axis=0) - x_normalized * np.sum(dout * x_normalized, axis=0))
    
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_normalized, axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N = x.shape[0]
    D = x.shape[1]

    sample_mean = np.mean(x, axis=1)
    sample_var = np.var(x, axis=1)
    x_normalized = (x - sample_mean.reshape(N, 1)) / np.sqrt(sample_var.reshape(N, 1) + eps)
    out = gamma * x_normalized + beta

    cache = (x, x_normalized, sample_mean, sample_var, gamma, beta, eps)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, D = dout.shape

    (x, x_normalized, sample_mean, sample_var, gamma, beta, eps) = cache

    # 0. 计算中间梯度
    dx_normalized = dout * gamma
    std_inv = (1.0 / np.sqrt(sample_var + eps)).reshape(N, 1).repeat(D, axis=1)
    x_minus_mean = x - sample_mean.reshape(N, 1).repeat(D, axis=1)
    # 1. 计算方差梯度（隐含耦合项）
    dvar = np.sum(dx_normalized * x_minus_mean * (-0.5) * std_inv**3, axis=1).reshape(N, 1).repeat(D, axis=1)
    # 2. 计算均值梯度（隐含耦合项）
    dmean = np.sum(dx_normalized * (-std_inv), axis=1).reshape(N, 1).repeat(D, axis=1) + dvar * np.mean(-2 * x_minus_mean, axis=1).reshape(N, 1).repeat(D, axis=1)
    # 3. 合并梯度
    dx = dx_normalized * std_inv
    dx += dvar * (2/D) * x_minus_mean
    dx += dmean / D

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_normalized, axis=0)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask  # Apply mask to input data

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dx = dout * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    stride = conv_param["stride"]
    pad = conv_param["pad"]
    
    N, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    F, HH, WW = w.shape[0], w.shape[2], w.shape[3]
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride
    
    out = np.zeros((N, F, H_out, W_out))
    x_pad = np.pad(x, pad_width=((0, 0), (0, 0), (pad, pad), (pad, pad)))

    for n in range(N):
      for f in range(F):
        for h_out in range(H_out):
          for w_out in range(W_out):
            image_start_h = h_out * stride
            image_start_w = w_out * stride
            value = 0
            for i in range(HH):
              for j in range(WW):
                for z in range(C):
                  # print(f"w[{f}][{z}][{i}][{j}] * x_pad[{n}][{z}][{image_start_h + i}][{image_start_w + j}]")
                  value += w[f][z][i][j] * x_pad[n][z][image_start_h + i][image_start_w + j]
            value += b[f]
            out[n][f][h_out][w_out] = value
                  
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    (x, w, b, conv_param) = cache
    stride = conv_param["stride"]
    pad = conv_param["pad"]
    N, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    F, HH, WW = w.shape[0], w.shape[2], w.shape[3]
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride
    
    x_pad = np.pad(x, pad_width=((0, 0), (0, 0), (pad, pad), (pad, pad)))
    
    dx = np.zeros(x.shape)
    dw = np.zeros(w.shape)
    for n in range(N):
      for f in range(F):
        for h_out in range(H_out):
          for w_out in range(W_out):
            image_start_h = h_out * stride
            image_start_w = w_out * stride
            for i in range(HH):
              for j in range(WW):
                for z in range(C):
                  # print(f"w[{f}][{z}][{i}][{j}] * x_pad[{n}][{z}][{image_start_h + i}][{image_start_w + j}]")
                  dw[f][z][i][j] += x_pad[n][z][image_start_h + i][image_start_w + j] * dout[n][f][h_out][w_out]
                  x_without_padding_i = image_start_h + i - pad
                  y_without_padding_j = image_start_w + j - pad
                  if x_without_padding_i >= 0 and x_without_padding_i < H and y_without_padding_j >= 0 and y_without_padding_j < W:
                    dx[n][z][x_without_padding_i][y_without_padding_j] += w[f][z][i][j] * dout[n][f][h_out][w_out]
    
    db = np.sum(dout, axis=(0, 2, 3))
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pool_width = pool_param['pool_width']
    pool_height = pool_param['pool_height']
    stride = pool_param['stride']
    N, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    H_out = int(1 + (H - pool_height) / stride)
    W_out = int(1 + (W - pool_width) / stride)

    # print(x.shape)
    out = np.zeros((N, C, H_out, W_out))
    for n in range(N):
      for c in range(C):
        for h in range(H_out):
          for w in range(W_out):
            value = -float("inf")
            for i in range(pool_height):
              for j in range(pool_width):
                # print(f"x[{n}][{c}][{i + h * stride}][{j + w * stride}]")
                value = max(value, x[n][c][i + h * stride][j + w * stride])
            out[n][c][h][w] = value
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, pool_param = cache
    pool_width = pool_param['pool_width']
    pool_height = pool_param['pool_height']
    stride = pool_param['stride']
    N, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    H_out = int(1 + (H - pool_height) / stride)
    W_out = int(1 + (W - pool_width) / stride)
    
    dx = np.zeros(x.shape)

    for n in range(N):
      for c in range(C):
        for h in range(H_out):
          for w in range(W_out):
            value = -float("inf")
            max_v_i = 0
            max_v_j = 0
            for i in range(pool_height):
              for j in range(pool_width):
                if x[n][c][i + h * stride][j + w * stride] > value:
                  value = x[n][c][i + h * stride][j + w * stride]
                  max_v_i = i
                  max_v_j = j
            dx[n][c][max_v_i + h * stride][max_v_j + w * stride] = dout[n][c][h][w]


    pass
                
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    
    # Reshape x to (N*H*W, C) for standard batch normalization
    x_reshaped = x.transpose(0, 2, 3, 1).reshape(-1, C)
    # Call standard batch normalization
    
    out_reshaped, cache = batchnorm_forward(x_reshaped, gamma, beta, bn_param)
    
    # Reshape back to (N, C, H, W)
    out = out_reshaped.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    N, C, H, W = dout.shape
    dout = dout.transpose(0, 2, 3, 1).reshape(-1, C)
    dx, dgamma, dbeta = batchnorm_backward(dout, cache)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    # Step 1: Reshape x to (N, G, C//G, H, W)
    x_reshaped = x.reshape(N, G, C // G, H, W)
    
    # Step 2: Compute mean and variance for each group
    mean = x_reshaped.mean(axis=(2, 3, 4), keepdims=True)  # Shape (N, G, 1, 1, 1)
    var = x_reshaped.var(axis=(2, 3, 4), keepdims=True)    # Shape (N, G, 1, 1, 1)
    
    # Step 3: Normalize each group
    x_normalized = (x_reshaped - mean) / np.sqrt(var + eps)
    
    # Reshape back to (N, C, H, W) for scaling/shifting
    x_normalized = x_normalized.reshape(N, C, H, W)
    
    # Step 4: Apply gamma and beta (broadcast from (1,C,1,1) to (N,C,H,W))
    out = gamma * x_normalized + beta
    
    # Cache for backward pass
    cache = {
        'x_normalized': x_normalized,  # After normalization, before gamma/beta
        'mean': mean,                  # Group means
        'var': var,                    # Group variances
        'gamma': gamma,                # Scale parameter
        'beta': beta,                  # Shift parameter
        'eps': eps,                    # Numerical stability term
        'G': G,                        # Number of groups
        'x_shape': (N, C, H, W)        # Original input shape
    }

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

def spatial_groupnorm_backward(dout, cache):
    """
    Backward pass for spatial group normalization.
    
    Inputs:
    - dout: Upstream derivatives of shape (N, C, H, W)
    - cache: Values from the forward pass (x_normalized, mean, var, gamma, beta, G, etc.)
    
    Returns:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter beta, of shape (1, C, 1, 1)
    """
    # 从缓存中提取前向传播的中间变量
    x_normalized = cache['x_normalized']  # 归一化后的数据 (N, C, H, W)
    mean = cache['mean']                  # 各组均值 (N, G, 1, 1, 1)
    var = cache['var']                    # 各组方差 (N, G, 1, 1, 1)
    gamma = cache['gamma']                # 缩放参数 (1, C, 1, 1)
    beta = cache['beta']                  # 偏移参数 (1, C, 1, 1)
    G = cache['G']                        # 分组数
    eps = cache['eps']                    # 数值稳定项
    N, C, H, W = cache['x_shape']         # 输入形状
    
    # 初始化梯度
    dx = np.zeros((N, C, H, W))
    dgamma = np.zeros_like(gamma)          # (1, C, 1, 1)
    dbeta = np.zeros_like(beta)            # (1, C, 1, 1)
    
    # Step 1: 计算 dbeta 和 dgamma（直接求和）
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)  # (1, C, 1, 1)
    dgamma = np.sum(dout * x_normalized, axis=(0, 2, 3), keepdims=True)  # (1, C, 1, 1)
    
    # Step 2: 计算 dx_normalized（上游梯度乘以 gamma）
    dx_normalized = dout * gamma          # (N, C, H, W)
    
    # Step 3: 将 dx_normalized 分组为 (N, G, C//G, H, W)
    dx_normalized_group = dx_normalized.reshape(N, G, C//G, H, W)
    x_group = x_normalized.reshape(N, G, C//G, H, W)
    
    # Step 4: 计算每组的标准差倒数 (std_inv = 1/sqrt(var + eps))
    std_inv = 1.0 / np.sqrt(var + eps)    # (N, G, 1, 1, 1)
    
    # Step 5: 计算 dx_group（链式法则）
    # dx_group = (dx_normalized_group - mean(dx_normalized_group) - x_group * mean(dx_normalized_group * x_group)) * std_inv
    dx_group = std_inv * (dx_normalized_group - np.mean(dx_normalized_group, axis=(2, 3, 4), keepdims=True) 
                          - x_group * np.mean(dx_normalized_group * x_group, axis=(2, 3, 4), keepdims=True))
    
    # Step 6: 恢复原始形状 (N, C, H, W)
    dx = dx_group.reshape(N, C, H, W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
