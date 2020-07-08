# Bijective flows
### These are the standard normalizing flows that adhere to the change of variable formula:
log p(x) = log p(f^{-1}(x)) + log |df^{-1}(x)/dx|

# What's implemented?
## affine.py
    - Identity transformation
        - x = z

    - Affine
        - LDU parameterization.  Slightly different from GLOW's LU https://arxiv.org/pdf/1807.03039.pdf
            - A = LDU
        - SVD parametrization.
            - A = USV^T
            - U and V are parametrized by householders https://arxiv.org/pdf/1611.09630.pdf
        - Standard dense.

    - 1x1 convolution - https://arxiv.org/pdf/1807.03039.pdf
        - As matrix multiplication across channel of individual pixels.
            - LDU, SVD, standard parametrization
        - Using a convolution

    - Local Dense
        - This generalizes the 1x1 convolution and is equivalent to matrix multiplication across the channel dimension of patches of pixels.  Best performance seems to come using the 1x1 convolution special case.

## conv.py
    - Circular convolution - https://papers.nips.cc/paper/8801-invertible-convolutional-flow.pdf

## coupling.py
    - Affine/additive coupling from RealNVP - https://arxiv.org/pdf/1605.08803.pdf
    - Uses chain rule to create complex non-linear transformations:
        - p(x,y) = p(x|y)p(y) = p(x|NN(y))p(y) = p(f^{-1}(x)|NN(y))|df^{-1}(x)/dx|p(y)

### igr.py
    - Invertible Gaussian Reparametrization - https://arxiv.org/pdf/1912.09588.pdf
    - The paper's log determinant is incorrect!
    - Transform from R^N -> interior of N simplex.

### maf.py
    - Masked autoregressive flow - https://arxiv.org/pdf/1705.07057.pdf
    - Efficient extension of coupling with full chain rule factorization

### nonlinearities.py
    - Leaky-Relu
    - Sigmoid and Logit
        - Both accept parameter to control domain/range.
        - If using Logit for dequantization in an image problem, make sure to set lmbda to a non-zero value to avoid huge numbers.

### normalization.py
    - Actnorm - https://arxiv.org/pdf/1807.03039.pdf
        - Data-dependent initialization is done automatically when you initialize a flow!

    - Batchnorm - https://arxiv.org/pdf/1605.08803.pdf
        - Actnorm seems to always work better.

### reshape.py
    - Squeeze - https://arxiv.org/pdf/1605.08803.pdf
        - Basically stacks 4 downsampled versions of an image on top of each other over the channel dimension.
    - UnSqueeze
    - Transpose
    - Reshape
    - Flatten
    - Reverse
        - Reverse the last dimension of an input.  Use this with coupling!

### spline.py
    - NeuralSpline - https://arxiv.org/pdf/1906.04032.pdf