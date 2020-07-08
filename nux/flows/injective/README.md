# Injective flows
### We have implemented noisy injective flows as described in https://arxiv.org/pdf/2006.13070v1.pdf.
### Every flow has a forward pass of p(x|z) and associated stochastic inverse q(z|x)=p(x|z)/int p(x|z')dz'

# What's implemented?
## nif.py
    - Gaussian NIF
        - p(x|z) = N(z|Az + b, Sigma)
    - Coupling Gaussian NIF
        - p(x1,x2|z1,z2) = N(x1|A1@z1+b(x2),\Sigma(x2))N(x2|A2@z2+b(z1),\Sigma(z1))

## upsample.py
    - Nearest Neighbors NIF
        - p(x|z) = N(x|upsample(z) + b, Sigma)
    - Coupling Nearest Neighbors NIF
        - Same idea as coupling from before

## importance_weighted.py
    - Importance weighted estimates of values associated with NIFs and its gradient.