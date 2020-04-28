from jax.flatten_util import ravel_pytree
import jax
import scipy
from jax import random, vmap, jit, grad
ravel_pytree = jit(ravel_pytree)
import jax.numpy as np
from jax.scipy.special import multigammaln
from functools import partial
from jax import custom_jvp

@jit
def cho_solve(chol, x):
    return jax.scipy.linalg.cho_solve((chol, True), x)

@jit
def gaussian_nat_to_std(J, h):
    Sigma = np.linalg.inv(J)
    mu = Sigma@h
    return mu, Sigma

@jit
def logZ_from_nat(J, h):
    J_inv = np.linalg.inv(J)
    logZ = 0.5*np.einsum('i,ij,j', h, J_inv, h)
    logZ -= 0.5*np.linalg.slogdet(J)[1]
    logZ += 0.5*h.shape[0]*np.log(2*np.pi)
    return logZ

@jit
def gaussian_std_to_nat_diag(mu0, Q0):
    if(Q0.ndim == 1):
        J0 = np.diag(1/Q0)
        h0 = mu0/Q0
    else:
        J0 = np.linalg.inv(Q0)
        h0 = J0@mu0

    logZ0 = 0.5*np.dot(h0, mu0)
    if(Q0.ndim == 1):
        logZ0 += 0.5*np.sum(np.log(Q0))
    else:
        logZ0 += 0.5*np.linalg.slogdet(Q0)[1]
    logZ0 += 0.5*h0.shape[0]*np.log(2*np.pi)
    return J0, h0, logZ0

@jit
def regression_joint_std_to_nat(A, B, Q, u):
    if(Q.ndim == 1):
        # Q is diagonal
        AT_Qinv = A.T/Q
    else:
        Q_inv = np.linalg.inv(Q)
        AT_Qinv = A.T@Q_inv

    Bu = B@u

    J11 = np.diag(1/Q) if Q.ndim == 1 else Q_inv
    J12 = -AT_Qinv.T
    J22 = AT_Qinv@A

    h1 = u/Q if Q.ndim == 1 else Q_inv@u
    h2 = -np.dot(AT_Qinv, Bu)

    logZ = 0.5*np.dot(Bu, h1)
    if(Q.ndim == 1):
        # Q is diagonal
        logZ += 0.5*np.sum(np.log(Q))
    else:
        logZ += 0.5*np.linalg.slogdet(Q)[1]
    logZ += 0.5*u.shape[0]*np.log(2*np.pi)
    return J11, J12, J22, h1, h2, logZ

@jit
def regression_posterior_std_to_nat(C, D, R, u, y):
    if(R.ndim == 1):
        # R is diagonal
        CT_Rinv = C.T/R
    else:
        R_inv = np.linalg.inv(R)
        CT_Rinv = C.T@R_inv

    ymDu = y - D@u

    J = CT_Rinv@C
    h = CT_Rinv@ymDu

    if(R.ndim == 1):
        # R is diagonal
        logZ = 0.5*np.dot(ymDu, ymDu/R)
        logZ += 0.5*np.sum(np.log(R))
    else:
        logZ = 0.5*np.dot(ymDu, R_inv@ymDu)
        logZ += 0.5*np.linalg.slogdet(R)[1]

    logZ += 0.5*y.shape[0]*np.log(2*np.pi)
    return J, h, logZ

@jit
def gaussian_integrate_x(J11, J12, J22, h1, h2, logZ):
    J22_chol = np.linalg.cholesky(J22)

    J22_inv_J12T = cho_solve(J22_chol, J12.T)
    J22_inv_h2 = cho_solve(J22_chol, h2)

    Jy = J11 - J12@J22_inv_J12T
    hy = h1 - np.dot(J22_inv_J12T.T, h2)
    logZy = logZ
    logZy -= 0.5*np.dot(h2, J22_inv_h2)
    logZy += np.log(np.diag(J22_chol)).sum()
    logZy -= 0.5*hy.shape[0]*np.log(2*np.pi)

    return Jy, hy, logZy

@jit
def gaussian_integrate_y(J11, J12, J22, h1, h2, logZ):
    return gaussian_integrate_x(J22, J12.T, J11, h2, h1, logZ)

@jit
def forward_scan_body(carry, inputs):
    params, last_message = carry
    (mu0, Q0), (A, B, Q), (C, D, R) = params
    Jf, hf, logZf = last_message
    u, y, mask_val = inputs

    Jyt, hyt, logZyt = regression_posterior_std_to_nat(C, D, R, u, y)
    J11, J12, J22, h1, h2, logZ = regression_joint_std_to_nat(A, B, Q, u)
    Jf, hf, logZf = gaussian_integrate_x(J11, J12, J22 + Jf, h1, h2 + hf, logZ + logZf)
    message = (Jf + Jyt*mask_val, hf + hyt*mask_val, logZf + logZyt*mask_val)
    return (params, message), message

@jit
def forward(params, us, ys, mask):
    assert len(us) == len(ys)
    (mu0, Q0), (A, B, Q), (C, D, R) = params

    # Base case
    Jy0, hy0, logZy0 = regression_posterior_std_to_nat(C, D, R, us[0], ys[0])
    J0, h0, logZ0 = gaussian_std_to_nat_diag(mu0, Q0)
    Jf0, hf0, logZf0 = J0 + Jy0*mask[0], h0 + hy0*mask[0], logZ0 + logZy0*mask[0]

    # Recursions
    carry = (params, (Jf0, hf0, logZf0))
    _, (Jfs, hfs, logZfs) = jax.lax.scan(forward_scan_body, carry, (us[1:], ys[1:], mask[1:]))

    # Append the first message
    Jfs = np.concatenate([Jf0[None], Jfs])
    hfs = np.concatenate([hf0[None], hfs])
    logZfs = np.concatenate([logZf0[None], logZfs])

    return Jfs, hfs, logZfs

@jit
def backward_scan_body(carry, inputs):
    params, last_message = carry
    (mu0, Q0), (A, B, Q), (C, D, R) = params

    Jb, hb, logZb = last_message
    u, y, mask_val = inputs

    Jyt, hyt, logZyt = regression_posterior_std_to_nat(C, D, R, u, y)
    J11, J12, J22, h1, h2, logZ = regression_joint_std_to_nat(A, B, Q, u)
    Jb, hb, logZb = gaussian_integrate_y(Jyt*mask_val + J11 + Jb, J12, J22, hyt*mask_val + h1 + hb, h2, logZyt*mask_val + logZ + logZb)
    message = (Jb, hb, logZb)
    return (params, message), message

@jit
def backward(params, us, ys, mask):
    assert len(us) == len(ys), 'Need to pad the first input of us'
    (mu0, Q0), (A, B, Q), (C, D, R) = params

    # Base case
    Jb0, hb0, logZb0 = np.zeros_like(A), np.zeros((A.shape[0],)), np.array(0.0)

    # Recursions
    carry = (params, (Jb0, hb0, logZb0))
    _, (Jbs, hbs, logZbs) = jax.lax.scan(backward_scan_body, carry, (us[1:][::-1], ys[1:][::-1], mask[1:][::-1]))

    # Append the first message
    Jbs = np.concatenate([Jb0[None], Jbs])
    hbs = np.concatenate([hb0[None], hbs])
    logZbs = np.concatenate([logZb0[None], logZbs])

    return Jbs[::-1], hbs[::-1], logZbs[::-1]

@jit
def kalman_filter_body(forward_messages, backward_messages):
    Jf, hf, logZf = forward_messages
    Jb, hb, logZb = backward_messages

    # Compute E[x_{t}].  These the smoothed means
    Js, hs, logZs = Jf + Jb, hf + hb, logZf + logZb
    mus, Qs = gaussian_nat_to_std(Js, hs)

    return mus, Qs

@jit
def kalman_filter(us, mask, params, ys):
    # This is differentiable!
    (mu0, Q0), (A, B, Q), (C, D, R) = params

    # Run message passing
    Jfs, hfs, logZfs = forward(params, us, ys, mask)
    Jbs, hbs, logZbs = backward(params, us, ys, mask)

    # Get the expected stats of all but the last state
    ret = vmap(kalman_filter_body)((Jfs[:-1], hfs[:-1], logZfs[:-1]),
                                   (Jbs[:-1], hbs[:-1], logZbs[:-1]))
    mus, Qs = ret

    # Do the last step
    mus_T, Qs_T = gaussian_nat_to_std(Jfs[-1], hfs[-1])
    mus = np.concatenate([mus, mus_T[None]], axis=0)
    Qs = np.concatenate([Qs, Qs_T[None]], axis=0)

    # Also compute the log likelihood
    logpy = logZ_from_nat(Jfs[-1], hfs[-1]) - logZfs[-1]

    return mus, Qs, logpy

@jit
def predict_body(carry, inputs):
    params, last_x = carry
    key, u = inputs
    kx, ky = random.split(key, 2)

    (A, B, Q_chol), (C, D, R_chol) = params

    # Sample the next x
    x = A@last_x + B@u + Q_chol@random.normal(kx, last_x.shape)
    y = C@x + D@u + R_chol@random.normal(ky, (C.shape[0],))

    return (params, x), (x, y)

@jit
def predict(us, mask, params, ys, future_us, key):
    (mu0, Q0), (A, B, Q), (C, D, R) = params

    # Compute the initial state for the future
    mus, _, _ = kalman_filter(us, mask, params, ys)

    last_x = mus[-1]
    if(Q.ndim == 1):
        Q_chol = np.sqrt(Q)
        R_chol = np.sqrt(R)
    else:
        Q_chol = np.linalg.cholesky(Q)
        R_chol = np.linalg.cholesky(R)

    T = future_us.shape[0]
    keys = np.array(random.split(key, T))
    carry = (((A, B, Q_chol), (C, D, R_chol)), mus[-1])
    _, (xs, ys) = jax.lax.scan(predict_body, carry, (keys, future_us))
    return xs, ys