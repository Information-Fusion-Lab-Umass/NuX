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
def tri_solve(chol, x):
    return jax.scipy.linalg.solve_triangular(chol, x, lower=True)

@partial(jit, static_argnums=(1,))
def easy_niw(key, n):
    keys = random.split(key, 2)
    mu = random.normal(keys[0], (n,))
    kappa = 0.2

    psi = random.normal(keys[1], (n, n))
    psi = psi.T@psi + np.eye(n)
    nu = n + 1.0

    return mu, kappa, psi, nu

@partial(jit, static_argnums=(1,))
def easy_niw_nat(key, n):
    return niw_std_to_nat(*easy_niw(key, n))

@jit
def niw_std_to_nat(mu, kappa, psi, nu):
    n = mu.shape[0]
    n1 = -0.5*(kappa*np.outer(mu, mu) + psi)
    n2 = -0.5*kappa
    n3 = kappa*mu
    n4 = -0.5*(nu + n + 2.0)
    return n1, n2, n3, n4

@jit
def niw_nat_to_std(n1, n2, n3, n4):
    n = n3.shape[0]
    psi = -2*n1 + 0.5/n2*np.outer(n3, n3)
    kappa = -2*n2
    mu = -0.5/n2*n3
    nu = -2*n4 - n - 2
    return mu, kappa, psi, nu

@jit
def niw_logZ_from_std(mu, kappa, psi, nu):
    n = mu.shape[0]
    logZ = -0.5*n*np.log(kappa)
    logZ -= 0.5*nu*np.linalg.slogdet(psi)[1]
    logZ += 0.5*n*np.log(2*np.pi)
    logZ += 0.5*nu*n*np.log(2)
    logZ += multigammaln(0.5*nu, n)
    return logZ

@jit
def niw_logZ_from_nat(n1, n2, n3, n4):
    mu, kappa, psi, nu = niw_nat_to_std(n1, n2, n3, n4)
    return niw_logZ_from_std(mu, kappa, psi, nu)

@jit
def niw_expected_stats(n1, n2, n3, n4):
    ret = jit(jax.grad(niw_logZ_from_nat, argnums=(0, 1, 2, 3)))(n1, n2, n3, n4)
    Sigma_inv, mu0TSigma_invmu0, mu0TSigma_inv, logdetSigma = ret
    Sigma_inv = 0.5*(Sigma_inv.T + Sigma_inv)
    return Sigma_inv, mu0TSigma_invmu0, mu0TSigma_inv, logdetSigma

def niw_sample(mu, kappa, psi, nu):
    Sigma = scipy.stats.invwishart.rvs(scale=psi, df=int(nu))
    mu0 = scipy.stats.multivariate_normal.rvs(mean=mu, cov=Sigma/kappa)
    return mu0, Sigma

def niw_logpdf(mu0, Sigma, mu, kappa, psi, nu):
    log_pSigma = scipy.stats.invwishart.logpdf(Sigma, scale=psi, df=int(nu))
    log_pmu0gSigma = scipy.stats.multivariate_normal.logpdf(mu0, mean=mu, cov=Sigma/kappa)
    return log_pmu0gSigma + log_pSigma

@partial(jit, static_argnums=(1, 2))
def easy_mniw(key, n, p):
    keys = random.split(key, 3)
    M = random.normal(keys[0], (n, p))
    V = random.normal(keys[1], (p, p))
    V = V.T@V + np.eye(p)

    psi = random.normal(keys[2], (n, n))
    psi = psi.T@psi + np.eye(n)
    nu = n + p + 1.0

    return M, V, psi, nu

@partial(jit, static_argnums=(1, 2))
def easy_mniw_nat(key, n, p):
    return mniw_std_to_nat(*easy_mniw(key, n, p))

@jit
def mniw_std_to_nat(M, V, psi, nu):
    n, p = M.shape
    V_inv = np.linalg.inv(V)

    n1 = -0.5*(M@V_inv@M.T + psi)
    n2 = -0.5*V_inv
    n3 = V_inv@M.T
    n4 = -0.5*(nu + p + n + 1.0)
    return n1, n2, n3, n4

@jit
def mniw_nat_to_std(n1, n2, n3, n4):
    p, n = n3.shape
    n2_inv = np.linalg.inv(n2)

    psi = -2*n1 + 0.5*n3.T@n2_inv@n3
    V = -0.5*n2_inv
    M = -0.5*n3.T@n2_inv
    nu = -2*n4 - n - p - 1
    return M, V, psi, nu

@jit
def mniw_logZ_from_std(M, V, psi, nu):
    n, p = M.shape
    logZ = 0.5*n*np.linalg.slogdet(V)[1]
    logZ -= 0.5*nu*np.linalg.slogdet(psi)[1]
    logZ += 0.5*n*p*np.log(2*np.pi)
    logZ += 0.5*nu*n*np.log(2)
    logZ += multigammaln(0.5*nu, n)
    return logZ

@jit
def mniw_logZ_from_nat(n1, n2, n3, n4):
    M, V, psi, nu = mniw_nat_to_std(n1, n2, n3, n4)
    return mniw_logZ_from_std(M, V, psi, nu)

@jit
def mniw_expected_stats(n1, n2, n3, n4):
    ret = jit(jax.grad(mniw_logZ_from_nat, argnums=(0, 1, 2, 3)))(n1, n2, n3, n4)
    Sigma_inv, ATSigma_invA, ATSigma_inv, logdetSigma = ret
    Sigma_inv = 0.5*(Sigma_inv.T + Sigma_inv)
    ATSigma_invA = 0.5*(ATSigma_invA.T + ATSigma_invA)
    return Sigma_inv, ATSigma_invA, ATSigma_inv, logdetSigma

# def mniw_nat_from_expected_stats(t1, t2, t3, t4):
#     return n1, n2, n3, n4

def mniw_sample(M, V, psi, nu):
    Sigma = scipy.stats.invwishart.rvs(scale=psi, df=int(nu))
    A = scipy.stats.matrix_normal.rvs(mean=M, rowcov=Sigma, colcov=V)
    return A, Sigma

def mniw_logpdf(A, Sigma, M, V, psi, nu):
    log_pSigma = scipy.stats.invwishart.logpdf(Sigma, scale=psi, df=int(nu))
    log_pAgSigma = scipy.stats.matrix_normal.logpdf(A, mean=M, rowcov=Sigma, colcov=V)
    return log_pAgSigma + log_pSigma

@jit
def logZ_from_std(mu, Sigma):
    Sigma_inv = np.linalg.inv(Sigma)
    logZ = 0.5*np.einsum('i,ij,j', mu, Sigma_inv, mu)
    logZ += 0.5*np.linalg.slogdet(Sigma)[1]
    logZ += 0.5*mu.shape[0]*np.log(2*np.pi)
    return logZ

@jit
def logZ_from_nat(J, h):
    J_inv = np.linalg.inv(J)
    logZ = 0.5*np.einsum('i,ij,j', h, J_inv, h)
    logZ -= 0.5*np.linalg.slogdet(J)[1]
    logZ += 0.5*h.shape[0]*np.log(2*np.pi)
    return logZ

@jit
def gaussian_std_to_nat(mu, Sigma):
    # Not actually the natural paramters,
    # but dropping the -0.5 makes everything nicer
    J = np.linalg.inv(Sigma)
    h = J@mu
    return J, h

@jit
def gaussian_nat_to_std(J, h):
    Sigma = np.linalg.inv(J)
    mu = Sigma@h
    return mu, Sigma

@jit
def gaussian_std_to_nat_vi(E_mu0Sigma0):
    Sigma0_inv, mu0TSigma_invmu0, mu0TSigma_inv, logdetSigma0 = E_mu0Sigma0
    J0 = Sigma0_inv
    h0 = mu0TSigma_inv
    logZ0 = 0.5*mu0TSigma_invmu0
    logZ0 += 0.5*logdetSigma0
    logZ0 += 0.5*h0.shape[0]*np.log(2*np.pi)
    return J0, h0, logZ0

@jit
def regression_joint_std_to_nat(E_ASigma, u):
    Sigma_inv, ATSigma_invA, ATSigma_inv, logdetSigma = E_ASigma
    Sigma_inv_u = Sigma_inv@u

    J11 = Sigma_inv
    J12 = -ATSigma_inv.T
    J22 = ATSigma_invA

    h1 = Sigma_inv_u
    h2 = -np.dot(ATSigma_inv, u)

    logZ = 0.5*np.dot(u, Sigma_inv_u)
    logZ += 0.5*logdetSigma
    logZ += 0.5*u.shape[0]*np.log(2*np.pi)
    return J11, J12, J22, h1, h2, logZ

@jit
def regression_posterior_std_to_nat(E_CR, y):
    R_inv, CTR_invC, CTR_inv, logdetR = E_CR
    R_inv_y = R_inv@y

    J = CTR_invC
    h = CTR_inv@y

    logZ = 0.5*np.dot(y, R_inv_y)
    logZ += 0.5*logdetR
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
    E_mu0Sigma0, E_ASigma, E_CR = params
    Jf, hf, logZf = last_message
    u, y, mask_val = inputs

    Jyt, hyt, logZyt = regression_posterior_std_to_nat(E_CR, y)
    J11, J12, J22, h1, h2, logZ = regression_joint_std_to_nat(E_ASigma, u)
    Jf, hf, logZf = gaussian_integrate_x(J11, J12, J22 + Jf, h1, h2 + hf, logZ + logZf)
    message = (Jf + Jyt*mask_val, hf + hyt*mask_val, logZf + logZyt*mask_val)
    return (params, message), message

@jit
def forward(params, us, ys, mask):
    assert len(us) == len(ys), 'Need to pad the first input of us'
    E_mu0Sigma0, E_ASigma, E_CR = params

    # Base case
    Jy0, hy0, logZy0 = regression_posterior_std_to_nat(E_CR, ys[0])
    J0, h0, logZ0 = gaussian_std_to_nat_vi(E_mu0Sigma0)
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
    E_mu0Sigma0, E_ASigma, E_CR = params

    Jb, hb, logZb = last_message
    u, y, mask_val = inputs

    Jyt, hyt, logZyt = regression_posterior_std_to_nat(E_CR, y)
    J11, J12, J22, h1, h2, logZ = regression_joint_std_to_nat(E_ASigma, u)
    Jb, hb, logZb = gaussian_integrate_y(Jyt*mask_val + J11 + Jb, J12, J22, hyt*mask_val + h1 + hb, h2, logZyt*mask_val + logZ + logZb)
    message = (Jb, hb, logZb)
    return (params, message), message

@jit
def backward(params, us, ys, mask):
    assert len(us) == len(ys), 'Need to pad the first input of us'
    _, E_ASigma, _ = params
    Sigma_inv, _, _, _ = E_ASigma

    # Base case
    Jb0, hb0, logZb0 = np.zeros_like(Sigma_inv), np.zeros(Sigma_inv.shape[0]), np.array(0.0)

    # Recursions
    carry = (params, (Jb0, hb0, logZb0))
    _, (Jbs, hbs, logZbs) = jax.lax.scan(backward_scan_body, carry, (us[1:][::-1], ys[1:][::-1], mask[1:][::-1]))

    # Append the first message
    Jbs = np.concatenate([Jb0[None], Jbs])
    hbs = np.concatenate([hb0[None], hbs])
    logZbs = np.concatenate([logZb0[None], logZbs])

    return Jbs[::-1], hbs[::-1], logZbs[::-1]

@jit
def qx_expected_stats_body_vi(params, forward_messages, backward_messages, next_backward_messages, inputs):
    E_mu0Sigma0, E_ASigma, E_CR = params
    Sigma_inv, ATSigma_invA, ATSigma_inv, logdetSigma = E_ASigma
    R_inv, CTR_invC, CTR_inv, logdetR = E_CR

    Jf, hf, logZf = forward_messages
    Jb, hb, logZb = backward_messages
    Jb_next, hb_next, logZb_next = next_backward_messages

    last_y, y, u, mask_val = inputs

    # Compute E[x_{t}].  These the smoothed means
    Js, hs, logZs = Jf + Jb, hf + hb, logZf + logZb
    Ext, Sigmas = gaussian_nat_to_std(Js, hs)

    # Compute the covariance matrix between x_{t} and x_{t+1}
    Jyt, hyt, logZyt = regression_posterior_std_to_nat(E_CR, y)
    J11, J12, J22, h1, h2, logZ = regression_joint_std_to_nat(E_ASigma, u)
    J_hat = np.block([[Jb_next + Jyt*mask_val + J11, J12     ],
                      [J12.T                       , J22 + Jf]])
    h_hat = np.hstack([hb_next + hyt*mask_val + h1, h2 + hf])

    mu_hat, Sigma_hat = gaussian_nat_to_std(J_hat, h_hat)
    expectations = np.outer(mu_hat, mu_hat) + Sigma_hat

    x_dim = h1.shape[0]
    Extxt = expectations[x_dim:, x_dim:] # Bottom right
    Extxtp1 = expectations[x_dim:,:x_dim] # Bottom left

    # Append the gradient.  Remember that gradient is not needed if mask is False!
    dy = CTR_inv.T@Ext - R_inv@last_y

    return Ext, Extxt, Extxtp1, dy

@jit
def qx_expected_stats_and_emission_grad_vi(params, us, ys, mask):
    T = ys.shape[0]

    Jfs, hfs, logZfs = forward(params, us, ys, mask)
    Jbs, hbs, logZbs = backward(params, us, ys, mask)

    # Get the expected stats of all but the last state
    ret = vmap(partial(qx_expected_stats_body_vi, params))((Jfs[:-1], hfs[:-1], logZfs[:-1]),
                                                           (Jbs[:-1], hbs[:-1], logZbs[:-1]),
                                                           (Jbs[1:], hbs[1:], logZbs[1:]),
                                                           (ys[:-1], ys[1:], us[1:], mask[1:]))
    Ext, Extxt, Extxtp1, dy = ret

    # Do the last step
    mus, Sigmas = gaussian_nat_to_std(Jfs[-1], hfs[-1])
    Ext = np.concatenate([Ext, mus[None]], axis=0)
    Extxt = np.concatenate([Extxt, (np.outer(mus, mus) + Sigmas)[None]], axis=0)

    # Final gradient
    _, _, E_CR = params
    R_inv, CTR_invC, CTR_inv, logdetR = E_CR

    dy = np.concatenate([dy, (CTR_inv.T@mus - R_inv@ys[T-1])[None]])

    # Also compute the log likelihood
    logpy = logZ_from_nat(Jfs[-1], hfs[-1]) - logZfs[-1]

    return Ext, Extxt, Extxtp1, dy, logpy

@jit
def lds(us, masks, params, ys):
    qmu0Sigma0_nat_params, qASigma_nat_params, qCR_nat_params = params

    # Compute the expected sufficient stats of q(mu0, Sigma0), q(A, Sigma) and q(C, R)
    E_mu0Sigma0 = niw_expected_stats(qmu0Sigma0_nat_params)
    E_ASigma = mniw_expected_stats(qASigma_nat_params)
    E_CR = mniw_expected_stats(qCR_nat_params)
    vi_params = (E_mu0Sigma0, E_ASigma, E_CR)

    # Compute the expected sufficient stats of q(x)
    Ext, Extxt, Extxtp1, dy, logpy = qx_expected_stats_and_emission_grad_vi(vi_params, us, ys, mask)
    return logpy, Ext

@jit
def lds_qx_stats(us, mask, params, ys):
    qmu0Sigma0_nat_params, qASigma_nat_params, qCR_nat_params = params

    # Compute the expected sufficient stats of q(mu0, Sigma0), q(A, Sigma) and q(C, R)
    E_mu0Sigma0 = niw_expected_stats(*qmu0Sigma0_nat_params)
    E_ASigma = mniw_expected_stats(*qASigma_nat_params)
    E_CR = mniw_expected_stats(*qCR_nat_params)
    vi_params = (E_mu0Sigma0, E_ASigma, E_CR)

    # Compute the expected sufficient stats of q(x)
    Ext, Extxt, Extxtp1, dy, logpy = qx_expected_stats_and_emission_grad_vi(vi_params, us, ys, mask)
    T = ys.shape[0]

    # Update the natural parameters
    eltwise_add = lambda x, y: tuple(map(np.add, x, y))

    # Update the initial state parameters
    t1 = -0.5*Extxt[0]
    t2 = -0.5
    t3 = Ext[0]
    t4 = -0.5
    qmu0Sigma0_stats = (t1, t2, t3, t4)

    # Update the transition parameters
    vmap_outer = jit(vmap(np.outer))
    t1 = -0.5*Extxt[1:].sum(axis=0)
    t1 += vmap_outer(Ext[1:], us[1:]).sum(axis=0)
    t1 -= 0.5*vmap_outer(us[1:], us[1:]).sum(axis=0)

    t2 = -0.5*Extxt[:-1].sum(axis=0)

    t3 = Extxtp1.sum(axis=0)
    t3 -= vmap_outer(us[1:], Ext[:-1]).sum(axis=0)

    t4 = -0.5*(T - 1)
    qASigma_stats = (t1, t2, t3, t4)

    # Update the emission parameters
    t1 = -0.5*vmap_outer(ys*mask[:,None], ys*mask[:,None]).sum(axis=0)
    t2 = -0.5*Extxt.sum(axis=0)
    t3 = vmap_outer(Ext, ys*mask[:,None]).sum(axis=0)
    t4 = -0.5*T
    qCR_stats = (t1, t2, t3, t4)

    qx_stats = (qmu0Sigma0_stats, qASigma_stats, qCR_stats)
    return Ext, logpy, qx_stats, dy

@partial(custom_jvp, nondiff_argnums=(0, 1, 2, 3, 4, 5))
def lds_svi(us, mask, priors, T, rho, theta, y_batch):
    Ext, logpy, qx_stats, dy = lds_qx_stats(us, mask, theta, y_batch)

    # SVI update
    stat_scale = T/y_batch.shape[0]
    flat_stats, unflatten = ravel_pytree(qx_stats)
    flat_theta, _ = ravel_pytree(theta)
    flat_priors, _ = ravel_pytree(priors)
    flat_updated_theta = (1.0 - rho)*flat_theta + rho*(flat_priors + stat_scale*flat_stats)
    updated_theta = unflatten(flat_updated_theta)

    return logpy, Ext, updated_theta

@lds_svi.defjvp
def lds_svi_jvp(us, mask, priors, T, rho, theta, primals, tangents):
    y_batch, = primals
    y_batch_dot, = tangents
    Ext, logpy, qx_stats, dy = lds_qx_stats(us, mask, theta, y_batch)

    # SVI update
    stat_scale = T/y_batch.shape[0]
    flat_stats, unflatten = ravel_pytree(qx_stats)
    flat_theta, _ = ravel_pytree(theta)
    flat_priors, _ = ravel_pytree(priors)
    flat_updated_theta = (1.0 - rho)*flat_theta + rho*(flat_priors + stat_scale*flat_stats)
    updated_theta = unflatten(flat_updated_theta)

    primals_out = (logpy, Ext, updated_theta)

    # Make dummy outputs for grads of Ext and theta
    dupdated_theta = unflatten(0.0*flat_updated_theta)
    dExt = 0.0*Ext

    dy_batch = np.sum(y_batch_dot*dy)

    tangents_out = (dy_batch, dExt, dupdated_theta)
    return primals_out, tangents_out

"""
FOR EM, NEED TO FIND MAPPING FROM EXPECTED SUFFICIENT STATS TO NATURAL PARAMETERS
"""