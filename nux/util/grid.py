import jax
from jax import random, jit
import jax.numpy as jnp
import matplotlib.pyplot as plt

def two_dim_plot_limits(data):
  assert data.shape[-1] == 2, "Only works for 2d data"
  data = data.reshape((-1, 2))
  (xmin, ymin), (xmax, ymax) = data.min(axis=0), data.max(axis=0)
  xspread, yspread = xmax - xmin, ymax - ymin
  xmin -= 0.25*xspread
  xmax += 0.25*xspread
  ymin -= 0.25*yspread
  ymax += 0.25*yspread
  return xmin, xmax, ymin, ymax

def contour_grid(xmin,
                 xmax,
                 ymin,
                 ymax,
                 n_x,
                 n_y,
                 n_importance_samples=None):
  x_range, y_range = jnp.linspace(xmin, xmax, 100), jnp.linspace(ymin, ymax, 100)
  X, Y = jnp.meshgrid(x_range, y_range)
  XY = jnp.dstack([X, Y]).reshape((-1, 2))

  if n_importance_samples is not None:
    XY = jnp.broadcast_to(XY[None,...], (n_importance_samples,) + XY.shape)

  def reshape_to_grid(Z):
    return Z.reshape(X.shape)

  return X, Y, XY, reshape_to_grid

def contour_grid_from_data(data, n_x, n_y, n_importance_samples):
  return contour_grid(*two_dim_plot_limits(data), n_x, n_y, n_importance_samples)

