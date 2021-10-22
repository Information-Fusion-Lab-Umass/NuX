import jax
from jax import random, jit
import jax.numpy as jnp

def two_dim_plot_limits(data, border=0.25, xborder=None, yborder=None):
  assert data.shape[-1] == 2, "Only works for 2d data"
  data = data.reshape((-1, 2))
  (xmin, ymin), (xmax, ymax) = data.min(axis=0), data.max(axis=0)
  xspread, yspread = xmax - xmin, ymax - ymin
  if xborder is None:
    xborder = border
  if yborder is None:
    yborder = border
  xmin -= xborder*xspread
  xmax += xborder*xspread
  ymin -= yborder*yspread
  ymax += yborder*yspread
  return xmin, xmax, ymin, ymax

def contour_grid(xmin,
                 xmax,
                 ymin,
                 ymax,
                 n_x,
                 n_y,
                 n_importance_samples=None):
  x_range, y_range = jnp.linspace(xmin, xmax, n_x), jnp.linspace(ymin, ymax, n_y)
  X, Y = jnp.meshgrid(x_range, y_range)
  XY = jnp.dstack([X, Y]).reshape((-1, 2))

  if n_importance_samples is not None:
    XY = jnp.broadcast_to(XY[None,...], (n_importance_samples,) + XY.shape)

  def reshape_to_grid(Z):
    return Z.reshape(X.shape)

  return X, Y, XY, reshape_to_grid

def contour_grid_from_data(data, n_x, n_y, n_importance_samples):
  return contour_grid(*two_dim_plot_limits(data), n_x, n_y, n_importance_samples)

