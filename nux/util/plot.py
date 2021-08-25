import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib.path as mpath

def make_segments(x, y):
  """
  Create list of line segments from x and y coordinates, in the correct format
  for LineCollection: an array of the form numlines x (points per line) x 2 (x
  and y) array
  """

  points = np.array([x, y]).T.reshape(-1, 1, 2)
  segments = np.concatenate([points[:-1], points[1:]], axis=1)
  return segments

def colorline(ax, x, y, z=None, colors=None, cmap=plt.get_cmap("inferno"),
              norm=plt.Normalize(0.0, 1.0), linewidth=1, alpha=1.0):
  """
  http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
  http://matplotlib.org/examples/pylab_examples/multicolored_line.html
  Plot a colored line with coordinates x and y
  Optionally specify colors in the array z
  Optionally specify a colormap, a norm function and a line width
  """

  # Default colors equally spaced on [0,1]:
  if z is None:
    z = np.linspace(0.0, 1.0, len(x))

  # Special case if a single number:
  if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
    z = np.array([z])

  z = np.asarray(z)

  segments = make_segments(x, y)
  if colors is not None:
    lc = mcoll.LineCollection(segments, colors=colors,
                              linewidth=linewidth, alpha=alpha)
  else:
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

  ax.add_collection(lc)

  return lc

