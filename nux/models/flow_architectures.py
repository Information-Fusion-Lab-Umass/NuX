import nux
from typing import Optional, Mapping, Callable, Sequence
from functools import partial

__all__ = ["multiscale_architecture",
           "FlatLogisticCDFMixtureFlow",
           "RealNVPModel",
           "GLOW",
           "FlowPlusPlus"]

################################################################################################################

def build_architecture(architecture: Sequence[Callable],
                       coupling_algorithm: Callable,
                       actnorm: bool=False,
                       actnorm_axes: Sequence[int]=-1,
                       glow: bool=True,
                       one_dim: bool=False,
                       factor: Optional[Callable]=nux.multi_scale):
  n_squeeze = 0

  layers = []
  for i, layer in list(enumerate(architecture)):

    # We don't want to put anything in front of the squeeze
    if layer == "sq":
      layers.append(nux.Squeeze())
      if glow and one_dim == False:
        layers.append(nux.OneByOneConv())
      n_squeeze += 1
      continue

    # We don't want to put anything in front of the unsqueeze
    if layer == "unsq":
      layers.append(nux.UnSqueeze())
      n_squeeze -= 1
      continue

    # Should we do a multiscale factorization?
    if layer == "ms":
      # Recursively build the multiscale
      inner_flow = build_architecture(architecture=architecture[i + 1:],
                                      coupling_algorithm=coupling_algorithm,
                                      actnorm=actnorm,
                                      actnorm_axes=actnorm_axes,
                                      glow=glow,
                                      one_dim=one_dim)
      layers.append(factor(inner_flow))
      break

    # Actnorm.  Not needed if we're using 1x1 conv because the 1x1
    # conv is initialized with weight normalization so that its outputs
    # have 0 mean and 1 stddev.
    if actnorm:
      layers.append(nux.ActNorm(axis=actnorm_axes))

    # Use a dense connection instead of reverse?
    if glow:
      if one_dim:
        layers.append(nux.AffineLDU())
      else:
        layers.append(nux.OneByOneConv())
    else:
      layers.append(nux.Reverse())

    # Create the layer
    if layer == "chk":
      alg = coupling_algorithm(split_kind="checkerboard")
    elif layer == "chnl":
      alg = coupling_algorithm(split_kind="channel")
    else:
      assert 0
    layers.append(alg)

  # Remember to unsqueeze so that we end up with the same shaped output
  for i in range(n_squeeze):
    layers.append(nux.UnSqueeze())

  return nux.sequential(*layers)

################################################################################################################

def FlatLogisticCDFMixtureFlow(n_components=32,
                               n_blocks=4,
                               masked_coupling=False,
                               apply_transform_to_both_halves=False,
                               network_kwargs=None,
                               create_network=None):

  coupling_algorithm = partial(nux.LogisticMixtureLogit,
                               n_components=n_components,
                               with_affine_coupling=True,
                               masked=masked_coupling,
                               apply_to_both_halves=apply_transform_to_both_halves,
                               network_kwargs=network_kwargs,
                               create_network=create_network)

  architecture = ["chnl"]*n_blocks

  return build_architecture(architecture,
                            coupling_algorithm,
                            actnorm=True,
                            actnorm_axes=(-1,),
                            glow=True,
                            one_dim=True)

################################################################################################################

def multiscale_architecture(coupling_algorithm,
                            n_checkerboard_splits=3,
                            n_channel_splits=3,
                            n_scales=1,
                            actnorm=True,
                            actnorm_axes=-1,
                            glow=True):
  assert n_scales > 0, "Use n_scales=1 to not have any multiscale factors"

  architecture = ["chk"]*n_checkerboard_splits + ["sq"] + ["chnl"]*n_channel_splits
  architecture = (architecture + ["ms"])*n_scales
  architecture = architecture[:-1]

  return build_architecture(architecture,
                            coupling_algorithm,
                            actnorm=actnorm,
                            actnorm_axes=actnorm_axes,
                            glow=glow,
                            one_dim=False)

def RealNVPModel(n_checkerboard_splits=3,
            n_channel_splits=3,
            n_scales=3,
            masked_coupling=False,
            apply_transform_to_both_halves=False,
            network_kwargs=None,
            create_network=None,
            one_dim=False):

  coupling_algorithm = partial(nux.RealNVP,
                               kind="affine",
                               masked=masked_coupling,
                               apply_to_both_halves=apply_transform_to_both_halves,
                               network_kwargs=network_kwargs,
                               create_network=create_network)

  return multiscale_architecture(coupling_algorithm,
                                 n_checkerboard_splits=n_checkerboard_splits,
                                 n_channel_splits=n_channel_splits,
                                 n_scales=n_scales,
                                 actnorm=False,
                                 glow=False)

def GLOW(n_checkerboard_splits=3,
         n_channel_splits=3,
         n_scales=3,
         masked_coupling=False,
         apply_transform_to_both_halves=False,
         network_kwargs=None,
         create_network=None,
         one_dim=False):

  coupling_algorithm = partial(nux.RealNVP,
                               kind="affine",
                               masked=masked_coupling,
                               apply_to_both_halves=apply_transform_to_both_halves,
                               network_kwargs=network_kwargs,
                               create_network=create_network)

  return multiscale_architecture(coupling_algorithm,
                                 n_checkerboard_splits=n_checkerboard_splits,
                                 n_channel_splits=n_channel_splits,
                                 n_scales=n_scales,
                                 actnorm=True,
                                 glow=True)

################################################################################################################

def FlowPlusPlus(n_components=32,
                 n_checkerboard_splits_before=4,
                 n_channel_splits=2,
                 n_checkerboard_splits_after=3,
                 masked_coupling=False,
                 apply_transform_to_both_halves=False,
                 network_kwargs=None,
                 create_network=None,
                 one_dim=False):

  coupling_algorithm = partial(nux.LogisticMixtureLogit,
                               n_components=n_components,
                               with_affine_coupling=True,
                               masked=masked_coupling,
                               apply_to_both_halves=apply_transform_to_both_halves,
                               network_kwargs=network_kwargs,
                               coupling=True,
                               create_network=create_network)

  architecture = ["chk"]*n_checkerboard_splits_before

  if one_dim == False:
    architecture += ["sq"]

  architecture += ["chnl"]*n_channel_splits
  architecture += ["chk"]*n_checkerboard_splits_after

  if one_dim:
    actnorm_axes = (-1,)
  else:
    actnorm_axes = (-3, -2, -1)

  return build_architecture(architecture,
                            coupling_algorithm,
                            actnorm=True,
                            actnorm_axes=actnorm_axes,
                            glow=True,
                            one_dim=one_dim)
