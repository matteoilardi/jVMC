import jax
jax.config.update("jax_enable_x64", True)
import flax
#from flax import nn
import flax.linen as nn
import jax.numpy as jnp

import jVMC.global_defs as global_defs
import jVMC.nets.activation_functions as act_funs
from jVMC.nets.initializers import init_fn_args

from functools import partial

import jVMC.nets.initializers


class CpxRBM(nn.Module):
    """Restricted Boltzmann machine with complex parameters.

    Initialization arguments:
        * ``s``: Computational basis configuration.
        * ``numHidden``: Number of hidden units.
        * ``bias``: ``Boolean`` indicating whether to use bias.

    """
    numHidden: int = 2
    bias: bool = False

    @nn.compact
    def __call__(self, s):

        layer = nn.Dense(self.numHidden, use_bias=self.bias,
                         **init_fn_args(kernel_init=jVMC.nets.initializers.cplx_init,
                                        bias_init=jax.nn.initializers.zeros,
                                        dtype=global_defs.tCpx)
                         )

        return jnp.sum(act_funs.log_cosh(layer(2 * s.ravel() - 1)))

# ** end class CpxRBM


class CpxRBM_TI(nn.Module):
    """Translationally Invariant Restricted Boltzmann machine with complex parameters.

    Initialization arguments:
        * ``s``: Computational basis configuration.
        * ``density``: Hidden/visible unit ratio (i. e. number of convolutional filters).
        * ``bias``: ``Boolean`` indicating whether to use bias (for hidden units only).

    """
    density: int = 3
    bias: bool = True

    @nn.compact
    def __call__(self, s):
        x = 2. * s - 1.
        spin_shape = x.shape

        # Add batch (leading) and channel (trailing) dimensions
        x = x[None, ..., None]
        x = nn.Conv(
            features=self.density,
            kernel_size=spin_shape,
            padding='CIRCULAR',
            use_bias=self.bias,
            **init_fn_args(
                kernel_init=jVMC.nets.initializers.cplx_init,
                bias_init=jax.nn.initializers.zeros,
                dtype=global_defs.tCpx
            )
        )(x)

        # x = x[0] # Remove batch dimension
        return jnp.sum(act_funs.log_cosh(x))

# ** end class CpxRBM_TI


class CpxRBM_TI1d_Var(nn.Module):
    """Translationally Invariant Restricted Boltzmann machine with complex parameters.

    Initialization arguments:
        * ``s``: Computational basis configuration.
        * ``density``: Hidden/visible unit ratio (i. e. number of convolutional filters).
        * ``bias``: ``Boolean`` indicating whether to use bias (for hidden units only).

    """
    density: int = 3
    bias: bool = True

    @nn.compact
    def __call__(self, s):
        x = 2. * s.ravel() - 1.
        N = x.shape[0]

        W = self.param("W", jVMC.nets.initializers.cplx_init, (self.density, N), global_defs.tCpx)
        circulant_matrix = (jnp.arange(N)[None, :] - jnp.arange(N)[:, None]) % N
        W_circulant = W[:, circulant_matrix]
        x = W_circulant @ x

        if self.bias:
            b = self.param("b", jax.nn.initializers.zeros, (self.density,), global_defs.tCpx)
            x += b[:, None]

        return jnp.sum(act_funs.log_cosh(x))

# ** end class CpxRBM_TI1d


class CpxRBM_Nospinflip(nn.Module):
    """Restricted Boltzmann machine with complex parameters.

    Initialization arguments:
        * ``s``: Computational basis configuration.
        * ``numHidden``: Number of hidden units.
        * ``bias``: ``Boolean`` indicating whether to use bias.

    """
    numHidden: int = 2
    bias: bool = False

    @nn.compact
    def __call__(self, s):

        layer = nn.Dense(self.numHidden, use_bias=self.bias,
                         **init_fn_args(kernel_init=jVMC.nets.initializers.cplx_init,
                                        bias_init=jax.nn.initializers.zeros,
                                        dtype=global_defs.tCpx)
                         )

        return jnp.sum(act_funs.log_cosh(layer(s.ravel())))


class RBM(nn.Module):
    """Restricted Boltzmann machine with real parameters.

    Initialization arguments:
        * ``s``: Computational basis configuration.
        * ``numHidden``: Number of hidden units.
        * ``bias``: ``Boolean`` indicating whether to use bias.

    """
    numHidden: int = 2
    bias: bool = False

    @nn.compact
    def __call__(self, s):

        layer = nn.Dense(self.numHidden, use_bias=self.bias,
                         **init_fn_args(kernel_init=jax.nn.initializers.lecun_normal(dtype=global_defs.tReal),
                                        bias_init=jax.nn.initializers.zeros,
                                        dtype=global_defs.tReal)
                        )

        return jnp.sum(jnp.log(jnp.cosh(layer(2 * s - 1))))

# ** end class RBM
