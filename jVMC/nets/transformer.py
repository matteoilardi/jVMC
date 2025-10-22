import jax
jax.config.update("jax_enable_x64", True)
import flax
import flax.linen as nn
import jax.numpy as jnp
import jax.random as random
import numpy as np
from einops import rearrange

import jVMC
import jVMC.global_defs as global_defs
from jVMC.nets.activation_functions import activationFunctions, log_cosh
from jVMC.nets.initializers import init_fn_args
# NOTE This implementation uses built-in initializers from flax since all parameters are real-valued.
# NOTE Consider defining custom initializers for complex-valued parameters instead.

class Embedder(nn.Module):
    """
    Spin Patch Embedder.

    Steps:
        1) Map spin values from {0, 1} to {-1, 1}.
        2) Reshape the 1D spin configuration into non-overlapping patches
           of length `patch_len`.
        3) Apply a shared linear transformation (Dense layer) to each patch,
           projecting it into the embedding space of dimension `embed_dim`.

    Notes:
        - The same linear transformation is applied to every patch, meaning
          spatial information is encoded solely through patch ordering.
        - This module is analogous to the patch embedding stage in Vision
          Transformers (ViTs), but tailored to binary spin configurations.

    Input shape:
        (seq_len,)

    Output shape:
        (eff_len, embed_dim)
        where eff_len = seq_len / patch_len
    """
    embed_dim: int
    patch_len: int

    @nn.compact
    def __call__(self, spins):
        x = 2. * spins.ravel() - 1.
        x = rearrange(x, '(eff_len patch_len) -> eff_len patch_len', patch_len=self.patch_len)
        x = nn.Dense(self.embed_dim, **init_fn_args(
            kernel_init=nn.initializers.xavier_uniform(),
            dtype=global_defs.tReal
        ))(x)
        return x


class FMHA(nn.Module):
    """
    Factored Multi-Head Attention (FMHA) layer.

    Steps:
        1) Save the input for the residual connection at the end.
        2) Apply Layer Normalization in the embedding space.
        3) Compute value vectors v via a dense (linear) layer of size embed_dim.
        4) Split v into n_heads heads: each of dimension head_dim = embed_dim / n_heads.
        5) Contract v with a learned, input-independent attention tensor J of shape
           (n_heads, eff_len, eff_len), producing per-head outputs.
        6) Reshape and concatenate the heads back into the embedding dimension.
        7) Apply a final dense layer in embedding space and add the residual connection.

    Notes:
        - jnp.matmul is used for efficiency (w. r. t. jnp.einsum); it treats leading axes 
          as batch axes, so the first axis (n_heads) is preserved across the multiplication. 
          This is why v is reshaped before applying J.
        - The parameter J represents learned attention weights that do not depend
          on the input, unlike standard self-attention where weights are computed
          dynamically via QK^T.

    Input shape:
        (eff_len, embed_dim)

    Output shape:
        (eff_len, embed_dim)
    """
    n_heads: int
    # NOTE No need to specify embed_dim, since the flax Module infers it from the input it receives

    @nn.compact
    def __call__(self, x):
        embed_dim = x.shape[-1]
        if embed_dim % self.n_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by n_heads ({self.n_heads})")
        
        skip = x

        x = nn.LayerNorm(**init_fn_args(dtype=global_defs.tReal))(x)
        v = nn.Dense(embed_dim, name='V', **init_fn_args(
            kernel_init=nn.initializers.xavier_uniform(),
            dtype=global_defs.tReal
        ))(x)
        
        v = rearrange(v, 'eff_len (n_heads head_dim) -> n_heads eff_len head_dim', n_heads=self.n_heads)
        J = self.param('J', nn.initializers.xavier_uniform(), (v.shape[0], v.shape[1], v.shape[1]), global_defs.tReal)
        x = jnp.matmul(J, v)
        
        x = rearrange(x, 'n_heads eff_len head_dim -> eff_len (n_heads head_dim)')
        x = nn.Dense(embed_dim, name='W', **init_fn_args(
            kernel_init=nn.initializers.xavier_uniform(),
            dtype=global_defs.tReal
        ))(x)

        return x + skip
    
    

class FeedForward(nn.Module):
    """
    Feed-Forward Block.

    Steps:
        1) Store input for residual connection.
        2) Apply Layer Normalization in embedding space.
        3) Pass through one or more hidden Dense layers of size 2 * embed_dim,
           each followed by a ReLU activation.
        4) Apply a final Dense layer reducing dimensionality back to embed_dim.
        5) Add the residual connection to the output.

    Notes:
        - This block corresponds to the position-wise feed-forward network
          used in standard Transformer architectures.

    Input shape:
        (eff_len, embed_dim)

    Output shape:
        (eff_len, embed_dim)
    """
    n_layers: int

    @nn.compact
    def __call__(self, x):
        skip = x

        x = nn.LayerNorm(**init_fn_args(dtype=global_defs.tReal))(x)
        embed_dim = x.shape[-1]

        for i in range(self.n_layers):
            x = nn.Dense(2*embed_dim, name=f'ff_hidden_{i}', **init_fn_args(
                kernel_init=nn.initializers.xavier_uniform(),
                dtype=global_defs.tReal
            ))(x)
            x = activationFunctions['relu'](x)
            
        x = nn.Dense(embed_dim, name='ff_out', **init_fn_args(
            kernel_init=nn.initializers.xavier_uniform(),
            dtype=global_defs.tReal
        ))(x)

        return x + skip



class OutputHead(nn.Module):
    """
    1) Sum over the patch (sequence) axis
    2) Layer normalization in the embedding space
    3) Two separate dense layers of size embed_dim produce amplitude (amp) and phase
    4) Apply layer normalization independently to amp and phase
    5) Combine into a complex output: out = amp + j * phase
    6) Apply elementwise nonlinearity: out = log(cosh(out))
    7) Sum along the embedding dimension to produce a scalar output

    input shape: (eff_len, embed_dim)
    output shape: (1,)
    """

    @nn.compact
    def __call__(self, x):
        embed_dim = x.shape[1]
        x = jnp.sum(x, axis=0)
        x = nn.LayerNorm(**init_fn_args(dtype=global_defs.tReal))(x)

        amp = nn.Dense(embed_dim, name='out_amp', **init_fn_args(
            kernel_init=nn.initializers.xavier_uniform(),
            dtype=global_defs.tReal
        ))(x)
        amp = nn.LayerNorm(name='out_ln_amp', **init_fn_args(dtype=global_defs.tReal))(amp)

        phase = nn.Dense(embed_dim, name='out_phase', **init_fn_args(
            kernel_init=nn.initializers.xavier_uniform(),
            dtype=global_defs.tReal
        ))(x)
        phase = nn.LayerNorm(name='out_ln_phase', **init_fn_args(dtype=global_defs.tReal))(phase)

        out = jnp.array(amp + 1j*phase, dtype=global_defs.tCpx)
        out = log_cosh(out)
        return jnp.sum(out)


class CpxVisionTransformer(nn.Module):
    """
    Complex-valued Vision Transformer using Factored Multi-Head Attention (FMHA).

    1) Embed spin configurations into patch-wise embeddings.
    2) Apply a stack of Transformer-like layers, each consisting of:
         - Factored Multi-Head Attention (FMHA).
         - Feed-Forward (FF) block with residual connection.
    3) Aggregate patch embeddings through the OutputHead to produce a complex scalar.

    Characteristics:
        - Non-holomorphic.
        - Not autoregressive: all spins are processed in parallel.
        - No explicit translational invariance (though FMHA can be made invariant).
        - Coupling constants are not input features (this is required for jVMC compatibility).

    Based on:
        Rende, *nqs-models/ising_fnqs* (HuggingFace)

    Input shape:
        (seq_len,)

    Output shape:
        (1,)
    """

    patch_len: int = 4
    embed_dim: int = 72
    n_layers: int = 6
    n_heads: int = 12
    n_layers_ff: int = 1

    @nn.compact
    def __call__(self, spins):
        x = Embedder(embed_dim=self.embed_dim, patch_len=self.patch_len)(spins)

        for _ in range(self.n_layers):
            x = FMHA(n_heads=self.n_heads)(x)
            x = FeedForward(n_layers=self.n_layers_ff)(x)

        x = OutputHead()(x)

        return x
