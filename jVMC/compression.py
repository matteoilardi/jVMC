import jax
import jax.numpy as jnp

from abc import ABC, abstractmethod

class Decompressor(ABC):
    @abstractmethod
    def __call__(self, compressed_params):
        ...
    @abstractmethod
    def zeros(self):
        ...

    def apply_transposed_jacobian_at_zero(self, v):
        mask = self.eigvals > self.threshold
        def decompress(compressed_params):
            return self.__call__(compressed_params * mask)

        vjp_fun = jax.vjp(decompress, self.zeros())[1]
        return vjp_fun(v)[0]

@jax.tree_util.register_pytree_node_class
class EVDDecompressor(Decompressor):

    def __init__(self, eigvecs, eigvals, threshold):
        self.eigvecs = eigvecs
        self.eigvals = eigvals
        # eigvecs[:, i] = i-th eigenvector

        self.threshold = threshold

    def  __call__(self, compressed_params: jnp.ndarray):
        mask = self.eigvals > self.threshold
        return (self.eigvecs * mask[None, :] * compressed_params[None, :]).sum(axis=1)

    def apply_transposed_jacobian_at_zero(self, v):
        mask = self.eigvals > self.threshold
        jacobian = self.eigvecs * mask[None, :]
        return v @ jacobian

    def zeros(self):
        return jnp.zeros_like(self.eigvals, dtype=jnp.complex128)

    def tree_flatten(self):
        return (self.eigvecs, self.eigvals), self.threshold

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, aux_data)


