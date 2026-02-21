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

@jax.tree_util.register_pytree_node_class
class EVDDecompressor(Decompressor):

    def __init__(self, eigvecs, eigvals, threshold):
        self.eigvecs = eigvecs
        self.eigvals = eigvals
        # eigvecs[:, i] = i-th eigenvector

        self.threshold = threshold

    def  __call__(self, compressed_params: jnp.ndarray):
        masked_c_params = jnp.where(self.eigvals > self.threshold, compressed_params, 0.)
        return (self.eigvecs * masked_c_params[None, :]).sum(axis=1)

    def assign_eigpairs(self, new_eigvecs, new_eigvals):
        assert new_eigvecs.shape == self.eigvecs.shape # Watch out: not jittable!
        assert new_eigvals.shape == self.eigvals.shape
        self.eigvecs = new_eigvecs
        self.eigvals = new_eigvals

    def zeros(self):
        return {"params": jnp.zeros_like(self.eigvals)}

    def tree_flatten(self):
        return (self.eigvecs, self.eigvals), self.threshold

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, aux_data)


