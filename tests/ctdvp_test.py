import unittest

import jax
import jax.numpy as jnp

from jVMC.vqs import NQS
from jVMC.compression import EVDDecompressor
from jVMC.nets.rbm import RBM, CpxRBM

class TestDecompressor(unittest.TestCase):

    def test_evd_decompressor(self):

        def f(arr):
            return jnp.sum(arr)

        eigvecs = jnp.array([[1., 0.], [0., 1.]])
        eigvals = jnp.array([0.4, 0.6])
        d = EVDDecompressor(eigvecs, eigvals, 0.5)

        x_c = jnp.array([22222., 11111.])
        fd = lambda x_c, d: f(d(x_c))
        gd_x = jax.jit(jax.grad(fd, argnums=0))(x_c, d)

        assert jnp.all(gd_x == jnp.array([0., 1.]))


    def test_c_gradients(self):
        s = jnp.array([1., 0., 1.])[None, None, :]
        net = RBM(numHidden=3, bias=False)
        vqs = NQS(net, batchSize=1)

        d = EVDDecompressor(jnp.eye(9), jnp.array([0.4, 1., 0.2, 1., 1., 1., 1., 1., 1.]), 0.5)
        vqs.assign_decompressor(d)

        cg = vqs.c_gradients(s)
        assert cg.ravel()[0] == 0. and cg.ravel()[2] == 0.

    def test_c_gradients_cpx(self):
        s = jnp.array([1., 0.])[None, None, :]
        net = CpxRBM(numHidden=2, bias=False)
        vqs = NQS(net, batchSize=1)


        d = EVDDecompressor(jnp.eye(8), jnp.array([0.4, 1., 0.2, 1., 1., 1., 1., 1.]), 0.5)
        vqs.assign_decompressor(d)

        cg = vqs.c_gradients(s).ravel()
        assert cg[0] == 0. and cg[2] == 0.
        assert cg[1] != 0. and jnp.all(cg[3:] != 0.)
