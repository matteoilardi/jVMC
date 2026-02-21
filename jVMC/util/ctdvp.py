import jax
import jax.numpy as jnp
import numpy as np

import jVMC
import jVMC.mpi_wrapper as mpi
from jVMC.stats import SampledObs
from jVMC.util import TDVPBase
from jVMC.compression import EVDDecompressor

import warnings

class cTDVP:
    # TODO complete docstring
    # TODO can pinvCutoff be removed entirely?
    """ This class provides functionality to solve the time-dependent variational principle of tVMC
    in a space of compressed parameters.
    While it could be used in principle for Stochastic Reconfiguration (imaginary time dynamics),
    this is ill advised.

    With the force vector

        :math:`F_k=\\langle \\mathcal O_{\\theta_k}^* E_{loc}^{\\theta}\\rangle_c`

    and the quantum Fisher matrix

        :math:`S_{k,k'} = \\langle (\\mathcal O_{\\theta_k})^* \\mathcal O_{\\theta_{k'}}\\rangle_c`

    and for real parameters :math:`\\theta\\in\\mathbb R`, the tVMC/SR equation reads

        :math:`\\text{Re}\\big[S_{k,k'}\\big]\\dot\\theta_{k'} = -\\text{Re}\\big[xF_k\\big]`

    Here, either :math:`x=1` for ground state search or :math:`x=i` (the imaginary unit) for real time dynamics.

    Initializer arguments:
        * ``sampler``: A sampler object.
        * ``rhsPrefactor``: Prefactor :math:`x` of the right hand side, see above.
        * ``pinvCutoff``: Eigenvalue cutoff used in pseudoinversion.
        * ``decompressorCutoff``: Directions in parameter space with eigenvalue below this theshold are suppressed.
        * ``diagonalizeOnDevice``: Choose whether to diagonalize :math:`S` on GPU or CPU.
        * ``mpiRoot``: MPI rank that takes care of calculating parameter updates.
    """
    def __init__(
        self, sampler, rhsPrefactor=1.j, pinvCutoff=1e-8, decompressorCutoff = 1e-6, diagonalizeOnDevice=False, mpiRoot = 0,
    ):
        self.sampler = sampler

        if rhsPrefactor != 1.j:
            warnings.warn(f"Got rhsPrefactor = {rhsPrefactor}, but atVMC is meant for real time dynamics only", category=UserWarning)
        self.rhsPrefactor = rhsPrefactor

        self.makeReal = jnp.real
        self.pinvCutoff = pinvCutoff
        self.decompressorCutoff = decompressorCutoff

        self.diagonalizeOnDevice = diagonalizeOnDevice
        self.mpiRoot = mpiRoot

    def get_energy_variance(self):
        return self.ElocVar0

    def get_energy_mean(self):
        return self.ElocMean0.real

    def calc_norm_lite(self, S, update):
        """Compute the local-in-time error (lite), normalized by the variance of the local energy."""
        return 1. - (1./self.ElocVar0) * (update @ S @ update)

    def __call__(self, netParameters, t, *, psi, hamiltonian, **rhsArgs):
        """ For given network parameters this function solves the variational equation (be it TDVP, minSR or aTDVP).

        This function returns the parameters' update per unit time (i. e. their time derivative along the simulation).
        Thereby an instance of the ``TDVP`` class is a suited callable for the right hand side of an ODE to be used
        in combination with the integration schemes implemented in ``jVMC.stepper``. Alternatively, the interface
        matches the scipy ODE solvers as well.

        Arguments:
            * ``netParameters``: Parameters of the NQS.
            * ``t``: Current time.
            * ``psi``: NQS ansatz. Instance of ``jVMC.vqs.NQS``.
            * ``hamiltonian``: Hamiltonian operator, i.e., an instance of a derived class of ``jVMC.operator.Operator``. \
                                *Notice:* Current time ``t`` is by default passed as argument when computing matrix elements.

        Further optional keyword arguments:
            * ``numSamples``: Number of samples to be used by MC sampler.
            * ``outp``: An instance of ``jVMC.OutputManager``. If ``outp`` is given, timings of the individual steps \
                are recorded using the ``OutputManger``.
            * ``intStep``: Integration step number of multi step method like Runge-Kutta. This information is used to store \
                quantities like energy or residuals at the initial integration step.

        Returns:
            The solution of the variational equation, :math:`\\dot\\theta`.
        """

        tmpParameters = psi.get_parameters()
        psi.set_parameters(netParameters)

        intStep = rhsArgs.get("intStep", None)
        self.outp = rhsArgs.get("outp", None)

        def start_timing(name):
            if self.outp is not None:
                self.outp.start_timing(name)

        def stop_timing(name, waitFor=None):
            if waitFor is not None:
                waitFor.block_until_ready()
            if self.outp is not None:
                self.outp.stop_timing(name)

        # Get sample
        start_timing("sampling")
        sampleConfigs, sampleLogPsi, p = self.sampler.sample(numSamples=rhsArgs.get("numSamples", None))
        stop_timing("sampling", waitFor=sampleConfigs)

        # Evaluate local energy
        start_timing("compute Eloc")
        Eloc = hamiltonian.get_O_loc(sampleConfigs, psi, sampleLogPsi, t)
        stop_timing("compute Eloc", waitFor=Eloc)
        Eloc = SampledObs(Eloc, p)

        # Evaluate gradients
        if intStep is None or intStep == 0:
            start_timing("compute gradients")
            sampleGradients = psi.gradients(sampleConfigs)
            stop_timing("compute gradients", waitFor=sampleGradients)
            sampleGradients = SampledObs(sampleGradients, p)

            start_timing("solve TDVP eqn.")
            self.initialize_decompressor(sampleGradients)
            stop_timing("solve TDVP eqn.") # TODO wait for decompressor (?)

        # Assign decompressor to psi
        psi.assign_decompressor(self.decompressor)

        # Evaluate gradients w. r. t. compressed parameters
        start_timing("compute gradients")
        sampleCGradients = psi.c_gradients(sampleConfigs)
        stop_timing("compute gradients", waitFor=sampleCGradients)
        sampleCGradients = SampledObs(sampleCGradients, p)

        # Solve TDVP
        start_timing("solve TDVP eqn.")
        update = self.solve(Eloc, sampleCGradients, intStep)
        stop_timing("solve TDVP eqn.")

        if self.outp is not None:
            self.outp.add_timing("MPI communication", mpi.get_communication_time())

        psi.set_parameters(tmpParameters)

        return update


    def initialize_decompressor(self, gradients: SampledObs):
        S = gradients.covar()
        S = self.makeReal(S)
        S = 0.5 * (S + S.T)

        ev, V = self.diagonalize(S)
        self.decompressor = EVDDecompressor(V, ev, self.decompressorCutoff)
        self.spectrum = ev
        # TODO decide here how many directions to suppress

    def solve(self, Eloc: SampledObs, c_gradients: SampledObs, intStep: int | None):
        S, F = self.get_tdvp_equation(Eloc, c_gradients)

        if mpi.rank == self.mpiRoot:

            idx = jnp.nonzero(F)[0]
            subS, subF = S[idx][:, idx], F[idx]

            subEv, subV = self.diagonalize(subS)
            c_subUpdate = self.calc_update(subEv, subV, subF)

            c_update = jnp.zeros(F.shape, dtype=F.dtype).at[idx].set(c_subUpdate)
            update = self.decompressor(c_update)

            if intStep is None or intStep == 0:
                self.ElocMean0 = self.ElocMean
                self.ElocVar0 = self.ElocVar

                self.metadata = {
                    "spectrum": self.spectrum,
                    "compressed_update": c_update,
                    "lite": self.calc_norm_lite(subS, c_subUpdate),
                }

        update = mpi.bcast_unknown_size(np.array(update))
        return jax.device_put(update, jVMC.global_defs.myDevice)


    def get_tdvp_equation(self, Eloc: SampledObs, gradients: SampledObs):
        self.ElocMean = complex(Eloc.mean()[0])
        self.ElocVar = float(Eloc.var()[0])

        F = gradients.covar(Eloc).ravel()
        S = gradients.covar()
        # NOTE all MPI ranks must call get_tdvp_equation inside solve beacause of global sums implicit in covariance calculations
        # NOTE however, only the root actually uses S and F directly

        F = self.makeReal((-self.rhsPrefactor) * F)
        S = self.makeReal(S)

        # Ensure S is symmetric
        S = 0.5 * (S + S.T)
        return S, F

    def calc_update(self, ev: jnp.ndarray, V: jnp.ndarray, F: jnp.ndarray) -> jnp.ndarray:
        invEv = np.where(ev > self.pinvCutoff, 1./ev, 0.)

        VT_F = V.T @ F
        update = V @ np.diag(invEv) @ VT_F

        return update

    def diagonalize(self, S: jnp.ndarray):
        if self.diagonalizeOnDevice:
            ev, V = jnp.linalg.eigh(S)
        else:
            S = np.array(S)
            ev, V = np.linalg.eigh(S)
            ev, V = jnp.array(ev), jnp.array(V)

        return ev, V

# ** end class cTDVP

