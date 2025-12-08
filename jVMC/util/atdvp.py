import jax
import jax.numpy as jnp
import numpy as np

import jVMC
import jVMC.mpi_wrapper as mpi
from jVMC.stats import SampledObs

from abc import ABC, abstractmethod
from functools import partial
import warnings
from collections import namedtuple


def expand_masked(a: np.ndarray, mask: np.ndarray):
    assert a.shape[0] == mask.sum(), "Shape mismatch"
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros_like(mask, dtype=a.dtype)
    out[mask] = a
    return out

Metadata = namedtuple("Metadata", ["lite", "mask", "importanceOnParams", "importanceOffParams"])
"""Used to bundle step metadata in atVMC"""


class TDVPBase(ABC):
    
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
        start_timing("compute gradients")
        sampleGradients = psi.gradients(sampleConfigs)
        stop_timing("compute gradients", waitFor=sampleGradients)
        sampleGradients = SampledObs(sampleGradients, p)
        
        # Solve TDVP
        start_timing("solve TDVP eqn.")
        update = self.solve(Eloc, sampleGradients, rhsArgs.get("intStep", None))
        stop_timing("solve TDVP eqn.")

        if self.outp is not None:
            self.outp.add_timing("MPI communication", mpi.get_communication_time())
        
        psi.set_parameters(tmpParameters)

        return update

    @abstractmethod
    def solve(self, Eloc: SampledObs, gradients: SampledObs, intStep: int | None) -> jnp.ndarray:
        ...


class aTDVP(TDVPBase):
    """ This class provides functionality to solve a time-dependent variational principle (TDVP).

    With the force vector

        :math:`F_k=\\langle \\mathcal O_{\\theta_k}^* E_{loc}^{\\theta}\\rangle_c`

    and the quantum Fisher matrix

        :math:`S_{k,k'} = \\langle (\\mathcal O_{\\theta_k})^* \\mathcal O_{\\theta_{k'}}\\rangle_c`

    and for real parameters :math:`\\theta\\in\\mathbb R`, the TDVP equation reads

        :math:`q\\big[S_{k,k'}\\big]\\dot\\theta_{k'} = -q\\big[xF_k\\big]`

    Here, either :math:`q=\\text{Re}` or :math:`q=\\text{Im}` and :math:`x=1` for ground state
    search or :math:`x=i` (the imaginary unit) for real time dynamics.

    Initializer arguments:
        * ``sampler``: A sampler object.
        * ``rhsPrefactor``: Prefactor :math:`x` of the right hand side, see above.
        * ``makeReal``: Specifies the function :math:`q`, either `'real'` for :math:`q=\\text{Re}` or `'imag'` for :math:`q=\\text{Im}`.
        * ``diagonalShift``: Regularization parameter :math:`\\rho` for ground state search, see above.
        * ``pinvCutoff``: Lower bound for the regularization parameter :math:`\\epsilon_{SVD}`, see above.
        * ``diagonalizeOnDevice``: Choose whether to diagonalize :math:`S` on GPU or CPU.
        * ``liteCutoff``: Upper limit for lite; the algorithm attempts to disable as many parameters as possible without exceeding this limit.”.
        * ``paramImportanceCutoff``: Active parameters with importance (deltaLiteOnParameters) below this limit are switched off.
        * ``minSwitchOff``: Minimum number of params selected for switch that triggers binary search for the optimal number.
        * ``mpiRoot``: MPI rank that takes care of calculating paramter update.
    """
    # TODO Complete docstring

    def __init__(
        self, sampler, rhsPrefactor=1.j, makeReal='real', diagonalShift=1e-4, pinvCutoff=1e-8, diagonalizeOnDevice=False,
        liteCutoff = 0.01, paramImportanceCutoff = None, minSwitchOff = 5, mpiRoot = 0,
    ):
        self.sampler = sampler
        if rhsPrefactor != 1.j:
            warnings.warn(f"Got rhsPrefactor = {rhsPrefactor}, but aTVMC is meant for real time dynamics only", category=UserWarning)
        self.rhsPrefactor = rhsPrefactor
        
        if makeReal == 'real':
            self.makeReal = np.real
        elif makeReal == 'imag':
            self.makeReal = np.imag
        else:
            raise ValueError("Argument makeReal should be either `real` or `imag`")
        
        self.pinvCutoff = pinvCutoff
        self.diagonalShift = diagonalShift

        self.mpiRoot = mpiRoot
        if mpi.rank == mpiRoot:
            self.diagonalizeOnDevice = diagonalizeOnDevice
    
            # atVMC-specific hyperparameters
            self.liteCutoff = liteCutoff
            self.paramImportanceCutoff = paramImportanceCutoff
            self.minSwitchOff = minSwitchOff
    
            # All parameters are active at the beginning
            self.currentMask = None
            self.nextMask = None
    
    def get_energy_variance(self):
        return self.ElocVar0

    def get_energy_mean(self):
        return jnp.real(self.ElocMean0)

    # TODO jit compile with NUMBA
    def solve(self, Eloc: SampledObs, gradients: SampledObs, intStep: int | None):
        S, F = self.get_tdvp_equation(Eloc, gradients)

        if mpi.rank == self.mpiRoot:
            # Read mask for this iteration (or initialize if first iteration)
            if intStep is None or intStep == 0:
                self.currentMask = self.nextMask if self.nextMask is not None else np.full(F.shape, True, dtype=bool)
            mask = self.currentMask
            
            subS, subF = S[np.ix_(mask, mask)], F[mask]
            invSubS, subUpdate = self.calc_update(subS, subF)
    
            if intStep is None or intStep == 0:
                self.ElocMean0 = self.ElocMean
                self.ElocVar0 = self.ElocVar
                
                # Calculate metadata
                lite = self.calc_lite(subS, subUpdate)
                importanceOnParams = aTDVP.calc_importance_on_params(subS, subUpdate)
                importanceOffParams = aTDVP.calc_importance_off_params(invSubS, subUpdate, S, F, mask)
                metadata = Metadata(lite, mask, importanceOnParams, importanceOffParams)
        
                # Calculate mask for the next iteration
                self.nextMask = self.switch_off_params(metadata, subS, subF) if lite < self.liteCutoff else self.switch_on_params(metadata)
                self.metadata = metadata._asdict()
    
            update = expand_masked(subUpdate, mask)

        update = mpi.bcast_unknown_size(update)
        return jax.device_put(update, jVMC.global_defs.myDevice)

    def get_tdvp_equation(self, Eloc: SampledObs, gradients: SampledObs):
        self.ElocMean = Eloc.mean()[0]
        self.ElocVar = Eloc.var()[0]

        F = gradients.covar_to_host(Eloc).ravel()
        S = gradients.covar_to_host()
        # NOTE all MPI ranks must call get_tdvp_equation inside solve beacause of global sums implicit in covariance calculations
        # NOTE however, only the root actually uses S and F directly
        
        F = self.makeReal((-self.rhsPrefactor) * F)
        S = self.makeReal(S)

        S = S + np.diag(self.diagonalShift * np.diag(S)) # NOTE multiplicative on each diagonal element
        
        return S, F

    def switch_off_params(self, metadata, subS: np.ndarray, subF: np.ndarray) -> np.ndarray[bool]:
        lite = metadata.lite
        mask = metadata.mask
        importanceOnParams = metadata.importanceOnParams
        nActive = importanceOnParams.shape[0]

        if nActive == 1:
            return mask

        paramIdxSorted = np.argsort(importanceOnParams)
        paramImportanceSorted = importanceOnParams[paramIdxSorted]
        cumImportance = np.cumsum(paramImportanceSorted)
        nSwitchOffTry = np.searchsorted(cumImportance, self.liteCutoff - lite)

        if nSwitchOffTry < self.minSwitchOff:
            newSubMask = np.ones(nActive, dtype=bool)
            newSubMask[paramIdxSorted[:1]] = False
            return expand_masked(newSubMask, mask)

        def lite_n_switchoff(nSwitchOff: int):
            idxSwitchOff = paramIdxSorted[:nSwitchOff]
            newSubMask = np.ones(nActive, dtype=bool)
            newSubMask[idxSwitchOff] = False
            
            subSubS, subSubF = subS[np.ix_(newSubMask, newSubMask)], subF[newSubMask]
            invSubSubS, subSubUpdate = self.calc_update(subSubS, subSubF)

            return self.calc_lite(subSubS, subSubUpdate)

        def search_n_switchoff(left, right):
            # Assumes that liteLeft is below cutoff, while liteRight is above cutoff
            while right > left + 1:
                mid = left + (right - left) // 2
                liteMid = lite_n_switchoff(mid)
                if liteMid < self.liteCutoff:
                    left = mid
                else:
                    right = mid
            return left

        if lite_n_switchoff(nSwitchOffTry) > self.liteCutoff:
            # The treshold for nSwitchOff must be in the interval [0, nSwitchOffTry]
            nSwitchOff = search_n_switchoff(0, nSwitchOffTry)    
        else: # nSwitchOffTry is small enough to keep the lite below the cutoff
            if nSwitchOffTry >= nActive - 1:
                nSwitchOff = nActive - 1 # Keep at least one active parameter
            else:
                if lite_n_switchoff(nActive - 1) < self.liteCutoff:
                    # Below the lite cutoff even with one active parameter
                    nSwitchOff = nActive - 1
                else:
                    # The treshold for nSwitchOff must be in the interval [nSwitchOffTry, nActive-1]
                    nSwitchOff = search_n_switchoff(nSwitchOffTry, nActive - 1)

        # Build and return mask for the chosen value of nSwitchOff
        idxSwitchOff = paramIdxSorted[:nSwitchOff]
        newSubMask = np.ones(nActive, dtype=bool)
        newSubMask[idxSwitchOff] = False
        return expand_masked(newSubMask, mask)


    def switch_on_params(self, metadata) -> np.ndarray[bool]:
        lite = metadata.lite
        mask = metadata.mask
        importanceOffParams = metadata.importanceOffParams

        if importanceOffParams.size == 0:
            return mask

        nNonActive = importanceOffParams.shape[0]

        paramIdxSortedAscending = np.argsort(importanceOffParams)
        paramImportanceSortedAscending = importanceOffParams[paramIdxSortedAscending]

        if self.paramImportanceCutoff is not None:
            # Determine which currently active parameters should be switched off because of low importance
            prevMaskImportant = expand_masked(metadata.importanceOnParams, mask) > self.paramImportanceCutoff
            # Determine how many inactive parameters should be excluded from activation because of low importance
            nIrrelevant = np.searchsorted(paramImportanceSortedAscending, self.paramImportanceCutoff)
        else:
            prevMaskImportant = mask
            nIrrelevant = 0
        
        paramIdxSorted = paramIdxSortedAscending[::-1]
        paramImportanceSorted = paramImportanceSortedAscending[::-1]
        
        cumImportance = np.cumsum(paramImportanceSorted)
        nSwitchOn = np.searchsorted(cumImportance, lite - self.liteCutoff) + 1
        nSwitchOn = min(nSwitchOn, nNonActive-nIrrelevant)
        idxSwitchOn = paramIdxSorted[:nSwitchOn]

        newSubMask = np.zeros(nNonActive, dtype=bool)
        newSubMask[idxSwitchOn] = True

        newMask = prevMaskImportant | expand_masked(newSubMask, ~mask)            
        return newMask

    def calc_update(self, S: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.diagonalizeOnDevice:
            S = jax.device_put(S, jVMC.global_defs.myDevice)
            w, V = jnp.linalg.eigh(S)
            w, V = np.array(w), np.array(V)
        else:
            w, V = np.linalg.eigh(S)

        invW = np.where(w > 1e-14, 1./w, 0.)
        invS = V @ np.diag(invW) @ V.conj().T
        update = invS @ F
        return invS, update

    def calc_lite(self, S, update):
        return self.ElocVar0 - update.conj() @ S @ update

    @staticmethod
    def calc_importance_on_params(subS: np.ndarray, subUpdate: np.ndarray) -> np.ndarray:
        return np.diag(subS) * subUpdate.conj()*subUpdate

    @staticmethod
    def calc_importance_off_params(invSubS: np.ndarray, subUpdate: np.ndarray, S: np.ndarray, F: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if np.all(mask):
            return np.array([])
        
        Vks = np.swapaxes(S[np.ix_(mask, ~mask)], 0, 1) # For each k, Vk is along axis 1, so that axis 0 can be treated as a batch axis
        Skk = np.diag(S[np.ix_(~mask, ~mask)])
        #numeratorVec = Vks.conj() @ subUpdate + 1.j*F[~mask]
        numeratorVec = Vks.conj() @ subUpdate - F[~mask]
        Vdag_invS_V = np.einsum("ij,ij->i", Vks.conj()@invSubS, Vks)
        result = 1. / (Skk - Vdag_invS_V) * numeratorVec.conj() * numeratorVec
        return np.real(result)
