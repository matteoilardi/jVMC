import jax
import jax.numpy as jnp
import numpy as np

import jVMC
import jVMC.mpi_wrapper as mpi
from jVMC.stats import SampledObs

from numba import njit

from abc import ABC, abstractmethod
import warnings
from collections import namedtuple
import functools as ft

# TODO use normalized lite
# TODO consider implementing a "soft" pinvCutoff
# TODO consider adding support for adaptiveHeun

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
# ** end class TDVPBase


# ============= Helper jittable functions =============

def _base_expand_masked(a: np.ndarray, mask: np.ndarray):
    assert a.shape[0] == mask.sum(), "Shape mismatch"
    mask = mask.astype(np.bool_)
    out = np.zeros_like(mask, dtype=a.dtype)
    out[mask] = a
    return out

def _base_calc_lite(S, update, ElocVar):
    return ElocVar - update.conj() @ S @ update

def _base_switch_on_params(metadata, paramImportanceCutoff, liteCutoff) -> np.ndarray[np.bool_]:
    lite = metadata.lite
    mask = metadata.mask
    importanceOffParams = metadata.importanceOffParams

    if importanceOffParams.size == 0:
        return mask

    nNonActive = importanceOffParams.shape[0]

    paramIdxSortedAscending = np.argsort(importanceOffParams)
    paramImportanceSortedAscending = importanceOffParams[paramIdxSortedAscending]

    if paramImportanceCutoff is not None:
        # Determine which currently active parameters should be switched off because of low importance
        prevMaskImportant = _expand_masked(metadata.importanceOnParams, mask) > paramImportanceCutoff
        # Determine how many inactive parameters should be excluded from activation because of low importance
        nIrrelevant = np.searchsorted(paramImportanceSortedAscending, paramImportanceCutoff)
    else:
        prevMaskImportant = mask
        nIrrelevant = 0

    paramIdxSorted = paramIdxSortedAscending[::-1]
    paramImportanceSorted = paramImportanceSortedAscending[::-1]

    cumImportance = np.cumsum(paramImportanceSorted)
    nSwitchOn = np.searchsorted(cumImportance, lite - liteCutoff) + 1
    nSwitchOn = min(nSwitchOn, nNonActive-nIrrelevant)
    idxSwitchOn = paramIdxSorted[:nSwitchOn]

    newSubMask = np.zeros(nNonActive, dtype=np.bool_)
    newSubMask[idxSwitchOn] = True

    newMask = prevMaskImportant | _expand_masked(newSubMask, ~mask)
    return newMask

# ** end of helper jittable functions definition

Metadata = namedtuple("Metadata", ["lite", "mask", "importanceOnParams", "importanceOffParams"])
"""Used to bundle step metadata in atVMC"""

class aTDVP(TDVPBase):
    """ This class provides functionality to solve the time-dependent variational principle of tVMC
    by means of the atVMC algorithm of arXiv:2506.08575. This algorithm involves an adaptive
    switch on/switch off of parameters based on their computed importance in the simulation (roughly
    how much the parameter helps reducing lite, i. e. local in time error).
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
        * ``makeReal``: Specifies the function :math:`q`, either `'real'` for :math:`q=\\text{Re}` or `'imag'` for :math:`q=\\text{Im}`.
        * ``diagonalShift``: Regularization parameter :math:`\\rho` for ground state search, see above.
        * ``pinvCutoff``: Lower bound for the regularization parameter :math:`\\epsilon_{SVD}`, see above.
        * ``diagonalizeOnDevice``: Choose whether to diagonalize :math:`S` on GPU or CPU.
        * ``liteCutoff``: Upper limit for lite; the algorithm attempts to disable as many parameters as possible without exceeding this limit.
        * ``paramImportanceCutoff``: Active parameters with importance (deltaLiteOnParameters) below this limit are switched off.
        * ``minSwitchOff``: Minimum number of params selected for switch that triggers binary search for the optimal number.
        * ``mpiRoot``: MPI rank that takes care of calculating parameter updates.
        * ``backend``: Specifies which implementation of numerical routines to use.
            Can be `'numpy'` (plain numpy functions) or `'numba'` (numba-jitted functions).
    """
    def __init__(
        self, sampler, rhsPrefactor=1.j, makeReal='real', diagonalShift=1e-4, pinvCutoff=1e-8, diagonalizeOnDevice=False,
        liteCutoff = 0.01, paramImportanceCutoff = None, minSwitchOff = 3, mpiRoot = 0, backend = 'numpy',
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

        # MPI root further initialization
        if mpi.rank == mpiRoot:
            if backend == 'numpy':
                self.backend = NumpyBackend(diagonalizeOnDevice=diagonalizeOnDevice)
            elif backend == 'numba':
                if diagonalizeOnDevice:
                    warnings.warn(
                        f"On-device diagonalization is not allowed with numba backend, falling back to host diagonalization",
                        category=UserWarning
                    )
                self.backend = NumbaBackend()
            else:
                raise ValueError("Available backends are `numpy` and `numba`")

            # atVMC-specific hyperparameters
            self.liteCutoff = liteCutoff
            self.paramImportanceCutoff = paramImportanceCutoff
            self.minSwitchOff = minSwitchOff

            # All parameters are active at the beginning
            self.currentMask = None
            self.nextMask = None

        # End of MPI root further initialization

    def get_energy_variance(self):
        return self.ElocVar0

    def get_energy_mean(self):
        return self.ElocMean0.real


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
                importanceOnParams = self.calc_importance_on_params(subS, subUpdate)
                importanceOffParams = self.calc_importance_off_params(invSubS, subUpdate, S, F, mask)
                metadata = Metadata(lite, mask, importanceOnParams, importanceOffParams)

                # Calculate mask for the next iteration
                self.nextMask = self.switch_off_params(metadata, subS, subF) if lite < self.liteCutoff else self.switch_on_params(metadata)

                # Save metadata
                self.metadata = {
                    "lite": lite,
                    "mask": mask,
                    "importanceOnParams": self.expand_masked(importanceOnParams, mask),
                    "importanceOffParams": self.expand_masked(importanceOffParams, ~mask),
                }

            update = self.expand_masked(subUpdate, mask)

        update = mpi.bcast_unknown_size(update)
        return jax.device_put(update, jVMC.global_defs.myDevice)

    def get_tdvp_equation(self, Eloc: SampledObs, gradients: SampledObs):
        self.ElocMean = complex(Eloc.mean()[0])
        self.ElocVar = float(Eloc.var()[0])

        F = gradients.covar_to_host(Eloc).ravel()
        S = gradients.covar_to_host()
        # NOTE all MPI ranks must call get_tdvp_equation inside solve beacause of global sums implicit in covariance calculations
        # NOTE however, only the root actually uses S and F directly

        F = self.makeReal((-self.rhsPrefactor) * F)
        S = self.makeReal(S)

        S = S + np.diag(self.diagonalShift * np.diag(S)) # NOTE multiplicative on each diagonal element

        return S, F

    def switch_off_params(self, metadata, subS: np.ndarray, subF:np.ndarray) -> np.ndarray[np.bool_]:
        return self.backend['switch_off_params'](metadata, subS, subF, self.ElocVar0, self.liteCutoff, self.minSwitchOff)

    def switch_on_params(self, metadata):
        return self.backend['switch_on_params'](metadata, self.paramImportanceCutoff, self.liteCutoff)

    def calc_update(self, S: np.ndarray, F: np.ndarray):
        return self.backend['calc_update'](S, F)

    def calc_lite(self, S, update):
        return self.backend['calc_lite'](S, update, self.ElocVar0)

    def calc_importance_on_params(self, subS: np.ndarray, subUpdate: np.ndarray) -> np.ndarray:
        return self.backend['calc_importance_on_params'](subS, subUpdate)

    def calc_importance_off_params(self, invSubS: np.ndarray, subUpdate: np.ndarray, S: np.ndarray, F: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self.backend['calc_importance_off_params'](invSubS, subUpdate, S, F, mask)

    def expand_masked(self, a: np.ndarray, mask: np.ndarray):
        return self.backend['expand_masked'](a, mask)

# ** end class aTDVP


def NumpyBackend(diagonalizeOnDevice: bool):
    global _expand_masked, _calc_lite, _switch_on_params

    _expand_masked = _base_expand_masked
    _calc_lite = _base_calc_lite
    _switch_on_params = _base_switch_on_params

    def _calc_update(S: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if diagonalizeOnDevice:
            S = jax.device_put(S, jVMC.global_defs.myDevice)
            w, V = jnp.linalg.eigh(S)
            w, V = np.array(w), np.array(V)
        else:
            w, V = np.linalg.eigh(S)

        invW = np.where(w > 1e-14, 1./w, 0.)
        invS = V @ np.diag(invW) @ V.conj().T
        update = invS @ F
        return invS, update

    def _calc_importance_on_params(subS: np.ndarray, subUpdate: np.ndarray) -> np.ndarray:
        return np.diag(subS) * subUpdate.conj()*subUpdate

    def _calc_importance_off_params(invSubS: np.ndarray, subUpdate: np.ndarray, S: np.ndarray, F: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if np.all(mask):
            return np.array([])

        Vks = np.swapaxes(S[np.ix_(mask, ~mask)], 0, 1) # For each k, Vk is along axis 1, so that axis 0 can be treated as a batch axis
        Skk = np.diag(S[np.ix_(~mask, ~mask)])
        #numeratorVec = Vks.conj() @ subUpdate + 1.j*F[~mask]
        numeratorVec = Vks.conj() @ subUpdate - F[~mask]
        Vdag_invS_V = np.einsum("ij,ij->i", Vks.conj()@invSubS, Vks)
        result = 1. / (Skk - Vdag_invS_V) * numeratorVec.conj() * numeratorVec
        return np.real(result)

    def _switch_off_params(metadata, subS: np.ndarray, subF: np.ndarray, ElocVar, liteCutoff, minSwitchOff) -> np.ndarray[np.bool_]:
        lite = metadata.lite
        mask = metadata.mask
        importanceOnParams = metadata.importanceOnParams
        nActive = importanceOnParams.shape[0]

        if nActive == 1:
            return mask

        paramIdxSorted = np.argsort(importanceOnParams)
        paramImportanceSorted = importanceOnParams[paramIdxSorted]
        cumImportance = np.cumsum(paramImportanceSorted)
        nSwitchOffTry = np.searchsorted(cumImportance, liteCutoff - lite)

        if nSwitchOffTry < minSwitchOff:
            newSubMask = np.ones(nActive, dtype=np.bool_)
            newSubMask[paramIdxSorted[:1]] = False
            return _expand_masked(newSubMask, mask)

        def lite_n_switchoff(nSwitchOff: int):
            idxSwitchOff = paramIdxSorted[:nSwitchOff]
            newSubMask = np.ones(nActive, dtype=np.bool_)
            newSubMask[idxSwitchOff] = False

            subSubS, subSubF = subS[np.ix_(newSubMask, newSubMask)], subF[newSubMask]
            invSubSubS, subSubUpdate = _calc_update(subSubS, subSubF)

            return _calc_lite(subSubS, subSubUpdate, ElocVar)

        def search_n_switchoff(left, right):
            # Assumes that liteLeft is below cutoff, while liteRight is above cutoff
            while right > left + 1:
                mid = left + (right - left) // 2
                liteMid = lite_n_switchoff(mid)
                if liteMid < liteCutoff:
                    left = mid
                else:
                    right = mid
            return left

        if lite_n_switchoff(nSwitchOffTry) > liteCutoff:
            # The threshold for nSwitchOff must be in the interval [0, nSwitchOffTry]
            nSwitchOff = search_n_switchoff(0, nSwitchOffTry)    
        else: # nSwitchOffTry is small enough to keep the lite below the cutoff
            if nSwitchOffTry >= nActive - 1:
                nSwitchOff = nActive - 1 # Keep at least one active parameter
            else:
                if lite_n_switchoff(nActive - 1) < liteCutoff:
                    # Below the lite cutoff even with one active parameter
                    nSwitchOff = nActive - 1
                else:
                    # The threshold for nSwitchOff must be in the interval [nSwitchOffTry, nActive-1]
                    nSwitchOff = search_n_switchoff(nSwitchOffTry, nActive - 1)

        # Build and return mask for the chosen value of nSwitchOff
        idxSwitchOff = paramIdxSorted[:nSwitchOff]
        newSubMask = np.ones(nActive, dtype=np.bool_)
        newSubMask[idxSwitchOff] = False
        return _expand_masked(newSubMask, mask)

    return {
        "expand_masked": _expand_masked,
        "calc_lite": _calc_lite,
        "calc_update": _calc_update,#ft.partial(_calc_update, diagonalizeOnDevice=diagonalizeOnDevice),
        "calc_importance_on_params": _calc_importance_on_params,
        "calc_importance_off_params": _calc_importance_off_params,
        "switch_off_params": _switch_off_params,
        "switch_on_params": _switch_on_params,
    }

# ** end function NumpyBackend


def NumbaBackend():
    global _expand_masked, _calc_lite, _switch_on_params

    _expand_masked = njit(_base_expand_masked)
    _calc_lite = njit(_base_calc_lite)
    _switch_on_params = njit(_base_switch_on_params)

    @njit
    def _calc_update(S: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w, V = np.linalg.eigh(S)

        invW = np.where(w > 1e-14, 1./w, 0.)
        invS = V @ np.diag(invW) @ V.conj().T
        update = invS @ F
        return invS, update

    @njit
    def _calc_importance_on_params(subS: np.ndarray, subUpdate: np.ndarray) -> np.ndarray:
        n = subUpdate.shape[0]
        res = np.empty(n, dtype=np.float64)

        for i in range(n):
            res[i] = subS[i, i] * subUpdate[i]**2

        return res

    @njit
    def _calc_importance_off_params(invSubS: np.ndarray, subUpdate: np.ndarray, S: np.ndarray, F: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if np.all(mask):
            return np.empty(0, dtype=np.float64)

        # Calculate Vks
        cutS = S[mask]
        cutS = cutS[:, ~mask]
        Vks = np.swapaxes(cutS, 0, 1) # For each k, Vk is along axis 1, so that axis 0 can be treated as a batch axis

        # Calculate Skk
        nNonActive = np.sum(~mask)
        Skk = np.empty(nNonActive, dtype=np.float64)

        j = 0
        for i in range(mask.shape[0]):
            if not mask[i]:
                Skk[j] = S[i, i]
                j += 1

        #numeratorVec = Vks.conj() @ subUpdate + 1.j*F[~mask]
        numeratorVec = Vks.conj() @ subUpdate - F[~mask]

        # Calculate Vdag_invS_V
        Vdag_invS = Vks.conj()@invSubS
        Vdag_invS_V = np.empty(nNonActive, dtype=np.float64)
        Vdag_invS = np.ascontiguousarray(Vdag_invS)
        Vks = np.ascontiguousarray(Vks)

        for i in range(nNonActive):
            Vdag_invS_V[i] = Vdag_invS[i] @ Vks[i]

        # Calculate result
        result = 1. / (Skk - Vdag_invS_V) * numeratorVec.conj() * numeratorVec
        return np.real(result)

    @njit
    def _lite_n_switch_off(nSwitchOff, paramIdxSorted: np.ndarray, subS: np.ndarray, subF: np.ndarray, ElocVar) -> float:
        idxSwitchOff = paramIdxSorted[:nSwitchOff]
        nActive = paramIdxSorted.shape[0]
        newSubMask = np.ones(nActive, dtype=np.bool_)
        newSubMask[idxSwitchOff] = False

        cutSubS = subS[newSubMask]
        subSubS = cutSubS[:, newSubMask]
        subSubF = subF[newSubMask]

        invSubSubS, subSubUpdate = _calc_update(subSubS, subSubF)
        return _calc_lite(subSubS, subSubUpdate, ElocVar)

    @njit
    def _search_n_switch_off(left, right, paramIdxSorted: np.ndarray, subS: np.ndarray, subF: np.ndarray, ElocVar, liteCutoff) -> int:
        # Assumes that liteLeft is below cutoff, while liteRight is above cutoff
        while right > left + 1:
            mid = left + (right - left) // 2
            liteMid = _lite_n_switch_off(mid, paramIdxSorted, subS, subF, ElocVar)
            if liteMid < liteCutoff:
                left = mid
            else:
                right = mid
        return left

    @njit
    def _switch_off_params(metadata, subS: np.ndarray, subF: np.ndarray, ElocVar, liteCutoff, minSwitchOff) -> np.ndarray[bool]:
        lite = metadata.lite
        mask = metadata.mask
        importanceOnParams = metadata.importanceOnParams
        nActive = importanceOnParams.shape[0]

        if nActive == 1:
            return mask

        paramIdxSorted = np.argsort(importanceOnParams)
        paramImportanceSorted = importanceOnParams[paramIdxSorted]
        cumImportance = np.cumsum(paramImportanceSorted)
        nSwitchOffTry = np.searchsorted(cumImportance, liteCutoff - lite)

        if nSwitchOffTry < minSwitchOff:
            newSubMask = np.ones(nActive, dtype=np.bool_)
            newSubMask[paramIdxSorted[:1]] = False
            return _expand_masked(newSubMask, mask)

        # Initialize nSwitchOff
        if _lite_n_switch_off(nSwitchOffTry, paramIdxSorted, subS, subF, ElocVar) > liteCutoff:
            # The threshold for nSwitchOff must be in the interval [0, nSwitchOffTry]
            nSwitchOff = _search_n_switch_off(0, nSwitchOffTry, paramIdxSorted, subS, subF, ElocVar, liteCutoff)
        else: # nSwitchOffTry is small enough to keep the lite below the cutoff
            if nSwitchOffTry >= nActive - 1:
                nSwitchOff = nActive - 1 # Keep at least one active parameter
            else:
                if _lite_n_switch_off(nActive - 1, paramIdxSorted, subS, subF, ElocVar) < liteCutoff:
                    # Below the lite cutoff even with one active parameter
                    nSwitchOff = nActive - 1
                else:
                    # The threshold for nSwitchOff must be in the interval [nSwitchOffTry, nActive-1]
                    nSwitchOff = _search_n_switch_off(nSwitchOffTry, nActive - 1, paramIdxSorted, subS, subF, ElocVar, liteCutoff)

        # Build and return mask for the chosen value of nSwitchOff
        idxSwitchOff = paramIdxSorted[:nSwitchOff]
        newSubMask = np.ones(nActive, dtype=np.bool_)
        newSubMask[idxSwitchOff] = False
        return _expand_masked(newSubMask, mask)

    return {
        "expand_masked": _expand_masked,
        "calc_lite": _calc_lite,
        "calc_update": _calc_update,
        "calc_importance_on_params": _calc_importance_on_params,
        "calc_importance_off_params": _calc_importance_off_params,
        "switch_off_params": _switch_off_params,
        "switch_on_params": _switch_on_params,
    }
# ** end function NumbaBackend
