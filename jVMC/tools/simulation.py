import jax
import jax.numpy as jnp
import jVMC

from enum import Enum
from typing import Annotated, Any, Literal, Optional, Union

import tomli as tomllib
from pydantic import BaseModel, Field, model_validator, ValidationError

import sys

# ================== IO ====================

class IOConfig(BaseModel):
    checkpoint: Optional[str]
    output: str

# ================ ANSATZ ==================

class CpxRBMParams(BaseModel):
    numHidden: int
    bias: bool

class CpxRBM_TIParams(BaseModel):
    numHidden: int
    bias: bool

class CpxVisionTransformerParams(BaseModel):
    patch_len: int
    embed_dim: int
    n_layers: int
    n_heads: int
    n_layers_ff: int

    @model_validator(mode="after")
    def check_n_heads(self) -> "CpxVisionTransformerParams":
        if self.embed_dim % self.n_heads != 0:
            raise ValueError(f"Embedding dimension must be divisible by number of attention heads")
        return self

class AnsatzBase(BaseModel):
    def build(self):
        NETS = {
            "CpxRBM": jVMC.nets.rbm.CpxRBM,
            "CpxRBM_TI": jVMC.nets.rbm.CpxRBM_TI,
            "CpxVisionTransformer": jVMC.nets.transformer.CpxVisionTransformer,
        }

        if self.net not in NETS:
            raise ValueError(f"Net type: {self.net} is not supported")
        return NETS[self.net](**self.parameters.model_dump())

class CpxRBMConfig(AnsatzBase):
    net: Literal["CpxRBM"]
    parameters: CpxRBMParams

class CpxRBM_TIConfig(AnsatzBase):
    net: Literal["CpxRBM_TI"]
    parameters: CpxRBM_TIParams

class CpxVisionTransformerConfig(AnsatzBase):
    net: Literal["CpxVisionTransformer"]
    parameters: CpxVisionTransformerParams

AnsatzConfig = Annotated[
    Union[CpxRBMConfig, CpxRBM_TIConfig, CpxVisionTransformerConfig],
    Field(discriminator="net")
]

# ============= PHYSICAL SYSTEM =============

class PhysicalSystemConfig(BaseModel):
    L: int
    g: float

# =========== EQUATION OF MOTION =============

class SRParams(BaseModel):
    diagonalShift: float = Field(ge=0)
    makeReal: str
    diagonalizeOnDevice: bool

class tVMCParams(BaseModel):
    diagonalShift: float = Field(ge=0)
    pinvCutoff: float = Field(ge=0)
    makeReal: str
    diagonalizeOnDevice: bool

class atVMCParams(BaseModel):
    diagonalShift: float = Field(ge=0)
    pinvCutoff: float = Field(ge=0)
    liteCutoff: float = Field(ge=0)
    paramImportanceCutoff: Optional[float] = None
    minSwitchOff: int = Field(ge=0)
    backend: str

class minSRParams(BaseModel):
    diagonalShift: float = Field(ge=0)
    pinvTol: float = Field(ge=0)
    diagonalizeOnDevice: bool

class BaseEqOfMotion(BaseModel):
    def build(self, sampler):
        if self.mode == "SR":
            return jVMC.util.TDVP(sampler, rhsPrefactor=1., **self.parameters.model_dump())
        elif self.mode == "tVMC":
            return jVMC.util.TDVP(sampler, rhsPrefactor=1.j, **self.parameters.model_dump())
        elif self.mode == "atVMC":
            return jVMC.util.aTDVP(sampler, rhsPrefactor=1.j, makeReal="real", **self.parameters.model_dump(), mpiRoot=0)
        elif self.mode == "minSR":
            return jVMC.util.MinSR(sampler, **self.parameters.model_dump())
        else:
            raise ValueError(f"Algorithm {self.mode} is not supported")


class SRConfig(BaseEqOfMotion):
    mode: Literal["SR"]
    parameters: SRParams

class tVMCConfig(BaseEqOfMotion):
    mode: Literal["tVMC"]
    parameters: tVMCParams

class atVMCConfig(BaseEqOfMotion):
    mode: Literal["atVMC"]
    parameters: atVMCParams

class minSRConfig(BaseEqOfMotion):
    mode: Literal["minSR"]
    parameters: minSRParams

EqOfMotionConfig = Annotated[
    Union[SRConfig, tVMCConfig, atVMCConfig, minSRConfig],
    Field(discriminator="mode"),
]


# ================ INTEGRATOR ===================

class StepperType(str, Enum):
    Euler = "Euler"
    Heun = "Heun"

class StepperParams(BaseModel):
    timeStep: float = Field(gt=0)
    nSteps: int = Field(gt=0)

class StepperConfig(BaseModel):
    mode: StepperType
    parameters: StepperParams

    def build(self):
        if self.mode == StepperType.Euler:
            return jVMC.util.Euler(timeStep=self.parameters.timeStep)
        elif self.mode == StepperType.Heun:
            return jVMC.util.Heun(timeStep=self.parameters.timeStep)
        else:
            raise ValueError(f"Stepper: {self.stepper} is not supported")


# ================ SAMPLER =======================

class SamplerType(str, Enum):
    MC = "MC"

class UpdateProposerType(str, Enum):
    spin_flip_Z2 = "spin_flip_Z2"

class UpdateProposerConfig(BaseModel):
    updateProposer: UpdateProposerType

    def build(self):
        if self.updateProposer == UpdateProposerType.spin_flip_Z2:
            return jVMC.sampler.propose_spin_flip_Z2
        else:
            raise ValueError(f"Update proposer: {self.updateProposer} not supported")

class SamplerSettings(BaseModel):
    numSamples: int = Field(gt=0)
    numChains: int = Field(gt=0)
    sweepSteps: int = Field(gt=0)
    thermalizationSweeps: int = Field(ge=0)

    @model_validator(mode="after")
    def check_samples_chains(self) -> "SamplerSettings":
        if self.numSamples < self.numChains:
            raise ValueError(f"numSamples must be greater than numChains")
        return self


class SamplerConfig(BaseModel):
    mode: SamplerType
    settings: SamplerSettings
    proposer: UpdateProposerConfig

    def build(self, psi, L, random_key):
        if self.mode == SamplerType.MC:
            return jVMC.sampler.MCSampler(
                psi, (L,), random_key,
                **self.settings.model_dump(),
                updateProposer=self.proposer.build(),
            )
        else:
            raise ValueError(f"Sampler mode: {self.mode} not supported")


# ======= OBSERVABLE MEASUREMENTS ==========

class MeasurementConfig(BaseModel):
    measurementSamples: int = Field(ge=0)

# =============== CONFIG ===================

class Config(BaseModel):
    IO: IOConfig
    ansatz: AnsatzConfig
    physical_system: PhysicalSystemConfig
    eq_of_motion: EqOfMotionConfig
    integrator: StepperConfig
    sampler: SamplerConfig
    measurements: MeasurementConfig

    @model_validator(mode="after")
    def check_patch_dim_if_transformer(self) -> "Config":
        if self.ansatz.net == "CpxVisionTransformer":
            if self.physical_system.L  % self.ansatz.parameters.patch_len != 0:
                raise ValueError(f"Number of spins must be divisible by length of transformer patch")
        return self

def load_config(path: str):
    with open(path, 'rb') as f:
        data = tomllib.load(f)

    try:
        config = Config.model_validate(data)
    except ValidationError as e:
        for error in e.errors():
            path = ".".join(str(item) for item in error['loc'])
            print(f"  [{path}] -> {error['msg']}")
        sys.exit(1)

    return config


# ================ MAIN ====================

def main():
    if len(sys.argv) < 2:
        print("Usage: simulate <config.toml>")
        return

    # Parse configuration file
    config = load_config(sys.argv[1])

    # Checkpoint (if provided) and output file
    inputManager = jVMC.util.output_manager.OutputManager(config.IO.checkpoint, append=True) if config.IO.checkpoint else None
    outputManager = jVMC.util.output_manager.OutputManager(config.IO.output)

    # Variational quantum state
    net = config.ansatz.build()
    psi = jVMC.vqs.NQS(net, seed=1)

    # Physical system
    L = config.physical_system.L
    g = config.physical_system.g

    # Initialize network from checkpoint if provided
    if inputManager is not None:
        print("Initializing from checkpoint")
        _, checkpoint_params = inputManager.get_network_checkpoint()

        dummy_spins = jnp.zeros((L,))
        psi.init_net(dummy_spins[None, None, :]) # Add two leading axes for device and batch dimensions
        psi.set_parameters(checkpoint_params)
    del inputManager

    # Observables
    hamiltonian = jVMC.operator.BranchFreeOperator()
    for l in range(L):
        hamiltonian.add(jVMC.operator.scal_opstr(-1., (jVMC.operator.Sz(l), jVMC.operator.Sz((l + 1) % L))))
        hamiltonian.add(jVMC.operator.scal_opstr(g, (jVMC.operator.Sx(l), )))

    magnetization = jVMC.operator.BranchFreeOperator()
    for l in range(L):
        magnetization.add(jVMC.operator.scal_opstr(1./L, (jVMC.operator.Sx(l),)))

    observables = {"magnetization": magnetization, "energy": hamiltonian}

    # Sampler
    sampler = config.sampler.build(psi, L, jax.random.key(8))

    # Equation of motion
    tdvpEquation = config.eq_of_motion.build(sampler)

    # Integrator
    stepper = config.integrator.build()
    N_STEPS = config.integrator.parameters.nSteps

    # Measurement samples
    MEASUREMENT_SAMPLES = config.measurements.measurementSamples

    # Simulation loop
    for step in range(N_STEPS):
        updatedParams, _ = stepper.step(0, tdvpEquation, psi.get_parameters(), hamiltonian=hamiltonian, psi=psi, outp=outputManager)
        psi.set_parameters(updatedParams)

        energy_per_spin = jax.numpy.real(tdvpEquation.ElocMean0) / L
        var_energy_per_spin = tdvpEquation.ElocVar0 / L
        print(f"Step: {step}\tEnergy: {energy_per_spin}")

        measurements = jVMC.util.measure(observables, psi=psi, sampler=sampler, numSamples=MEASUREMENT_SAMPLES)
        measurements["energy"]["mean(time ev samples)"] = energy_per_spin
        measurements["energy"]["variance(time ev samples)"] = var_energy_per_spin

        outputManager.write_observables(step, **measurements)
        outputManager.write_metadata(step, **tdvpEquation.metadata)

        if step == N_STEPS - 1:
            outputManager.write_network_checkpoint(step, psi.get_parameters())

    outputManager.print_timings()
    for name, value in outputManager.timings.items():
        outputManager.write_attribute(name, value["total"], "timings")

    return


if __name__ == "__main__":
    main()
