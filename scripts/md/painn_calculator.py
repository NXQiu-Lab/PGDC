"""ASE Calculator wrapper for the PaiNN model trained by ``painn.py``.

This module lets any ASE workflow (geometry optimization, molecular dynamics,
phonon calculation, etc.) use a SchNetPack PaiNN checkpoint as an energy/force
engine. It is intentionally self-contained so that it can be imported by
stand-alone scripts without pulling in the full training pipeline.

Stress can be requested either from models explicitly trained on the 3x3 stress
tensor, or by activating the stress response on energy/force-only checkpoints
via ``schnetpack.md.utils.activate_model_stress``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from schnetpack.interfaces import AtomsConverter
from schnetpack.representation.painn import PaiNN
from schnetpack.transform import ASENeighborList

import schnetpack as spk


# ---------------------------------------------------------------------------
# schnetpack <2.2.0 compatibility patch (mirrors ``painn.py``)
# ---------------------------------------------------------------------------


def _apply_schnetpack_compat() -> None:
    """Patch old schnetpack transforms that break under PyTorch 2.x."""
    import inspect
    import textwrap

    import schnetpack.transform.atomistic as _atm

    def _has_or_bug(cls, **kwargs):
        try:
            cls("dummy", **kwargs)
            return False
        except RuntimeError as e:
            return "Boolean value of Tensor with more than one value is ambiguous" in str(e)
        except Exception:
            return False

    def _patch_init(cls):
        try:
            src = inspect.getsource(cls.__init__)
        except (OSError, TypeError):
            return
        src = textwrap.dedent(src)
        fixed = src.replace(
            "atomrefs = atomrefs or torch.zeros((zmax,))",
            "atomrefs = atomrefs if atomrefs is not None else torch.zeros((zmax,))",
        ).replace(
            "property_mean = property_mean or torch.zeros((1,))",
            "property_mean = property_mean if property_mean is not None else torch.zeros((1,))",
        ).replace(
            "scale = scale or torch.ones((1,))",
            "scale = scale if scale is not None else torch.ones((1,))",
        ).replace(
            "super().__init__()",
            f"super({cls.__name__}, self).__init__()",
        )
        if fixed == src:
            return
        namespace = dict(cls.__init__.__globals__)
        namespace[cls.__name__] = cls
        exec(fixed, namespace)
        cls.__init__ = namespace["__init__"]

    if _has_or_bug(_atm.RemoveOffsets, remove_atomrefs=True, atomrefs=torch.zeros(5)):
        _patch_init(_atm.RemoveOffsets)

    if _has_or_bug(_atm.AddOffsets, add_atomrefs=True, atomrefs=torch.zeros(5)):
        _patch_init(_atm.AddOffsets)

    if hasattr(_atm, "ScaleProperty"):
        try:
            _atm.ScaleProperty("dummy", scale=torch.zeros(5))
        except RuntimeError as e:
            if "Boolean value of Tensor with more than one value is ambiguous" in str(e):
                _patch_init(_atm.ScaleProperty)
        except Exception:
            pass


_apply_schnetpack_compat()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_painn_model(model_path: str | Path, device: str = "cpu") -> Any:
    """Load a SchNetPack / PaiNN model saved by ``painn.py``.

    Handles the ``electronic_embeddings`` compatibility issue that appears
    with older PaiNN checkpoints.

    Args:
        model_path: Path to the saved torch model (full object, not state_dict).
        device: "cpu" or "cuda".

    Returns:
        The loaded model in eval mode on ``device``.
    """
    model = torch.load(
        str(model_path),
        map_location=device,
        weights_only=False,
    )

    for module in model.modules():
        if isinstance(module, PaiNN) and not hasattr(module, "electronic_embeddings"):
            module.electronic_embeddings = nn.ModuleList([])

    model.eval()
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# ASE Calculator
# ---------------------------------------------------------------------------


class PaiNNCalculator(Calculator):
    """ASE Calculator backed by a SchNetPack PaiNN potential.

    The model is expected to produce ``energy`` (eV) and ``forces`` (eV/Å).
    Stress (eV/Å³) is additionally supported when the model was trained with
    ``--stress-weight > 0`` or when ``calc_stress=True`` is used to activate the
    stress response at inference time.

    Only fully periodic structures (``pbc = [True, True, True]``) are supported,
    because the neighbor list is configured with the ASE periodic convention.

    Args:
        model_path: Path to the saved PaiNN model.
        device: "cpu" or "cuda".
        cutoff: Neighbor-list cutoff in Å. Must match the training cutoff.
        calc_stress: If True, compute the 3x3 stress tensor. For models not
            trained on stress, this activates the strain response automatically.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
        cutoff: float = 5.0,
        calc_stress: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.device = device
        self.cutoff = cutoff
        self.model = load_painn_model(model_path, device=device)

        model_outputs = getattr(self.model, "model_outputs", [])
        self.calc_stress = calc_stress or spk.properties.stress in model_outputs
        if self.calc_stress and spk.properties.stress not in model_outputs:
            from schnetpack.md.utils import activate_model_stress

            self.model = activate_model_stress(self.model, spk.properties.stress)

        props = list(self.implemented_properties)
        if self.calc_stress:
            props.append("stress")
        self.implemented_properties = props

        self.converter = AtomsConverter(
            neighbor_list=ASENeighborList(cutoff=cutoff),
            device=device,
        )

    def _to_ase_atoms(self, phonopy_atoms) -> Atoms:
        """Convert a PhonopyAtoms object to an ASE Atoms object."""
        return Atoms(
            symbols=phonopy_atoms.symbols,
            positions=phonopy_atoms.positions,
            cell=phonopy_atoms.cell,
            pbc=True,
        )

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        if properties is None:
            properties = ["energy"]
        super().calculate(atoms, properties, system_changes)

        if not self.atoms.pbc.all():
            raise ValueError(
                f"{self.__class__.__name__} requires full periodic boundary conditions "
                "(pbc=[True, True, True])."
            )

        # Work on a wrapped copy so that atoms crossing a cell boundary do not
        # confuse the neighbor list.
        atoms_wrapped = self.atoms.copy()
        atoms_wrapped.wrap(eps=1e-12)

        inputs = self.converter(atoms_wrapped)
        inputs["_positions"].requires_grad_(True)

        with torch.set_grad_enabled(True):
            pred = self.model(inputs)

        energy = pred[spk.properties.energy].detach().cpu().numpy().reshape(())
        forces = pred[spk.properties.forces].detach().cpu().numpy()

        self.results = {
            "energy": float(energy),
            "forces": forces.astype(float),
            "free_energy": float(energy),
        }

        if self.calc_stress and spk.properties.stress in pred:
            stress = pred[spk.properties.stress].detach().cpu().numpy().squeeze()
            self.results["stress"] = stress.astype(float)
