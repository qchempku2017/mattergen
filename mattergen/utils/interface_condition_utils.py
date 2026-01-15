"""Implement automatic weighted mask computation for interfacial structures."""
import torch

from pymatgen.core.interface import Interface


def compute_interface_structure_reference_weighted_mask(
    interface_structure: Interface,
    fully_unmask_depth: float=3.0,
    partially_masked_depth: float=6.0,
):