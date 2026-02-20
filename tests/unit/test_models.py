from __future__ import annotations

import pandas as pd
import pytest

from bayesprism.models import Prism, RefPhi, ValidationError


def test_prism_validates_gene_alignment() -> None:
    phi_cs = RefPhi(pd.DataFrame([[0.5, 0.5]], index=["s1"], columns=["g1", "g2"]), pseudo_min=0.5)
    phi_ct = RefPhi(pd.DataFrame([[0.5, 0.5]], index=["t1"], columns=["g1", "g2"]), pseudo_min=0.5)
    mixture = pd.DataFrame([[10, 12]], index=["m1"], columns=["g1", "g2"])

    Prism(
        phi_cell_state=phi_cs,
        phi_cell_type=phi_ct,
        map={"t1": ["s1"]},
        key=None,
        mixture=mixture,
    )


def test_prism_raises_on_gene_mismatch() -> None:
    phi_cs = RefPhi(pd.DataFrame([[0.5, 0.5]], index=["s1"], columns=["g1", "g2"]), pseudo_min=0.5)
    phi_ct = RefPhi(pd.DataFrame([[0.5, 0.5]], index=["t1"], columns=["g1", "g2"]), pseudo_min=0.5)
    mixture = pd.DataFrame([[10, 12]], index=["m1"], columns=["g1", "g3"])

    with pytest.raises(ValidationError):
        Prism(
            phi_cell_state=phi_cs,
            phi_cell_type=phi_ct,
            map={"t1": ["s1"]},
            key=None,
            mixture=mixture,
        )
