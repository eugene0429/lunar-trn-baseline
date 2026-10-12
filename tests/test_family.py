import numpy as np

from lunar_trn.localize import LocalizationResult


def test_result_family_defaults_to_ti():
    r = LocalizationResult("no_fix", np.eye(4), np.full((6, 6), np.nan), {}, {}, "coarse")
    assert r.family == "T-I"


def test_result_family_can_be_set():
    r = LocalizationResult("fix", np.eye(4), np.eye(6), {}, {}, "pose", family="P-G")
    assert r.family == "P-G"


def test_craters_package_imports():
    import lunar_trn.craters  # noqa: F401
