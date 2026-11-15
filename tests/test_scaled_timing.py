"""scripts/scaled_timing.py loads run_experiments.py with the scale protocol's k set to 5."""

from __future__ import annotations

import importlib.util

from wikilense.config import REPO_ROOT

spec = importlib.util.spec_from_file_location("scaled_timing",
                                              REPO_ROOT / "scripts" / "scaled_timing.py")
timing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timing)


def test_the_scale_protocol_runs_at_k_5():
    rx = timing.experiments_module()
    assert rx.SCALE_K == 5 == timing.K
    assert rx.SCALE_REPEATS == 3 and rx.EF_CHOSEN == 100
    assert callable(rx.measure_scale)
