import ast
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import porespy as ps
from porespy.tools import Settings, get_edt
from porespy.tools import _utils


@pytest.fixture(autouse=True)
def restore_ncores():
    original = ps.settings.ncores
    yield
    ps.settings.ncores = original


def _fallback_edt(monkeypatch, backend):
    real_import = importlib.import_module

    def import_module(name, *args, **kwargs):
        if name == "pyedt":
            raise ModuleNotFoundError(name)
        if name == "edt":
            return SimpleNamespace(edt=backend)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(_utils.importlib, "import_module", import_module)
    return get_edt()


def test_settings_singleton_is_not_reinitialized():
    ps.settings.ncores = 2
    assert Settings() is ps.settings
    assert Settings().ncores == 2


def test_existing_edt_callable_reads_current_setting(monkeypatch):
    calls = []

    def backend(image, **kwargs):
        calls.append(kwargs)
        return image

    edt = _fallback_edt(monkeypatch, backend)
    image = np.ones((3, 3), dtype=bool)
    ps.settings.ncores = 2
    edt(image)
    ps.settings.ncores = 4
    edt(image)
    edt(image, parallel=1)

    assert [call["parallel"] for call in calls] == [2, 4, 1]


def test_pyedt_backend_does_not_receive_parallel(monkeypatch):
    calls = []

    def backend(image, **kwargs):
        calls.append(kwargs)
        return image

    monkeypatch.setattr(
        _utils.importlib,
        "import_module",
        lambda name: SimpleNamespace(edt=backend),
    )
    edt = get_edt()
    ps.settings.ncores = 4
    edt(np.ones((3, 3), dtype=bool))
    assert calls == [{}]


class _ProcessWithAffinity:
    def __init__(self, cpus):
        self.cpus = cpus

    def cpu_affinity(self):
        return self.cpus


def test_cpu_count_prefers_affinity(monkeypatch):
    monkeypatch.setattr(
        _utils.psutil, "Process", lambda: _ProcessWithAffinity(range(3))
    )
    monkeypatch.setattr(_utils.psutil, "cpu_count", lambda logical: 12)
    monkeypatch.setattr(_utils, "_get_cgroup_cpu_limit", lambda: None)
    assert _utils._get_available_cpu_count() == 3


def test_cpu_count_falls_back_when_affinity_unavailable(monkeypatch):
    class Process:
        def cpu_affinity(self):
            raise NotImplementedError

    monkeypatch.setattr(_utils.psutil, "Process", Process)
    monkeypatch.setattr(_utils.psutil, "cpu_count", lambda logical: 6)
    monkeypatch.setattr(_utils, "_get_cgroup_cpu_limit", lambda: None)
    assert _utils._get_available_cpu_count() == 6


def test_cpu_count_is_at_least_one(monkeypatch):
    class Process:
        def cpu_affinity(self):
            raise AttributeError

    monkeypatch.setattr(_utils.psutil, "Process", Process)
    monkeypatch.setattr(_utils.psutil, "cpu_count", lambda logical: None)
    monkeypatch.setattr(_utils, "_get_cgroup_cpu_limit", lambda: None)
    assert _utils._get_available_cpu_count() == 1


def test_cpu_count_honors_cgroup_quota(monkeypatch):
    monkeypatch.setattr(
        _utils.psutil, "Process", lambda: _ProcessWithAffinity(range(8))
    )
    monkeypatch.setattr(_utils, "_get_cgroup_cpu_limit", lambda: 2)
    assert _utils._get_available_cpu_count() == 2


def test_cgroup_v2_fractional_quota_rounds_up(monkeypatch):
    def read_text(path):
        if str(path) == "/sys/fs/cgroup/cpu.max":
            return "150000 100000\n"
        raise FileNotFoundError

    monkeypatch.setattr(_utils.Path, "read_text", read_text)
    assert _utils._get_cgroup_cpu_limit() == 2


def test_regions_to_network_uses_configured_edt(monkeypatch):
    raw_edt = importlib.import_module("edt").edt
    calls = []

    def backend(image, **kwargs):
        calls.append(kwargs.copy())
        return raw_edt(image, **kwargs)

    edt = _fallback_edt(monkeypatch, backend)
    from porespy.networks import _getnet_orig

    monkeypatch.setattr(_getnet_orig, "edt", edt)
    ps.settings.ncores = 3
    regions = np.zeros((7, 8), dtype=int)
    regions[1:6, 1:4] = 1
    regions[1:6, 4:7] = 2
    _getnet_orig.regions_to_network(regions)

    assert len(calls) >= 2
    assert all(call["parallel"] == 3 for call in calls)


def test_production_code_has_no_raw_edt_imports_or_fixed_thread_counts():
    source_root = Path(ps.__file__).parent
    violations = []
    for filename in source_root.rglob("*.py"):
        tree = ast.parse(filename.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "edt":
                violations.append(f"raw EDT import in {filename}")
            if isinstance(node, ast.Call):
                name = getattr(node.func, "id", None)
                for keyword in node.keywords:
                    fixed = isinstance(keyword.value, ast.Constant)
                    if name == "edt" and keyword.arg == "parallel" and fixed:
                        violations.append(f"fixed parallel value in {filename}")
    assert violations == []
