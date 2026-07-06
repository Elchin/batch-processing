"""Tests for Option 2 restart validation helpers in wiemip_end_to_end."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "batch_processing"
    / "extra"
    / "wiemip_end_to_end.py"
)


def _load_wiemip_end_to_end():
    spec = importlib.util.spec_from_file_location("wiemip_end_to_end", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["wiemip_end_to_end"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def e2e():
    return _load_wiemip_end_to_end()


def test_is_staging_input_path_detects_staging_dirname(e2e, tmp_path):
    staging = tmp_path / e2e.STAGING_INPUT_DIRNAME
    staging.mkdir()
    (staging / "run-mask.nc").write_bytes(b"")
    assert e2e._is_staging_input_path(staging) is True


def test_is_staging_input_path_detects_nested_under_split(e2e, tmp_path):
    split_root = tmp_path / "my_split"
    split_root.mkdir()
    staging = split_root / e2e.STAGING_INPUT_DIRNAME
    staging.mkdir()
    (staging / "run-mask.nc").write_bytes(b"")
    (split_root / e2e.WIEMIP_SPLIT_METADATA_FILENAME).write_text("{}")
    (split_root / "batch_0").mkdir()
    assert e2e._is_staging_input_path(staging) is True


def test_is_staging_input_path_false_for_plain_setup(e2e, tmp_path):
    setup = tmp_path / "setup_stable"
    setup.mkdir()
    (setup / "run-mask.nc").write_bytes(b"")
    assert e2e._is_staging_input_path(setup) is False


def test_validate_option2_restart_raises_on_batch_count_mismatch(e2e, tmp_path):
    restart_root = tmp_path / "source_split"
    restart_root.mkdir()
    (restart_root / "batch_0").mkdir()
    (restart_root / "batch_1").mkdir()

    input_path = tmp_path / "setup"
    input_path.mkdir()

    with pytest.raises(ValueError, match="split at --split has 3"):
        e2e._validate_option2_restart(
            input_path=input_path,
            restart_from_path=restart_root,
            nbatches=3,
            restart_file="restart-sp.nc",
        )
