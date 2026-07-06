"""Tests for batch submission helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from batch_processing.utils.utils import (
    BatchSubmitOptions,
    extract_sbatch_job_id,
    submit_batch_jobs,
    wait_for_job_ids,
)


def test_extract_sbatch_job_id_from_stdout():
    assert extract_sbatch_job_id("Submitted batch job 12345\n") == "12345"


@patch("batch_processing.utils.utils.time.sleep")
@patch("batch_processing.utils.utils.subprocess.run")
def test_wait_for_job_ids_until_queue_clear(mock_run, _mock_sleep):
    mock_run.side_effect = [
        MagicMock(returncode=0, stdout="12345\n"),
        MagicMock(returncode=0, stdout=""),
    ]

    wait_for_job_ids(["12345"], poll_interval_seconds=1)

    assert mock_run.call_count == 2


@patch("batch_processing.utils.utils.time.sleep")
@patch("batch_processing.utils.utils.submit_job")
def test_submit_batch_jobs_submit_all_skips_capacity_wait(
    mock_submit_job,
    _mock_sleep,
    tmp_path: Path,
):
    scripts = []
    for batch_id in (1, 2, 3):
        batch_dir = tmp_path / f"batch_{batch_id}"
        batch_dir.mkdir()
        script = batch_dir / "slurm_runner.sh"
        script.write_text("#!/bin/bash\n", encoding="utf-8")
        scripts.append(script)

    mock_submit_job.side_effect = [
        MagicMock(returncode=0, stdout="Submitted batch job 101\n", stderr=""),
        MagicMock(returncode=0, stdout="Submitted batch job 102\n", stderr=""),
        MagicMock(returncode=0, stdout="Submitted batch job 103\n", stderr=""),
    ]

    with patch("batch_processing.utils.utils.wait_for_submit_capacity") as mock_wait:
        submitted, failed, skipped = submit_batch_jobs(
            scripts,
            BatchSubmitOptions(submit_all=True, submit_delay_seconds=0),
        )

    assert submitted == 3
    assert failed == 0
    assert skipped == 0
    assert mock_submit_job.call_count == 3
    mock_wait.assert_not_called()


@patch("batch_processing.utils.utils.time.sleep")
@patch("batch_processing.utils.utils.wait_for_submit_capacity")
@patch("batch_processing.utils.utils.submit_job")
def test_submit_batch_jobs_throttle_waits_for_capacity(
    mock_submit_job,
    mock_wait_for_capacity,
    _mock_sleep,
    tmp_path: Path,
):
    batch_dir = tmp_path / "batch_1"
    batch_dir.mkdir()
    script = batch_dir / "slurm_runner.sh"
    script.write_text("#!/bin/bash\n", encoding="utf-8")

    mock_submit_job.return_value = MagicMock(
        returncode=0,
        stdout="Submitted batch job 101\n",
        stderr="",
    )

    submitted, failed, skipped = submit_batch_jobs(
        [script],
        BatchSubmitOptions(submit_all=False, submit_delay_seconds=0),
    )

    assert submitted == 1
    assert failed == 0
    assert skipped == 0
    mock_wait_for_capacity.assert_called_once()


@patch("batch_processing.utils.utils.is_batch_complete", return_value=True)
@patch("batch_processing.utils.utils.submit_job")
def test_submit_batch_jobs_skip_complete(
    mock_submit_job,
    _mock_is_complete,
    tmp_path: Path,
):
    batch_dir = tmp_path / "batch_0"
    batch_dir.mkdir()
    script = batch_dir / "slurm_runner.sh"
    script.write_text("#!/bin/bash\n", encoding="utf-8")

    submitted, failed, skipped = submit_batch_jobs(
        [script],
        BatchSubmitOptions(skip_complete=True, submit_delay_seconds=0),
    )

    assert submitted == 0
    assert failed == 0
    assert skipped == 1
    mock_submit_job.assert_not_called()


@patch("batch_processing.utils.utils.submit_job")
def test_submit_batch_jobs_reports_failure(mock_submit_job, tmp_path: Path):
    batch_dir = tmp_path / "batch_0"
    batch_dir.mkdir()
    script = batch_dir / "slurm_runner.sh"
    script.write_text("#!/bin/bash\n", encoding="utf-8")

    mock_submit_job.return_value = MagicMock(
        returncode=1,
        stdout="",
        stderr="sbatch: error",
    )

    submitted, failed, skipped = submit_batch_jobs(
        [script],
        BatchSubmitOptions(submit_delay_seconds=0),
    )

    assert submitted == 0
    assert failed == 1
    assert skipped == 0
