from pathlib import Path

from batch_processing.cmd.base import BaseCommand
from batch_processing.cmd.elapsed import ElapsedCommand
from batch_processing.utils.utils import (
    BatchSubmitOptions,
    interpret_path,
    submit_batch_jobs,
)


class BatchRunCommand(BaseCommand):
    def __init__(self, args):
        super().__init__()
        self._args = args
        self.base_batch_dir = Path(interpret_path(args.batches))
        self._args.base_batch_dir = self.base_batch_dir

    def execute(self):
        full_paths = sorted(self.base_batch_dir.glob("*/slurm_runner.sh"))
        if len(full_paths) == 0:
            print(
                "Couldn't find any slurm_runner scripts. ",
                f"Is {self._args.batches} the correct path?",
            )
            exit(1)

        throttle = bool(getattr(self._args, "throttle", False))
        options = BatchSubmitOptions(
            submit_all=not throttle,
            max_concurrent=getattr(self._args, "max_concurrent", 16),
            max_queue_depth=getattr(self._args, "max_queue_depth", 32),
            submit_delay_seconds=getattr(self._args, "submit_delay", 0.25),
            poll_interval_seconds=getattr(self._args, "poll_interval", 30),
            skip_complete=getattr(self._args, "skip_complete", False),
            dry_run=getattr(self._args, "dry_run", False),
        )

        submitted, failed, _skipped = submit_batch_jobs(full_paths, options)
        if failed:
            print(f"[WARN] {failed} batch job(s) failed to submit.")
        if submitted == 0 and failed == 0:
            print("[SUBMIT] No jobs were submitted.")

        if not options.dry_run and submitted > 0:
            ElapsedCommand(self._args).execute()
