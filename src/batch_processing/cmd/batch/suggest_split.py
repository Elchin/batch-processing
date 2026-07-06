from __future__ import annotations

from pathlib import Path

from batch_processing.cmd.base import BaseCommand
from batch_processing.utils.split_planning import build_split_plan, format_split_plan_report
from batch_processing.utils.utils import interpret_path


class SuggestSplitCommand(BaseCommand):
    def __init__(self, args):
        super().__init__()
        self._args = args

    def execute(self) -> None:
        total_years = (
            int(self._args.p)
            + int(self._args.e)
            + int(self._args.s)
            + int(self._args.t)
            + int(self._args.n)
        )
        pilot_batch_dir = getattr(self._args, "pilot_batch_dir", None)
        if pilot_batch_dir:
            pilot_batch_dir = Path(interpret_path(pilot_batch_dir))

        plan = build_split_plan(
            interpret_path(self._args.input_path),
            target_batches=int(self._args.target_batches),
            target_walltime_hours=getattr(self._args, "target_walltime_hours", None),
            mpi_ranks=int(getattr(self._args, "mpi_ranks", 8) or 8),
            total_years=total_years,
            pilot_batch_dir=pilot_batch_dir,
            pilot_hours=getattr(self._args, "pilot_hours", None),
            pilot_cells_override=getattr(self._args, "pilot_cells", None),
            cmt0_filter=bool(getattr(self._args, "cmt0_filter", False)),
            no_max_cmt=bool(getattr(self._args, "no_max_cmt", False)),
            max_cmt=int(getattr(self._args, "max_cmt", 74)),
            max_concurrent=int(getattr(self._args, "max_concurrent", 16)),
        )

        batches_path = getattr(self._args, "batches", None) or "<split_output_path>"
        print(format_split_plan_report(plan, batches_path=batches_path))
