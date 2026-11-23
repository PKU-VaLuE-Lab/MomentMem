"""Invoke the self-contained v3 converter for one generated task dataset."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


TASK_ENV_ROOT = Path(__file__).resolve().parent
V3_ROOT = TASK_ENV_ROOT.parent
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

from converter.specs import ALL_TASK_NAMES  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=ALL_TASK_NAMES, required=True)
    parser.add_argument("--name", required=True, help="input .h5/.json filename stem")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="dataset root; output is written to lerobot_format/<task>",
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    module = TASK_ENV_ROOT / args.task / f"{args.task}.py"
    if not module.is_file():
        print(f"[v3 run_convert] missing task module: {module}", file=sys.stderr)
        return 2
    command = [
        sys.executable,
        "-m",
        "converter",
        "--name",
        args.name,
        "--input-dir",
        str(args.input_dir.resolve()),
        "--output-root",
        str(args.output_root.resolve()),
        "--fps",
        str(args.fps),
        "--expected-task",
        args.task,
    ]
    if args.overwrite:
        command.append("--overwrite")
    print(f"[v3 run_convert] cwd={V3_ROOT} command={' '.join(command)}", flush=True)
    return subprocess.call(command, cwd=V3_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
