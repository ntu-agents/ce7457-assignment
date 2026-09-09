#!/usr/bin/env python3
"""Run from this directory: uv run make_submission.py

Zips the required files below, plus any extra .py helpers under submission/, into submission.zip.
Run the public tests yourself before submitting; this script does not run them.
"""
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

REQUIRED = [
    "__init__.py",
    "answer_sheet.py",
    "constants.py",
    "question1/__init__.py",
    "question1/mdp.py",
    "question1/mdp_solver.py",
    "question2/__init__.py",
    "question2/agents.py",
    "question2/train_monte_carlo.py",
    "question2/train_q_learning.py",
    "question2/utils.py",
    "question3/__init__.py",
    "question3/agents.py",
    "question3/evaluate_dqn.py",
    "question3/evaluate_reinforce.py",
    "question3/networks.py",
    "question3/replay.py",
    "question3/train_dqn.py",
    "question3/train_reinforce.py",
    "question4/__init__.py",
    "question4/agents.py",
    "question4/evaluate_ppo.py",
    "question4/rollout.py",
    "question4/train_ppo.py",
    "question4/lunarlander_latest.pt",
    "question5/__init__.py",
    "question5/evaluate_ppo.py",
    "question5/train_ppo.py",
    "question5/lunarlander_hparam_latest.pt",
    "util/hparam_sweeping.py",
    "util/result_processing.py",
]
SKIP = {"__pycache__", "venv", "env", "site-packages"}


def main():
    root = Path(__file__).resolve().parent
    submission = root / "submission"
    present = [name for name in REQUIRED if (submission / name).is_file()]
    missing = [name for name in REQUIRED if name not in present]
    if not present:
        sys.exit("No required files found under submission/. Run this script from the assignment directory.")

    files = {submission / name for name in present}
    files.update(
        path for path in submission.rglob("*.py")
        if path.is_file() and not any(
            part.startswith(".") or part in SKIP for part in path.relative_to(submission).parts
        )
    )
    output = root / "submission.zip"
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(root).as_posix())
    print(f"Created {output} ({len(files)} files)")

    if missing:
        print(
            f"\nWARNING: submission.zip is INCOMPLETE. {len(missing)} required file(s) are missing "
            "and will earn no marks:\n  " + "\n  ".join(f"submission/{name}" for name in missing),
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
