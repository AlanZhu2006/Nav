#!/usr/bin/env python3
"""Recompute Phase-B session-level labels from the teacher authority.

Defect (found 2026-08-08, blocks jobs 15499307/15499309): the collector
derives ``session_max_covis`` from its own DINO shortlist instead of the
teacher's full candidate set.  A shortlist is a subset, so its maximum is a
lower bound, and a session whose best-covis candidate falls outside the
shortlist is silently recorded as a weaker class than it is.  Measured on the
audited artifacts:

  train  (top-2): 17/480 sessions differ, max drift 0.0896
  dev    (top-8):  9/120 sessions differ, max drift 0.7308, 2 class flips
                   (a positive session stored as strict no-match, and one
                   stored as ambiguous)

The per-row ``teacher_covis`` is already bit-identical to the teacher, so the
GPU collection does not need to be repeated: only three derived columns are
wrong and all three are pure aggregates of teacher data.

This script rewrites ``session_max_covis``, ``session_has_positive`` and
``session_is_strict_no_match`` from the teacher, writes a NEW artifact
directory (the input is never mutated), and records provenance plus the exact
set of repaired sessions.  It fails closed if the per-row teacher alignment is
not exact, because that would mean the artifact disagrees with the teacher on
something this repair cannot legitimately fix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROWS_NAME = "lingbot_goal_loop_closure_rows.csv"
CHECKPOINT_NAME = "lingbot_goal_loop_closure_checkpoint.sqlite3"
SESSION_KEY = ["scene", "session_id"]
ROW_KEY = ["scene", "session_id", "candidate_frame"]
REPAIRED_COLUMNS = (
    "session_max_covis",
    "session_has_positive",
    "session_is_strict_no_match",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repair(run_dir: Path, teacher_csv: Path, out_dir: Path,
           positive_threshold: float, negative_threshold: float) -> dict:
    rows_path = run_dir / ROWS_NAME
    rows = pd.read_csv(rows_path)
    teacher = pd.read_csv(teacher_csv)

    # Fail closed unless every collected candidate matches the teacher exactly:
    # this repair is only valid when the disagreement is confined to the
    # session-level aggregates.
    merged = rows.merge(
        teacher[ROW_KEY + ["teacher_covis"]].rename(
            columns={"teacher_covis": "authority_covis"}),
        on=ROW_KEY, how="left", validate="many_to_one")
    if merged["authority_covis"].isna().any():
        missing = int(merged["authority_covis"].isna().sum())
        raise RuntimeError(
            f"{missing} collected candidates are absent from the teacher")
    row_drift = float(np.max(np.abs(
        merged["teacher_covis"].to_numpy(dtype=float)
        - merged["authority_covis"].to_numpy(dtype=float))))
    if row_drift > 1e-9:
        raise RuntimeError(
            f"per-row teacher_covis disagrees by {row_drift}; the artifact "
            "needs recollection, not label repair")

    authority = teacher.groupby(SESSION_KEY, as_index=False)[
        "teacher_covis"].max().rename(
            columns={"teacher_covis": "authority_session_max"})
    repaired = rows.merge(
        authority, on=SESSION_KEY, how="left", validate="many_to_one")
    if repaired["authority_session_max"].isna().any():
        raise RuntimeError("a collected session is absent from the teacher")

    def classify(values: np.ndarray) -> np.ndarray:
        return np.where(values >= positive_threshold, "pos",
                        np.where(values <= negative_threshold, "neg", "amb"))

    before = repaired["session_max_covis"].to_numpy(dtype=float)
    after = repaired["authority_session_max"].to_numpy(dtype=float)
    changed = np.abs(before - after) > 1e-9
    flipped = classify(before) != classify(after)

    session_frame = repaired.loc[
        changed, SESSION_KEY + ["session_max_covis", "authority_session_max"]
    ].drop_duplicates()
    flip_frame = repaired.loc[
        flipped, SESSION_KEY + ["session_max_covis", "authority_session_max"]
    ].drop_duplicates()

    repaired["session_max_covis"] = after
    repaired["session_has_positive"] = after >= positive_threshold
    repaired["session_is_strict_no_match"] = after <= negative_threshold
    repaired = repaired.drop(columns=["authority_session_max"])
    if list(repaired.columns) != list(rows.columns):
        raise RuntimeError("column order changed during repair")

    out_dir.mkdir(parents=True, exist_ok=True)
    for source in run_dir.iterdir():
        if source.is_file() and source.name != ROWS_NAME:
            shutil.copy2(source, out_dir / source.name)
    out_rows = out_dir / ROWS_NAME
    repaired.to_csv(out_rows, index=False)

    # The resumable SQLite checkpoint carries the same payload and the auditor
    # cross-checks it against the CSV, so it has to be repaired in lockstep.
    checkpoint = out_dir / CHECKPOINT_NAME
    checkpoint_rows = 0
    if checkpoint.exists():
        authority_by_session = dict(
            zip(authority["session_id"], authority["authority_session_max"]))
        connection = sqlite3.connect(checkpoint)
        try:
            payloads = connection.execute(
                "SELECT seed_index, session_id, payload_json FROM rows"
            ).fetchall()
            updates = []
            for seed_index, session_id, payload_json in payloads:
                payload = json.loads(payload_json)
                maximum = authority_by_session.get(session_id)
                if maximum is None:
                    raise RuntimeError(
                        f"checkpoint session absent from teacher: {session_id}")
                payload["session_max_covis"] = float(maximum)
                payload["session_has_positive"] = bool(
                    maximum >= positive_threshold)
                payload["session_is_strict_no_match"] = bool(
                    maximum <= negative_threshold)
                updates.append((json.dumps(payload), seed_index))
            connection.executemany(
                "UPDATE rows SET payload_json = ? WHERE seed_index = ?",
                updates)
            connection.commit()
            checkpoint_rows = len(updates)
        finally:
            connection.close()

    report = {
        "repair": "phase_b_session_label_authority_v1",
        "reason": ("collector derived session_max_covis from its DINO "
                   "shortlist; the teacher full-candidate maximum is the "
                   "authority"),
        "source_run_dir": str(run_dir),
        "teacher_csv": str(teacher_csv),
        "teacher_sha256": sha256_file(teacher_csv),
        "source_rows_sha256": sha256_file(rows_path),
        "repaired_rows_sha256": sha256_file(out_rows),
        "repaired_checkpoint_rows": checkpoint_rows,
        "repaired_checkpoint_sha256": (
            sha256_file(checkpoint) if checkpoint.exists() else None),
        "repaired_columns": list(REPAIRED_COLUMNS),
        "positive_threshold": positive_threshold,
        "negative_threshold": negative_threshold,
        "rows": int(len(rows)),
        "sessions": int(rows["session_id"].nunique()),
        "sessions_changed": int(len(session_frame)),
        "sessions_class_flipped": int(len(flip_frame)),
        "max_abs_drift": float(np.max(np.abs(before - after))) if len(before) else 0.0,
        "changed_sessions": session_frame.to_dict(orient="records"),
        "class_flipped_sessions": flip_frame.to_dict(orient="records"),
    }
    (out_dir / "session_label_repair.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--teacher-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--positive-threshold", type=float, default=0.5)
    parser.add_argument("--negative-threshold", type=float, default=0.1)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite {args.out_dir}")
    report = repair(args.run_dir, args.teacher_csv, args.out_dir,
                    args.positive_threshold, args.negative_threshold)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("changed_sessions",
                                   "class_flipped_sessions")},
                     indent=2, sort_keys=True))
    for session in report["class_flipped_sessions"]:
        print("class flip:", session)


if __name__ == "__main__":
    main()
