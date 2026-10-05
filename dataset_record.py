# SPDX-License-Identifier: MIT
# Copyright (c) 2024-present X-Veon contributors
"""
The record a dataset directory carries about how it was built.

build_dataset.py writes it and refuses a directory holding data it did not record;
train.py reads it and refuses a directory without one. Together they make it impossible
to train on targets from an older builder without noticing.
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
from typing import Any

BUILD_INFO = "build_info.json"


def git_revision(repo_dir: str) -> tuple[str, bool]:
    """(HEAD revision, whether tracked files are modified) of the repository at repo_dir."""
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo_dir, capture_output=True, text=True).stdout.strip()
    return run("rev-parse", "HEAD") or "unknown", bool(run("status", "--porcelain", "--untracked-files=no"))


def new_record(options: dict[str, Any], repo_dir: str) -> dict[str, Any]:
    revision, dirty = git_revision(repo_dir)
    return {
        "revision": revision,
        "dirty": dirty,
        "options": options,
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }


def claim_destination(output_dir: str, record: dict[str, Any]) -> None:
    """Write the record into a new destination, or check the one already there.

    Exits when the directory holds .npy files without a record, or with a record from
    another revision or other options. An interrupted build of the same revision and
    options can be resumed.
    """
    info_path = os.path.join(output_dir, BUILD_INFO)
    if os.path.exists(info_path):
        with open(info_path) as f:
            existing = json.load(f)
        if existing.get("revision") != record["revision"] or existing.get("options") != record["options"]:
            sys.exit(
                f"{output_dir} was built by revision {existing.get('revision')} with options "
                f"{existing.get('options')}; this run is revision {record['revision']} with options "
                f"{record['options']}. Use a new, empty directory."
            )
        return
    if any(name.endswith(".npy") for name in os.listdir(output_dir)):
        sys.exit(
            f"{output_dir} already holds .npy files and has no {BUILD_INFO}: it was built by an "
            f"older builder. Use a new, empty directory."
        )
    with open(info_path, "w") as f:
        json.dump(record, f, indent=2)


def read_records(data_dirs: list[str]) -> dict[str, dict[str, Any]]:
    """Build records of training data directories (a trailing ':N' image limit is ignored).

    Exits, naming the directory, when one has no record.
    """
    records: dict[str, dict[str, Any]] = {}
    for entry in data_dirs:
        path = entry.rsplit(":", 1)[0] if ":" in entry and entry.rsplit(":", 1)[1].isdigit() else entry
        info_path = os.path.join(path, BUILD_INFO)
        if not os.path.exists(info_path):
            sys.exit(
                f"{path} has no {BUILD_INFO}: it was not built by the current build_dataset.py. "
                f"Rebuild it into a new directory."
            )
        with open(info_path) as f:
            records[path] = json.load(f)
    return records
