#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Dataset build records: written once, verified on resume, required by training."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_record import BUILD_INFO, claim_destination, new_record, read_records  # noqa: E402

RECORD = {"revision": "abc123", "dirty": False, "options": {"top_n": None}, "date": "2026-10-05T00:00:00+00:00"}


class ClaimDestinationTest(unittest.TestCase):
    def test_a_new_directory_gets_the_record_and_a_matching_one_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            claim_destination(d, RECORD)
            self.assertEqual(json.loads((Path(d) / BUILD_INFO).read_text()), RECORD)
            (Path(d) / "a.npy").write_bytes(b"x")
            claim_destination(d, {**RECORD, "date": "later", "dirty": True})   # same revision and options

    def test_data_without_a_record_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.npy").write_bytes(b"x")
            with self.assertRaises(SystemExit) as ctx:
                claim_destination(d, RECORD)
            self.assertIn("older builder", str(ctx.exception))
            self.assertFalse((Path(d) / BUILD_INFO).exists())

    def test_another_revision_or_other_options_are_refused(self) -> None:
        for changed in ({"revision": "def456"}, {"options": {"top_n": 10}}):
            with tempfile.TemporaryDirectory() as d:
                claim_destination(d, RECORD)
                with self.assertRaises(SystemExit) as ctx:
                    claim_destination(d, {**RECORD, **changed})
                self.assertIn("new, empty directory", str(ctx.exception))


class ReadRecordsTest(unittest.TestCase):
    def test_records_are_read_and_the_image_limit_suffix_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            claim_destination(d, RECORD)
            self.assertEqual(read_records([f"{d}:1500"]), {d: RECORD})

    def test_a_directory_without_a_record_is_named(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit) as ctx:
                read_records([d])
            self.assertIn(d, str(ctx.exception))
            self.assertIn(BUILD_INFO, str(ctx.exception))


class NewRecordTest(unittest.TestCase):
    def test_record_fields(self) -> None:
        record = new_record({"top_n": 5}, str(REPO_ROOT))
        self.assertEqual(set(record), {"revision", "dirty", "options", "date"})
        self.assertEqual(record["options"], {"top_n": 5})
        self.assertIsInstance(record["dirty"], bool)


if __name__ == "__main__":
    unittest.main()
