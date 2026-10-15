"""Preparation invariants and real official-evaluator checks when data is prepared."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gws import config
from gws.datasets import load_bird, load_spider1, load_spider200
from gws.evaluate_official import score_one
from gws.setup import extract_archive, validate_dataset


class PreparationTests(unittest.TestCase):
    def test_bundled_source_hashes(self):
        root = Path(__file__).resolve().parent.parent
        sources = json.loads((root/"vendor_sources.json").read_text())
        files = {p.name for p in (root/"baselines"/"vendor").iterdir() if p.is_file()}
        self.assertEqual(set(sources), files)
        for name, info in sources.items():
            self.assertEqual(hashlib.sha256((root/"baselines"/"vendor"/name).read_bytes()).hexdigest(),
                             info["sha256"], name)

    def test_archive_cannot_escape_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory)/"unsafe.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("../escape.txt", "bad")
            with self.assertRaises(ValueError):
                extract_archive(archive, Path(directory)/"output")
            self.assertFalse((Path(directory)/"escape.txt").exists())

    @unittest.skipUnless(Path(config.BIRD_DEV_JSON).is_file() and Path(config.BIRD_EVAL_SCRIPT).is_file(),
                         "Run reproduce.py --prepare-only for BIRD integration")
    def test_public_bird_and_official_comparator(self):
        validate_dataset(config.BIRD_ROOT, "bird")
        examples = load_bird()
        self.assertEqual(len(examples), 1534)
        ex = examples[0]
        self.assertEqual(score_one("bird", ex, ex.gold_sql), {"res": 1})
        self.assertEqual(score_one("bird", ex, "SELECT -987654321")["res"], 0)
        self.assertEqual(score_one("bird", ex, "DELETE FROM sqlite_master")["res"], 0)
        self.assertEqual(score_one("bird", ex, ex.gold_sql, "failure")["res"], 0)

    @unittest.skipUnless(Path(config.SPIDER1_DEV_JSON).is_file() and Path(config.SPIDER1_TESTSUITE).is_dir(),
                         "Run reproduce.py --prepare-only for Spider integration")
    def test_public_spider_cohort_and_official_comparator(self):
        validate_dataset(config.SPIDER1_ROOT, "spider")
        self.assertEqual(len(load_spider1()), 1034)
        cohort = load_spider200()
        self.assertEqual(len(cohort), 200)
        ex = cohort[0]
        self.assertEqual(score_one("spider200", ex, ex.gold_sql), {"res": 1})
        self.assertEqual(score_one("spider200", ex, "SELECT 1")["res"], 0)
        self.assertEqual(score_one("spider200", ex, ex.gold_sql, "failure")["res"], 0)


if __name__ == "__main__":
    unittest.main()
