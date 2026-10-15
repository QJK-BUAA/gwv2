"""Portable Ground--Explore--Commit integration check (no API or data download)."""
import unittest


if __name__ == "__main__":
    from test_runtime import RuntimeTests
    suite = unittest.TestSuite([RuntimeTests("test_pipeline_complete_contract_and_reads")])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
