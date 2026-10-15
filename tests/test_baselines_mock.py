"""Source-backed baseline smoke checks on a temporary real SQLite database."""
import unittest


if __name__ == "__main__":
    from test_runtime import RuntimeTests
    suite = unittest.TestSuite([RuntimeTests("test_baselines_use_distinct_linking_procedures")])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
