import tempfile
import time
import unittest
from pathlib import Path

from report_cache import ReportCache


class ReportCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = ReportCache(directory=str(Path(self.tmp.name) / "reports"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_set_get_roundtrip(self):
        self.cache.set("m1", {"probable_xi": ["A"], "saved_at_probe": 1})
        payload = self.cache.get("m1")
        self.assertEqual(payload["probable_xi"], ["A"])

    def test_persists_to_disk_and_reloads(self):
        self.cache.set("m1", {"value": 42})
        second = ReportCache(directory=str(Path(self.tmp.name) / "reports"))
        self.assertEqual(second.get("m1")["value"], 42)

    def test_expired_entry_is_ignored(self):
        self.cache.TTL = -1  # force expiry
        self.cache.set("m1", {"value": 1})
        self.assertIsNone(self.cache.get("m1"))

    def test_missing_key_returns_none(self):
        self.assertIsNone(self.cache.get("does-not-exist"))

    def test_overwrite_updates_value(self):
        self.cache.set("m1", {"value": 1})
        self.cache.set("m1", {"value": 2})
        self.assertEqual(self.cache.get("m1")["value"], 2)


if __name__ == "__main__":
    unittest.main()