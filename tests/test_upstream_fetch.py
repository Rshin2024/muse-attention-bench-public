import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fetch_upstream", ROOT / "scripts/fetch_upstream.py")
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


class UpstreamFetchTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads((ROOT / "third_party/provenance.json").read_text())

    def test_default_is_offline_plan(self):
        with patch.object(fetch.urllib.request, "urlopen", side_effect=AssertionError("network")):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(fetch.main([]), 0)
        self.assertFalse(json.loads(output.getvalue())["network_called"])

    def test_rejects_path_traversal_and_other_origin(self):
        for key, value in (("path", "third_party/muse_attention_decode/../../escape"),
                           ("source_url", "https://example.invalid/source")):
            manifest = copy.deepcopy(self.manifest)
            manifest["files"][0][key] = value
            with self.assertRaises(ValueError):
                fetch.validate_manifest(manifest)

    def test_requires_complete_unique_pinned_roster(self):
        for entries in (self.manifest["files"][:-1], self.manifest["files"] + [self.manifest["files"][0]]):
            manifest = {**self.manifest, "files": entries}
            with self.assertRaises(ValueError):
                fetch.validate_manifest(manifest)

    def test_verifies_bytes_before_writing(self):
        entry = self.manifest["files"][0]
        response = io.BytesIO(b"wrong bytes")
        response.geturl = lambda: fetch.RAW_PREFIX + Path(entry["path"]).name
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(fetch.urllib.request, "urlopen", return_value=response):
                with self.assertRaises(ValueError):
                    fetch.download([entry], Path(directory))
            self.assertFalse((Path(directory) / entry["path"]).exists())

    def test_does_not_follow_local_vendor_symlink(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as other:
            root = Path(directory)
            (root / "third_party").symlink_to(Path(other), target_is_directory=True)
            with self.assertRaises(ValueError):
                fetch.download(self.manifest["files"], root)


if __name__ == "__main__":
    unittest.main()
