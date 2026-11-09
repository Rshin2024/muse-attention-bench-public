import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("release_guard", ROOT / "scripts/check_public_release.py")
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class PublicReleaseGuardTests(unittest.TestCase):
    def test_credentials_and_private_metadata_are_reported_without_values(self):
        fixtures = ["sk-" + "a" * 48, "/" + "Users" + "/example/documents/",
                    "person" + "@" + "work.invalid", "10" + ".1.2.3"]
        for body in fixtures:
            problems, _ = guard.inspect_payload("example.txt", body.encode())
            self.assertTrue(problems)
            self.assertNotIn(body, json.dumps(problems))

    def test_scientific_values_and_placeholder_addresses_are_allowed(self):
        self.assertEqual(guard.text_findings("speedup=1.270167; 46/16; contact@example.invalid"), [])

    def test_exact_private_dictionary(self):
        self.assertEqual(guard.text_findings("identifier example-value", ("example-value",)),
                         ["private_identifier_dictionary_match"])

    def test_nonempty_restricted_JSON_field(self):
        self.assertEqual(guard.json_findings({"project_id": "example"}, set()),
                         ["nonempty_restricted_JSON_field"])
        self.assertEqual(guard.json_findings({"project_id": None}, set()), [])

    def test_excluded_source_body_and_embedded_copy(self):
        body = b"excluded expressive source fixture"
        excluded = {hashlib.sha256(body).hexdigest()}
        problems, _ = guard.inspect_payload("source.py", body, excluded_digests=excluded)
        self.assertIn("excluded_source_body", [v["kind"] for v in problems])
        self.assertEqual(guard.json_findings({"source": body.decode()}, excluded),
                         ["excluded_source_embedded_as_string"])

    def test_archive_members_are_scanned(self):
        content = io.BytesIO()
        with zipfile.ZipFile(content, "w") as archive:
            archive.writestr("safe/data.txt", "sk-" + "b" * 48)
        problems, counts = guard.inspect_payload("source.zip", content.getvalue())
        self.assertEqual(counts["payloads"], 2)
        self.assertIn("credential_pattern", [v["kind"] for v in problems])

    def test_renamed_archive_cannot_skip_payload_audit(self):
        content = io.BytesIO()
        with zipfile.ZipFile(content, "w") as archive:
            archive.writestr("safe/data.txt", "sk-" + "c" * 48)
        problems, _ = guard.inspect_payload("renamed.txt", content.getvalue())
        kinds = [value["kind"] for value in problems]
        self.assertIn("ZIP_magic_suffix_mismatch", kinds)
        self.assertIn("credential_pattern", kinds)

    def test_rejects_path_traversal_and_local_upstream_material(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("../escape", "third_party/muse_attention_decode/TASK.md"):
                with self.assertRaises(ValueError):
                    guard.audit(Path(directory), [name])

    def test_rejects_symlinked_release_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "target.txt").write_text("benign")
            (root / "alias.txt").symlink_to(root / "target.txt")
            with self.assertRaises(ValueError):
                guard.audit(root, ["alias.txt"])


if __name__ == "__main__":
    unittest.main()
