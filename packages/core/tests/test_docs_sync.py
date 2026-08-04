import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestDocsSync(unittest.TestCase):
    def test_generated_docs_are_up_to_date(self) -> None:
        cmd = [sys.executable, "tools/generate_reference_docs.py", "--check"]
        completed = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"generate_reference_docs --check failed:\n{completed.stdout}\n{completed.stderr}",
        )

    def test_docs_sync_checks_pass(self) -> None:
        cmd = [sys.executable, "tools/check_docs_sync.py"]
        completed = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"check_docs_sync failed:\n{completed.stdout}\n{completed.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
