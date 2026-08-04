import importlib
import importlib.util
import inspect
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestDocsIntegrity(unittest.TestCase):
    def test_key_modules_have_module_docstrings(self) -> None:
        module_names = [
            "concreteness_knn_core.config",
            "concreteness_knn_core.data",
            "concreteness_knn_core.knn",
            "concreteness_knn_core.prediction",
            "concreteness_knn_core.cli",
        ]
        for name in module_names:
            mod = importlib.import_module(name)
            self.assertTrue(mod.__doc__ and mod.__doc__.strip(), msg=f"Missing module docstring: {name}")

    def test_public_callables_have_docstrings(self) -> None:
        module_names = [
            "concreteness_knn_core.config",
            "concreteness_knn_core.data",
            "concreteness_knn_core.knn",
            "concreteness_knn_core.prediction",
            "concreteness_knn_core.cli",
        ]
        for name in module_names:
            mod = importlib.import_module(name)
            for symbol, obj in inspect.getmembers(mod):
                if symbol.startswith("_"):
                    continue
                if inspect.isfunction(obj) or inspect.isclass(obj):
                    if getattr(obj, "__module__", None) != name:
                        continue
                    self.assertTrue(
                        obj.__doc__ and obj.__doc__.strip(),
                        msg=f"Missing docstring for {name}.{symbol}",
                    )

    def test_required_docs_exist_and_nonempty(self) -> None:
        required = [
            "docs/overview.md",
            "docs/cli_reference.md",
            "docs/config_reference.md",
            "docs/prediction_pipeline.md",
            "docs/output_catalog.md",
            "docs/extending.md",
            "docs/glossary.md",
            "docs/maintenance.md",
            "docs/generated_config_reference.md",
            "docs/generated_output_manifest.md",
        ]
        for rel in required:
            path = REPO_ROOT / rel
            self.assertTrue(path.exists(), msg=f"Missing docs file: {rel}")
            self.assertTrue(path.read_text(encoding="utf-8").strip(), msg=f"Empty docs file: {rel}")

    def test_markdown_internal_links_resolve(self) -> None:
        script_path = REPO_ROOT / "tools" / "check_docs_sync.py"
        spec = importlib.util.spec_from_file_location("check_docs_sync", script_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        markdown_files = [REPO_ROOT / "README.md"] + sorted((REPO_ROOT / "docs").glob("*.md"))
        errors = module.check_markdown_links(markdown_files)
        self.assertFalse(errors, msg="\n".join(errors))


if __name__ == "__main__":
    unittest.main()
