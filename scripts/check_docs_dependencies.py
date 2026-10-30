"""Build the examples that exercise optional documentation dependencies.

Run with the docs and interactive dependency groups installed. Saved notebook
outputs are cleared, and execution errors fail the build. The smoke build uses
the project's Sphinx configuration and theme, without unrelated API generation.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from check_docs_outputs import NotebookImages

EXAMPLES = {
    "generators/reference/cylindrical_pillars_mesh": 4,
    "generators/reference/fractal_noise": 9,
    "filters/reference/nl_means_layered": 7,
    "metrics/reference/mesh_volume": 1,
    "metrics/reference/region_volumes": 3,
    "io/reference/to_stl": 2,
}


def build_examples(output):
    repo = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="porespy-docs-") as temporary:
        root = Path(temporary)
        docs = root / "docs"
        docs.mkdir()
        shutil.copytree(repo / "docs/_static", docs / "_static")
        config = (repo / "docs/conf.py").read_text(encoding="utf-8")
        config += "\n# This smoke build covers examples, not the API reference.\n"
        config += 'extensions.remove("autoapi.extension")\n'
        (docs / "conf.py").write_text(config, encoding="utf-8")
        for name in EXAMPLES:
            source = repo / "examples" / f"{name}.ipynb"
            notebook = json.loads(source.read_text(encoding="utf-8"))
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    cell["outputs"] = []
                    cell["execution_count"] = None
            target = root / "examples" / f"{name}.ipynb"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(notebook), encoding="utf-8")
        index = "Documentation dependency smoke tests\n====================================\n"
        index += "\n.. toctree::\n\n"
        index += "".join(f"   examples/{name}\n" for name in EXAMPLES)
        (docs / "index.rst").write_text(index, encoding="utf-8")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "sphinx",
                "-b",
                "html",
                "-W",
                "--keep-going",
                "-D",
                "nb_execution_mode=force",
                "-D",
                "nb_execution_raise_on_error=1",
                "-E",
                "-a",
                ".",
                str(output),
            ],
            cwd=docs,
            check=True,
        )
    for name, expected in EXAMPLES.items():
        page = output / "examples" / f"{name}.html"
        parser = NotebookImages()
        parser.feed(page.read_text(encoding="utf-8"))
        if len(parser.sources) < expected:
            raise RuntimeError(
                f"{name}: expected at least {expected} output images, "
                f"found {len(parser.sources)}"
            )
        for source in parser.sources:
            if not (page.parent / source).is_file():
                raise FileNotFoundError(f"{name}: missing output image {source}")
        print(f"{name}: {len(parser.sources)} output images verified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="HTML build directory")
    build_examples(parser.parse_args().output.resolve())
