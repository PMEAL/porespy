"""Regression coverage for the documentation publication smoke check."""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/check_docs_outputs.py"
PAGES = {
    "examples/general/getting_started.html": 5,
    "examples/simulations/tutorials/overview_of_two_phase_flow_part3.html": 8,
}


@pytest.mark.parametrize("failure", [None, "no_outputs", "partial", "asset", "page"])
def test_notebook_plot_publication_check(tmp_path, failure):
    for relative, count in PAGES.items():
        page = tmp_path / relative
        page.parent.mkdir(parents=True, exist_ok=True)
        (page.parent / "plot.png").write_bytes(b"plot asset")
        # Site chrome is not a substitute for notebook plots.
        html = '<img src="logo.png"><svg></svg>'
        if failure == "no_outputs":
            html += '<div class="cell_input"><img src="plot.png"></div>' * count
        else:
            for _ in range(count - (failure == "partial")):
                html += (
                    '<div class="cell_output docutils container"><div>'
                    '<img src="plot.png"></div></div>'
                )
        page.write_text(html, encoding="utf-8")
    if failure == "asset":
        (page.parent / "plot.png").unlink()
    if failure == "page":
        page.unlink()
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == (0 if failure is None else 1), result.stderr
