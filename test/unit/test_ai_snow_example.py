"""Run the AI tutorial's image and extraction cells without its optional model."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

import porespy as ps


def test_ai_tutorial_network_extraction():
    notebook = (
        Path(__file__).resolve().parents[2]
        / "examples/networks/tutorials/using_diffusive_size_factor_AI_with_snow.ipynb"
    )
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    namespace = {"np": np, "sp": sp, "ps": ps, "plt": plt}
    state = np.random.get_state()
    try:
        for cell in cells:
            if "snow2-example-regression" in cell.get("metadata", {}).get("tags", []):
                exec("".join(cell["source"]), namespace)
        regions = namespace["regions"]
        network = namespace["net"]
        assert regions.shape == namespace["im"].shape == (120, 120, 120)
        assert regions.max() == len(network["pore.coords"])
        conns = network["throat.conns"]
        assert len(conns) > 0
        assert conns.min() >= 0
        assert conns.max() < len(network["pore.coords"])
    finally:
        np.random.set_state(state)
        if "fig" in namespace:
            plt.close(namespace["fig"])
