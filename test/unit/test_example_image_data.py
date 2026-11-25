"""Exercise example data loading before and after the docs copy step."""

import json
from pathlib import Path
import shutil

import imageio.v2 as imageio
import numpy as np
import pytest
from skimage.io import imread_collection

REPO = Path(__file__).resolve().parents[2]
NOTEBOOKS = [
    "general/loading_images.ipynb",
    "general/denoising_and_segmenting_images.ipynb",
    "networks/tutorials/predicting_diffusive_size_factors_rock_sample.ipynb",
]


def tagged_cell(notebook, tag):
    cells = json.loads((REPO / "examples" / notebook).read_text(encoding="utf-8"))["cells"]
    return next(
        "".join(c["source"]) for c in cells if tag in c.get("metadata", {}).get("tags", [])
    )


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "source checkout with spaces"
    images = root / "docs/_static/images"
    images.mkdir(parents=True)
    source = REPO / "docs/_static/images"
    shutil.copy2(source / "Sample 1.tif", images)
    shutil.copytree(source / "Sample 1", images / "Sample 1")
    fixtures = root / "test/fixtures"
    fixtures.mkdir(parents=True)
    shutil.copy2(REPO / "test/fixtures/image_Leopard_slice100.npy", fixtures)
    return root


@pytest.mark.parametrize("notebook", NOTEBOOKS)
@pytest.mark.parametrize("location", ["examples", "docs/examples"])
def test_example_data_from_source_and_docs(checkout, monkeypatch, notebook, location):
    cwd = checkout / location / Path(notebook).parent
    cwd.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(cwd)
    namespace = {"np": np, "imread_collection": imread_collection}
    exec(tagged_cell(notebook, "image-data-setup"), namespace)
    if "image_dir" in namespace:
        stack = imageio.volread(namespace["image_dir"] / "Sample 1.tif")
        assert stack.shape == (10, 79, 114)
        assert stack.dtype == np.uint16
        if notebook == NOTEBOOKS[0]:
            exec(tagged_cell(notebook, "load-tiff-sequence"), namespace)
            assert len(namespace["seq"]) == 11
            assert namespace["seq"][0].shape == (89, 128)
    else:
        assert namespace["im"].shape == (100, 100, 100)
        assert np.array_equal(
            namespace["im"], np.load(REPO / "test/fixtures/image_Leopard_slice100.npy")
        )


@pytest.mark.parametrize("notebook", NOTEBOOKS)
def test_missing_checkout_has_actionable_error(tmp_path, monkeypatch, notebook):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="PoreSpy source checkout"):
        exec(tagged_cell(notebook, "image-data-setup"), {"np": np})


def test_empty_slice_folder_reports_missing_input(checkout, monkeypatch):
    monkeypatch.chdir(checkout)
    namespace = {"imread_collection": imread_collection}
    exec(tagged_cell(NOTEBOOKS[0], "image-data-setup"), namespace)
    for image in (namespace["image_dir"] / "Sample 1").glob("*.tif"):
        image.unlink()
    with pytest.raises(FileNotFoundError, match="No TIFF slices found"):
        exec(tagged_cell(NOTEBOOKS[0], "load-tiff-sequence"), namespace)
