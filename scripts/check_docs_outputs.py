"""Check known plot-bearing pages before publishing the documentation.

These notebooks exercise automatic inline display and explicit display(fig).
Count images inside notebook outputs, excluding logos and clipboard icons.
This is a rendering smoke check, not validation of every scientific example.
"""

import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

# Minimum figures expected from the current examples (including multi-panel plots).
EXPECTED_PLOTS = {
    "examples/general/getting_started.html": 5,
    "examples/simulations/tutorials/overview_of_two_phase_flow_part3.html": 8,
}


class NotebookImages(HTMLParser):
    def __init__(self):
        super().__init__()
        self.divs = []
        self.sources = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "div":
            self.divs.append("cell_output" in attrs.get("class", "").split())
        elif tag == "img" and any(self.divs):
            self.sources.append(attrs.get("src", ""))

    def handle_endtag(self, tag):
        if tag == "div" and self.divs:
            self.divs.pop()


def check_outputs(build_dir):
    errors = []
    for relative, expected in EXPECTED_PLOTS.items():
        page = build_dir / relative
        if not page.is_file():
            errors.append(f"Missing page: {relative}")
            continue
        parser = NotebookImages()
        parser.feed(page.read_text(encoding="utf-8"))
        if len(parser.sources) < expected:
            errors.append(
                f"{relative}: expected at least {expected} plots, found {len(parser.sources)}"
            )
        for source in parser.sources:
            url = urlsplit(source)
            if url.scheme or url.netloc or not url.path:
                errors.append(f"{relative}: expected a local plot asset, got {source!r}")
            elif not (page.parent / unquote(url.path)).is_file():
                errors.append(f"{relative}: missing plot asset {source}")
    return errors


if __name__ == "__main__":
    errors = check_outputs(Path(sys.argv[1]))
    if errors:
        sys.exit("Notebook plot check failed:\n" + "\n".join(errors))
    print("Notebook plot outputs and their image files are present.")
