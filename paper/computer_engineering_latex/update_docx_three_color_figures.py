"""Replace the three retained manuscript figures with submission color assets."""

from __future__ import annotations

import argparse
from pathlib import Path

from docx import Document
from docx.shared import Cm


ROOT = Path(__file__).resolve().parent


def image_part_for_shape(document: Document, shape_index: int):
    shape = document.inline_shapes[shape_index]
    blip = shape._inline.graphic.graphicData.pic.blipFill.blip
    return shape, document.part.related_parts[blip.embed]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    document = Document(args.source)
    if len(document.inline_shapes) != 3:
        raise RuntimeError(
            f"Expected exactly three retained figures, found {len(document.inline_shapes)}"
        )

    replacements = [
        ROOT / "figures" / "GS_OWL_framework_final_polished_600dpi.png",
        ROOT / "figures" / "layer_allocation_submission_color.png",
        ROOT / "figures" / "pllama_domain_mcq_submission_color.png",
    ]
    for index, replacement in enumerate(replacements):
        if not replacement.exists():
            raise FileNotFoundError(replacement)
        _, image_part = image_part_for_shape(document, index)
        image_part._blob = replacement.read_bytes()

    # Match the manuscript's previously verified full-width slot so the figure,
    # bilingual caption, and following subsection remain together on page 4.
    framework_shape = document.inline_shapes[0]
    framework_ratio = 170 / 88
    framework_shape.width = Cm(13.8)
    framework_shape.height = Cm(13.8 / framework_ratio)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    document.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
