import argparse
import csv
import shutil
import subprocess
import tempfile
from html import escape
from pathlib import Path


WIDTH = 1080
HEIGHT = 390

INK = "#25282B"
MUTED = "#6F7478"
GRID = "#D9DCDE"
BRICK = "#A65D4E"
STEEL = "#526F85"
SAGE = "#657A70"

MODEL_LABELS = {
    "llama1_7b": "LLaMA-1-7B",
    "llama2_7b": "LLaMA-2-7B",
    "llama1_13b": "LLaMA-1-13B",
    "llama1_30b": "LLaMA-1-30B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw the OSLA constraint sensitivity figure.")
    parser.add_argument("--input", type=Path, required=True, help="Path to osla_mechanism_ablation/summary_all.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("owl/analysis/paper_visualizations"),
        help="Directory for SVG, PDF, and PNG outputs",
    )
    parser.add_argument("--no-render", action="store_true", help="Only create SVG and HTML; skip Chrome PDF/PNG export")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"suite", "variant", "model_key", "ppl_test", "source_final_ppl"}
    if not rows:
        raise ValueError(f"No rows found in {path}")
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    for row in rows:
        row["relative_change"] = str(
            (float(row["ppl_test"]) / float(row["source_final_ppl"]) - 1.0) * 100.0
        )
    return rows


def find_row(rows: list[dict[str, str]], model_key: str, variant: str) -> dict[str, str]:
    matches = [row for row in rows if row["model_key"] == model_key and row["variant"] == variant]
    if len(matches) != 1:
        raise ValueError(f"Expected one row for {model_key}/{variant}, found {len(matches)}")
    return matches[0]


def line(x1: float, y1: float, x2: float, y2: float, **attrs: object) -> str:
    values = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, **attrs}
    return f"<line {' '.join(f'{key}=\"{value}\"' for key, value in values.items())}/>"


def text(x: float, y: float, value: str, **attrs: object) -> str:
    values = {"x": x, "y": y, **attrs}
    return f"<text {' '.join(f'{key}=\"{item}\"' for key, item in values.items())}>{escape(value)}</text>"


def rect(x: float, y: float, width: float, height: float, **attrs: object) -> str:
    values = {"x": x, "y": y, "width": width, "height": height, **attrs}
    return f"<rect {' '.join(f'{key}=\"{value}\"' for key, value in values.items())}/>"


def circle(cx: float, cy: float, radius: float, **attrs: object) -> str:
    values = {"cx": cx, "cy": cy, "r": radius, **attrs}
    return f"<circle {' '.join(f'{key}=\"{value}\"' for key, value in values.items())}/>"


def polygon(points: list[tuple[float, float]], **attrs: object) -> str:
    point_text = " ".join(f"{x},{y}" for x, y in points)
    values = {"points": point_text, **attrs}
    return f"<polygon {' '.join(f'{key}=\"{value}\"' for key, value in values.items())}/>"


def projection_panel(rows: list[dict[str, str]]) -> list[str]:
    items = []
    for model_key in ("llama1_7b", "llama2_7b"):
        row = find_row(rows, model_key, "projection_none")
        items.append((MODEL_LABELS[model_key], float(row["relative_change"])))

    elements = [
        text(28, 30, "(a)", class_="panel-tag"),
        text(64, 30, "预算投影的必要性", class_="panel-title"),
        text(398, 30, "70%稀疏率", class_="panel-note", text_anchor="end"),
    ]
    x0, x1 = 132, 398
    y0, y1 = 60, 302
    scale = lambda value: x0 + value / 5.0 * (x1 - x0)

    for tick in range(6):
        x = scale(float(tick))
        elements.append(line(x, y0, x, y1, stroke=GRID, stroke_width="1"))
        elements.append(text(x, y1 + 25, f"{tick}%", class_="tick", text_anchor="middle"))
    elements.append(line(x0, y1, x1, y1, stroke=INK, stroke_width="1.1"))

    for (label, value), y in zip(items, (132, 226)):
        endpoint = scale(value)
        elements.append(text(x0 - 14, y + 1, label, class_="ylabel", text_anchor="end"))
        elements.append(rect(x0, y - 20, endpoint - x0, 40, fill=BRICK))
        elements.append(circle(endpoint, y, 4.2, fill=INK))
        elements.append(text(endpoint + 12, y + 1, f"+{value:.2f}%", class_="value", text_anchor="start"))

    elements.append(text((x0 + x1) / 2, 366, "移除预算投影后的PPL相对增幅", class_="axis-title", text_anchor="middle"))
    return elements


def robustness_panel(rows: list[dict[str, str]]) -> list[str]:
    specs = [
        ("llama1_7b", "boundary_left2", "边界前移2层", "circle", STEEL),
        ("llama1_7b", "boundary_right2", "边界后移2层", "diamond", STEEL),
        ("llama2_7b", "boundary_left2", "边界前移2层", "circle", STEEL),
        ("llama2_7b", "boundary_right2", "边界后移2层", "diamond", STEEL),
        ("llama1_13b", "depth_normalized", "归一化深度", "square", SAGE),
        ("llama1_30b", "depth_normalized", "归一化深度", "square", SAGE),
    ]
    entries = []
    for model_key, variant, setting, marker, color in specs:
        row = find_row(rows, model_key, variant)
        entries.append(
            {
                "label": f"{MODEL_LABELS[model_key]}  {setting}",
                "value": float(row["relative_change"]),
                "marker": marker,
                "color": color,
            }
        )

    elements = [
        text(468, 30, "(b)", class_="panel-tag"),
        text(504, 30, "边界与深度定义的稳健性", class_="panel-title"),
    ]
    maximum = max(abs(entry["value"]) for entry in entries)
    elements.append(text(1052, 30, f"最大 |变化| = {maximum:.3f}%", class_="panel-note", text_anchor="end"))

    x0, x1 = 752, 1048
    y0, y1 = 58, 306
    minimum, maximum_x = -0.7, 0.3
    scale = lambda value: x0 + (value - minimum) / (maximum_x - minimum) * (x1 - x0)
    zero_x = scale(0.0)

    for tick in (-0.6, -0.4, -0.2, 0.0, 0.2):
        x = scale(tick)
        elements.append(line(x, y0, x, y1, stroke=GRID, stroke_width="1"))
        elements.append(text(x, y1 + 25, f"{tick:+.1f}%", class_="tick", text_anchor="middle"))
    elements.append(line(zero_x, y0, zero_x, y1, stroke=INK, stroke_width="1.35"))
    elements.append(line(x0, y1, x1, y1, stroke=INK, stroke_width="1.1"))

    y_positions = (78, 120, 162, 204, 258, 300)
    for entry, y in zip(entries, y_positions):
        value = entry["value"]
        endpoint = scale(value)
        elements.append(text(x0 - 16, y + 1, entry["label"], class_="ylabel", text_anchor="end"))
        elements.append(line(min(endpoint, zero_x), y, max(endpoint, zero_x), y, stroke=entry["color"], stroke_width="2.4"))

        if entry["marker"] == "circle":
            elements.append(circle(endpoint, y, 5.2, fill=entry["color"]))
        elif entry["marker"] == "diamond":
            elements.append(
                polygon(
                    [(endpoint, y - 6), (endpoint + 6, y), (endpoint, y + 6), (endpoint - 6, y)],
                    fill="white",
                    stroke=entry["color"],
                    stroke_width="2",
                )
            )
        else:
            elements.append(rect(endpoint - 5.2, y - 5.2, 10.4, 10.4, fill=entry["color"]))

        if value < -0.50:
            label_x, anchor = endpoint + 11, "start"
        elif value < 0:
            label_x, anchor = endpoint - 11, "end"
        else:
            label_x, anchor = endpoint + 11, "start"
        elements.append(text(label_x, y + 1, f"{value:+.3f}%", class_="value-small", text_anchor=anchor))

    elements.append(line(480, 229, 1048, 229, stroke=GRID, stroke_width="1"))
    elements.append(text((x0 + x1) / 2, 366, "相对默认配置的PPL变化", class_="axis-title", text_anchor="middle"))
    return elements


def make_svg(rows: list[dict[str, str]]) -> str:
    elements = projection_panel(rows) + robustness_panel(rows)
    body = "\n  ".join(elements).replace("class_=", "class=").replace("text_anchor=", "text-anchor=").replace("stroke_width=", "stroke-width=")
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">
  <title id="title">OSLA constraint sensitivity</title>
  <desc id="desc">Removing budget projection increases perplexity, while boundary shifts and normalized depth cause only small relative changes.</desc>
  <style>
    text {{ font-family: "Microsoft YaHei", "Noto Sans CJK SC", Arial, sans-serif; fill: {INK}; }}
    .panel-tag {{ font-size: 18px; font-weight: 700; dominant-baseline: middle; }}
    .panel-title {{ font-size: 18px; font-weight: 600; dominant-baseline: middle; }}
    .panel-note {{ font-size: 12px; fill: {MUTED}; dominant-baseline: middle; }}
    .ylabel {{ font-size: 13px; dominant-baseline: middle; }}
    .tick {{ font-size: 12px; fill: {MUTED}; dominant-baseline: middle; }}
    .value {{ font-size: 14px; font-weight: 600; dominant-baseline: middle; }}
    .value-small {{ font-size: 12.5px; font-weight: 600; dominant-baseline: middle; }}
    .axis-title {{ font-size: 13px; dominant-baseline: middle; }}
  </style>
  <rect width="{WIDTH}" height="{HEIGHT}" fill="white"/>
  {body}
</svg>
"""


def make_html(svg: str) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<style>
  @page {{ size: 7.5in 2.7083in; margin: 0; }}
  html, body {{ margin: 0; width: {WIDTH}px; height: {HEIGHT}px; overflow: hidden; background: white; }}
  svg {{ display: block; width: {WIDTH}px; height: {HEIGHT}px; }}
  * {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
</style>
</head>
<body>{svg}</body>
</html>
"""


def find_chrome() -> Path | None:
    candidates = [
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    executable = shutil.which("chrome") or shutil.which("msedge")
    return Path(executable) if executable else None


def render_with_chrome(chrome: Path, html_path: Path, output_dir: Path) -> None:
    pdf_path = output_dir / "osla_constraint_sensitivity.pdf"
    png_path = output_dir / "osla_constraint_sensitivity.png"
    with tempfile.TemporaryDirectory(prefix="osla-plot-") as profile:
        common = [
            str(chrome),
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--user-data-dir={profile}",
        ]
        subprocess.run(
            common
            + [
                "--no-pdf-header-footer",
                f"--print-to-pdf={pdf_path}",
                html_path.resolve().as_uri(),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        subprocess.run(
            common
            + [
                "--force-device-scale-factor=2",
                f"--window-size={WIDTH},{HEIGHT}",
                f"--screenshot={png_path}",
                html_path.resolve().as_uri(),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    svg = make_svg(rows)
    svg_path = output_dir / "osla_constraint_sensitivity.svg"
    html_path = output_dir / "osla_constraint_sensitivity.html"
    svg_path.write_text(svg, encoding="utf-8")
    html_path.write_text(make_html(svg), encoding="utf-8")

    if not args.no_render:
        chrome = find_chrome()
        if chrome is None:
            raise RuntimeError("Chrome or Edge is required for PDF/PNG export; rerun with --no-render for SVG only.")
        render_with_chrome(chrome, html_path, output_dir)


if __name__ == "__main__":
    main()
