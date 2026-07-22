"""Static six-panel visual report for Member 1 comparisons."""

from __future__ import annotations

import html
from pathlib import Path

import cv2
import numpy as np


def _make_card(image: np.ndarray, label: str, width: int = 560, height: int = 420) -> np.ndarray:
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    available_width = width - 24
    available_height = height - 58
    scale = min(available_width / image.shape[1], available_height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    x = (width - resized.shape[1]) // 2
    y = 46 + (available_height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    cv2.putText(
        canvas,
        label,
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (35, 35, 35),
        2,
        cv2.LINE_AA,
    )
    return canvas


def build_comparison_grid(images: dict[str, np.ndarray], output_path: Path) -> Path:
    """Write a tiled BGR JPEG using the insertion order of ``images``."""

    if not images:
        raise ValueError("images cannot be empty")
    cards = [_make_card(image, label) for label, image in images.items()]
    while len(cards) % 3:
        cards.append(np.full_like(cards[0], 248))
    rows = [cv2.hconcat(cards[index : index + 3]) for index in range(0, len(cards), 3)]
    grid = cv2.vconcat(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise OSError(f"Could not write comparison grid: {output_path}")
    return output_path


def write_comparison_html(output_dir: Path, context: dict) -> Path:
    """Write a self-contained static HTML page with linked comparison panels."""

    output_dir.mkdir(parents=True, exist_ok=True)
    source = html.escape(str(context["source"]))
    parameters = html.escape(str(context.get("parameters", "")))
    panels = context.get("panels", [])
    panel_html = "\n".join(
        f"""<article class=\"panel\">
          <h2>{html.escape(str(panel['label']))}</h2>
          <a href=\"{html.escape(str(panel['src']))}\"><img src=\"{html.escape(str(panel['src']))}\" alt=\"{html.escape(str(panel['label']))}\"></a>
          <p>{html.escape(str(panel.get('description', '')))}</p>
        </article>"""
        for panel in panels
    )
    document = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Member 1 Gaussian Filtering + BBHE Comparison</title>
  <style>
    :root {{ color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif; }}
    body {{ margin: 0; padding: 32px; background: #f4f6f8; color: #17202a; }}
    main {{ max-width: 1500px; margin: 0 auto; }}
    header {{ margin-bottom: 28px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    .meta {{ color: #52606d; margin: 4px 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 18px; }}
    .panel {{ background: white; border-radius: 14px; padding: 14px; box-shadow: 0 5px 18px rgba(23, 32, 42, .08); }}
    .panel h2 {{ font-size: 18px; margin: 0 0 10px; }}
    .panel img {{ display: block; width: 100%; height: auto; border-radius: 8px; background: #eef1f3; }}
    .panel p {{ color: #52606d; line-height: 1.45; min-height: 42px; }}
    @media (max-width: 1000px) {{ .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
    @media (max-width: 650px) {{ body {{ padding: 18px; }} .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>Gaussian Filtering + BBHE Comparison</h1>
    <p class=\"meta\"><strong>Source:</strong> {source}</p>
    <p class=\"meta\"><strong>Parameters:</strong> {parameters}</p>
    <p class=\"meta\">Click any panel to open its full-resolution generated image.</p>
  </header>
  <section class=\"grid\">
    {panel_html}
  </section>
</main>
</body>
</html>
"""
    path = output_dir / "comparison.html"
    path.write_text(document, encoding="utf-8")
    return path
