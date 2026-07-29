#!/usr/bin/env python3
"""Create report-ready HTML, CSV, and JSON artifacts from project results."""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hripcb_dashboard.reporting import build_report_pdf


def _export_pdf(records: list[dict], output_dir: Path) -> Path:
    path = output_dir / "report.pdf"
    payload = build_report_pdf(records, {"imgsz": 1024, "conf": 0.25, "iou": 0.70, "primary_metric": "map50_95"})
    path.write_bytes(payload)
    return path


def export_report(results_path: Path, output_dir: Path) -> Path:
    records = json.loads(Path(results_path).read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    keys = sorted({key for record in records for key in record.get("metrics", {})})
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "model", "module", "technique", "split", *keys])
        for record in records:
            writer.writerow([
                record["id"], record.get("model_id", "baseline"), record.get("module"), record.get("technique"), record.get("split"),
                *[record.get("metrics", {}).get(key, "") for key in keys],
            ])
    ranked = sorted(records, key=lambda record: float(record.get("metrics", {}).get("map50_95", float("-inf"))), reverse=True)
    rows = []
    for record in ranked:
        metrics = record.get("metrics", {})
        rows.append("<tr>" + "".join([
            f"<td>{html.escape(str(record['id']))}</td>",
            f"<td>{html.escape(str(record.get('module', '')))}</td>",
            f"<td>{html.escape(str(record.get('technique', '')))}</td>",
            f"<td>{float(metrics.get('map50_95', 0)):.4f}</td>",
            f"<td>{float(metrics.get('f1', 0)):.4f}</td>",
            f"<td>{float(metrics.get('precision', 0)):.4f}</td>",
            f"<td>{float(metrics.get('recall', 0)):.4f}</td>",
        ]) + "</tr>")
    best = ranked[0] if ranked else None
    document = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><title>HRIPCB Preprocessing Report</title><style>body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#172033;background:#f4f7fb;margin:0;padding:40px}}main{{max-width:1200px;margin:auto;background:white;padding:36px;border-radius:20px;box-shadow:0 15px 45px #28405c1a}}h1{{margin-top:0}}table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #dce5ef}}th{{background:#f4f7fb}}.best{{color:#198754;font-weight:700}}</style></head><body><main><h1>HRIPCB Preprocessing Comparison</h1><p>Generated from {len(records)} validation experiments. Primary metric: mAP50-95.</p><p class='best'>Best run: {html.escape(best['id']) if best else '—'} — {float(best['metrics'].get('map50_95', 0)):.4f}</p><table><thead><tr><th>ID</th><th>Module</th><th>Technique</th><th>mAP50-95</th><th>F1</th><th>Precision</th><th>Recall</th></tr></thead><tbody>{''.join(rows)}</tbody></table></main></body></html>"""
    report_path = output_dir / "report.html"
    report_path.write_text(document, encoding="utf-8")
    _export_pdf(records, output_dir)
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("runs/project_validation_comparison/results.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/project_report"))
    args = parser.parse_args()
    print(f"report: {export_report(args.results, args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
