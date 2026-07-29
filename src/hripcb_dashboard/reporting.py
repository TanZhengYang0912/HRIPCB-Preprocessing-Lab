"""Reusable report payload and PDF generation for the dashboard and CLI."""

from __future__ import annotations

import io
import json
import html
import math
from collections.abc import Iterable

from .filtering import collapse_shared_baseline, is_combined_record


def _metric(record: dict, key: str) -> float:
    try:
        return float((record.get("metrics") or {}).get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def record_metric_summary(records: Iterable[dict], metric: str = "map50_95") -> dict:
    """Return report-friendly coverage, winner, and rankings for records."""

    source = [dict(record) for record in records]
    display_source = collapse_shared_baseline(source)
    combined = [record for record in source if is_combined_record(record)]
    ranking_source = combined or source
    ranked = sorted(ranking_source, key=lambda record: (_metric(record, metric), str(record.get("id", ""))), reverse=True)
    all_ranked = sorted(display_source, key=lambda record: (_metric(record, metric), str(record.get("id", ""))), reverse=True)
    module_values = {str(record.get("module", "unknown")) for record in source}
    reference = [record for record in display_source if not is_combined_record(record)]
    baseline_control_count = sum(
        1 for record in display_source if str(record.get("module", "unknown")) == "baseline"
    )
    return {
        "count": len(source),
        "display_count": len(display_source),
        "reference_count": len(reference),
        "module_count": len(module_values - {"baseline"}),
        "baseline_control_count": baseline_control_count,
        "model_count": len({str(record.get("model_id", "baseline")) for record in source}),
        "combined_count": len(combined),
        "combined_module_count": len({str(record.get("module", "unknown")) for record in combined if str(record.get("module", "unknown")) != "baseline"}),
        "metric": metric,
        "best": ranked[0] if ranked else None,
        "ranked": ranked,
        "all_ranked": all_ranked,
    }


def format_parameters(parameters: dict) -> str:
    """Format parameters as one readable key/value pair per line."""

    if not parameters:
        return "No custom parameters"
    return "\n".join(f"{key}: {value}" for key, value in sorted(parameters.items()))


def _json_safe(value):
    """Convert non-finite numbers to JSON-compatible null values recursively."""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def dumps_json(value, *, sort_keys: bool = False) -> str:
    """Serialize dashboard data as strict JSON, replacing Infinity/NaN with null."""

    return json.dumps(_json_safe(value), indent=2, sort_keys=sort_keys, allow_nan=False)


def report_chart_payload(records: Iterable[dict]) -> dict:
    """Return the same comparison datasets shown in the Analysis dashboard."""

    from .analysis import build_analysis_payload

    return build_analysis_payload(records)


def _bar_chart_drawing(title: str, categories: list[str], series: dict[str, list[float]], *, width: float = 350, height: float = 190):
    """Build a compact vector bar chart that remains sharp in the PDF."""

    from reportlab.graphics.shapes import Drawing, Line, Rect, String
    from reportlab.lib import colors

    drawing = Drawing(width, height)
    left, right, top, bottom = 42, 12, 30, 42
    plot_width = width - left - right
    plot_height = height - top - bottom
    values = [value for values in series.values() for value in values if value is not None]
    maximum = max(values, default=1.0)
    tick_max = max(0.1, math.ceil(maximum * 10) / 10)
    colors_for_series = [
        colors.HexColor("#2563eb"),
        colors.HexColor("#14b8a6"),
        colors.HexColor("#f59e0b"),
        colors.HexColor("#ef4444"),
        colors.HexColor("#7c3aed"),
    ]

    drawing.add(String(width / 2, height - 12, title, textAnchor="middle", fontName="Helvetica-Bold", fontSize=9, fillColor=colors.HexColor("#172033")))
    for tick_index in range(0, 5):
        tick = tick_max * tick_index / 4
        y = bottom + plot_height * tick / tick_max
        drawing.add(Line(left, y, width - right, y, strokeColor=colors.HexColor("#dce5ef"), strokeWidth=0.4))
        drawing.add(String(left - 5, y - 2, f"{tick:.2f}", textAnchor="end", fontSize=6, fillColor=colors.HexColor("#697586")))

    series_names = list(series)
    group_width = plot_width / max(1, len(categories))
    bar_width = min(24, group_width / max(1, len(series_names)) * 0.72)
    total_bar_width = bar_width * len(series_names)
    for category_index, category in enumerate(categories):
        group_x = left + group_width * category_index
        start_x = group_x + (group_width - total_bar_width) / 2
        drawing.add(String(group_x + group_width / 2, bottom - 13, category, textAnchor="middle", fontSize=6, fillColor=colors.HexColor("#465468")))
        for series_index, name in enumerate(series_names):
            value = series[name][category_index] if category_index < len(series[name]) else None
            value = 0.0 if value is None else float(value)
            bar_height = plot_height * value / tick_max
            drawing.add(Rect(
                start_x + series_index * bar_width,
                bottom,
                max(1, bar_width - 1.2),
                bar_height,
                fillColor=colors_for_series[series_index % len(colors_for_series)],
                strokeColor=None,
            ))

    legend_x = left
    legend_y = height - 26
    for series_index, name in enumerate(series_names):
        item_width = max(45, len(name) * 4.5 + 16)
        drawing.add(Rect(legend_x, legend_y, 6, 6, fillColor=colors_for_series[series_index % len(colors_for_series)], strokeColor=None))
        drawing.add(String(legend_x + 9, legend_y - 1, name, fontSize=6, fillColor=colors.HexColor("#465468")))
        legend_x += item_width
    return drawing


def build_report_pdf(records: list[dict], protocol: dict) -> bytes:
    """Build a self-contained landscape PDF for report submission."""

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from .analysis import technique_label

    summary = record_metric_summary(records)
    charts = report_chart_payload(records)
    best = summary["best"]
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("ReportTitle", parent=styles["Title"], fontSize=20, leading=24, alignment=1, spaceAfter=4 * mm)
    body_style = ParagraphStyle("ReportBody", parent=styles["BodyText"], fontSize=8.5, leading=11, spaceAfter=1 * mm)
    cell_style = ParagraphStyle("ReportCell", parent=styles["BodyText"], fontSize=7.5, leading=9)
    small_style = ParagraphStyle("ReportSmall", parent=cell_style, fontSize=7, leading=8.5)
    header_style = ParagraphStyle("ReportHeader", parent=cell_style, textColor=colors.white, fontName="Helvetica-Bold")

    def paragraph(value, style=cell_style, *, line_breaks: bool = False):
        text = html.escape(str(value))
        if line_breaks:
            text = text.replace("\n", "<br/>")
        return Paragraph(text, style)

    page_width, _ = landscape(A4)

    def draw_footer(canvas, document):
        canvas.saveState()
        canvas.setFillColor(colors.HexColor("#697586"))
        canvas.setFont("Helvetica", 7)
        canvas.drawRightString(page_width - 12 * mm, 7 * mm, f"HRIPCB Preprocessing Report · Page {document.page}")
        canvas.restoreState()

    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=12 * mm,
        leftMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
    )
    story = [
        Paragraph("HRIPCB Preprocessing Comparison", title_style),
        paragraph(
            f"{summary['display_count']} displayed runs ({summary['count']} raw records) across "
            f"{summary['module_count']} member modules plus {summary['baseline_control_count']} baseline controls "
            f"and {summary['model_count']} models. "
            "Primary metric: mAP50-95.",
            body_style,
        ),
        Spacer(1, 3 * mm),
        paragraph(
            "Best run: " + (
                f"{best.get('id', '—')} · {best.get('module', '—')} / {technique_label(best.get('technique'))} · "
                f"mAP50-95={_metric(best, 'map50_95'):.4f}"
                if best else "—"
            ),
            body_style,
        ),
        Spacer(1, 4 * mm),
    ]
    protocol_data = [[paragraph("Frozen protocol", header_style), paragraph("Value", header_style)]]
    protocol_data.extend([[paragraph(key, body_style), paragraph(value, body_style)] for key, value in sorted(protocol.items())])
    protocol_table = Table(protocol_data, colWidths=[38 * mm, 70 * mm], hAlign="LEFT")
    protocol_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#172033")),
        ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#dce5ef")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#dce5ef")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7fb")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([protocol_table, Spacer(1, 6 * mm)])

    primary = charts["original_vs_combined"]
    primary_categories = [
        "Original" if row["label"] == "Original" else row["label"].replace("member", "M").replace(" / ", " ")
        for row in primary
    ]
    visual_story = [
        PageBreak(),
        Paragraph("Visual Comparisons", styles["Heading2"]),
        paragraph("The same validation comparisons used by the Analysis dashboard are included here for report and presentation use.", body_style),
        _bar_chart_drawing(
            "Original vs combined winners (mAP50-95)",
            primary_categories,
            {"mAP50-95": [row["map50_95"] for row in primary]},
            width=720,
            height=205,
        ),
        Spacer(1, 4 * mm),
    ]
    metric_rows = charts["metric_comparison"]
    metric_categories = [row["label"].replace("member", "M").replace(" / ", " ") for row in metric_rows]
    metric_series = {
        "Precision": [row["precision"] for row in metric_rows],
        "Recall": [row["recall"] for row in metric_rows],
        "mAP50": [row["map50"] for row in metric_rows],
        "mAP50-95": [row["map50_95"] for row in metric_rows],
        "F1": [row["f1"] for row in metric_rows],
    }
    stage_rows = charts["stage_comparison"]
    stage_series = {
        key: [row.get(key) for row in stage_rows]
        for key in ("Original", "Noise-only", "Contrast-only", "Combined")
    }
    visual_story.append(Table([[
        _bar_chart_drawing("Combined winner metrics", metric_categories, metric_series, width=350, height=205),
        _bar_chart_drawing("Processing stage comparison", [row["member"] for row in stage_rows], stage_series, width=350, height=205),
    ]], colWidths=[360, 360], hAlign="LEFT"))
    if charts["retrained_vs_baseline"]:
        retrained_rows = charts["retrained_vs_baseline"]
        visual_story.extend([
            Spacer(1, 4 * mm),
            _bar_chart_drawing(
                "Official test: baseline vs retrained candidate",
                [row["Metric"] for row in retrained_rows],
                {
                    "Baseline": [row["Baseline"] for row in retrained_rows],
                    "Retrained candidate": [row["Retrained candidate"] for row in retrained_rows],
                },
                width=720,
                height=205,
            ),
        ])
    story.extend(visual_story)
    story.extend([PageBreak(), Paragraph("Leaderboard", styles["Heading2"])])

    data = [[paragraph("ID", header_style), paragraph("Model", header_style), paragraph("Module", header_style), paragraph("Technique", header_style), paragraph("Split", header_style), paragraph("mAP50-95", header_style), paragraph("F1", header_style), paragraph("Precision", header_style), paragraph("Recall", header_style)]]
    for record in summary["ranked"]:
        data.append([
            paragraph(record.get("id", "")),
            paragraph(record.get("model_id", "baseline")),
            paragraph(record.get("module", "")),
            paragraph(record.get("technique", "")),
            paragraph(record.get("split", "")),
            paragraph(f"{_metric(record, 'map50_95'):.4f}"),
            paragraph(f"{_metric(record, 'f1'):.4f}"),
            paragraph(f"{_metric(record, 'precision'):.4f}"),
            paragraph(f"{_metric(record, 'recall'):.4f}"),
        ])
    table = Table(
        data,
        repeatRows=1,
        colWidths=[43 * mm, 20 * mm, 22 * mm, 31 * mm, 16 * mm, 22 * mm, 18 * mm, 23 * mm, 21 * mm],
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#172033")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#dce5ef")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7fb")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(table)
    story.extend([PageBreak(), Paragraph("Parameter Settings", styles["Heading2"]), paragraph("Each preprocessing setting is wrapped into readable key/value lines so long JSON strings never overflow the page.", body_style), Spacer(1, 2 * mm)])
    parameter_data = [[paragraph("ID", header_style), paragraph("Module / technique", header_style), paragraph("Parameters", header_style)]]
    for record in summary["ranked"]:
        parameter_data.append([
            paragraph(record.get("id", ""), small_style),
            paragraph(f"{record.get('module', '—')} / {record.get('technique', '—')}", small_style),
            paragraph(format_parameters(record.get("parameters", {})), small_style, line_breaks=True),
        ])
    parameter_table = Table(parameter_data, repeatRows=1, colWidths=[51 * mm, 47 * mm, 131 * mm])
    parameter_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#172033")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#dce5ef")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7fb")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(parameter_table)
    document.build(story, onFirstPage=draw_footer, onLaterPages=draw_footer)
    return buffer.getvalue()
