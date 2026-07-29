"""Reusable report payload and PDF generation for the dashboard and CLI."""

from __future__ import annotations

import io
import json
import html
import math
from collections.abc import Iterable


def _metric(record: dict, key: str) -> float:
    try:
        return float((record.get("metrics") or {}).get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def record_metric_summary(records: Iterable[dict], metric: str = "map50_95") -> dict:
    """Return report-friendly coverage, winner, and rankings for records."""

    source = [dict(record) for record in records]
    ranked = sorted(source, key=lambda record: (_metric(record, metric), str(record.get("id", ""))), reverse=True)
    module_values = {str(record.get("module", "unknown")) for record in source}
    baseline_control_count = sum(
        1 for record in source if str(record.get("module", "unknown")) == "baseline"
    )
    return {
        "count": len(source),
        "module_count": len(module_values - {"baseline"}),
        "baseline_control_count": baseline_control_count,
        "model_count": len({str(record.get("model_id", "baseline")) for record in source}),
        "metric": metric,
        "best": ranked[0] if ranked else None,
        "ranked": ranked,
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


def build_report_pdf(records: list[dict], protocol: dict) -> bytes:
    """Build a self-contained landscape PDF for report submission."""

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    summary = record_metric_summary(records)
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
            f"{summary['count']} runs across {summary['module_count']} member modules plus "
            f"{summary['baseline_control_count']} baseline control and {summary['model_count']} models. "
            "Primary metric: mAP50-95.",
            body_style,
        ),
        Spacer(1, 3 * mm),
        paragraph(
            "Best run: " + (
                f"{best.get('id', '—')} · {best.get('module', '—')} / {best.get('technique', '—')} · "
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
    story.extend([protocol_table, Spacer(1, 6 * mm), paragraph("Leaderboard", styles["Heading2"])])

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
