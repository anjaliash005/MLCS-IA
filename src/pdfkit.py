"""Small reportlab helpers shared by the technical report and the manuscript."""
from __future__ import annotations

import math
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, KeepTogether, Paragraph, Spacer, Table, TableStyle

INK = colors.HexColor("#0b0b0b")
INK2 = colors.HexColor("#52514e")
RULE = colors.HexColor("#c9c8c2")
HEAD_BG = colors.HexColor("#eef3fb")
ACCENT = colors.HexColor("#1c5cab")


def styles(base_size=10.0, leading=14.0):
    ss = getSampleStyleSheet()
    st = {
        "body": ParagraphStyle("body", parent=ss["Normal"], fontName="Times-Roman",
                               fontSize=base_size, leading=leading, alignment=TA_JUSTIFY,
                               spaceAfter=5, textColor=INK),
        "bodyL": ParagraphStyle("bodyL", parent=ss["Normal"], fontName="Times-Roman",
                                fontSize=base_size, leading=leading, alignment=TA_LEFT,
                                spaceAfter=4, textColor=INK),
        "bullet": ParagraphStyle("bullet", parent=ss["Normal"], fontName="Times-Roman",
                                 fontSize=base_size, leading=leading, leftIndent=14,
                                 bulletIndent=4, spaceAfter=2, alignment=TA_JUSTIFY, textColor=INK),
        "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=base_size + 4,
                             leading=base_size + 8, spaceBefore=12, spaceAfter=6, textColor=ACCENT),
        "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=base_size + 1.5,
                             leading=base_size + 5, spaceBefore=8, spaceAfter=4, textColor=INK),
        "h3": ParagraphStyle("h3", fontName="Helvetica-BoldOblique", fontSize=base_size,
                             leading=base_size + 3, spaceBefore=5, spaceAfter=3, textColor=INK),
        "cap": ParagraphStyle("cap", fontName="Times-Italic", fontSize=base_size - 1,
                              leading=base_size + 2, alignment=TA_CENTER, spaceAfter=8, textColor=INK2),
        "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=7.4, leading=9, textColor=INK),
        "cellB": ParagraphStyle("cellB", fontName="Helvetica-Bold", fontSize=7.4, leading=9, textColor=INK),
        "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=19, leading=24,
                                alignment=TA_CENTER, spaceAfter=10, textColor=INK),
        "subtitle": ParagraphStyle("subtitle", fontName="Helvetica", fontSize=11.5, leading=16,
                                   alignment=TA_CENTER, textColor=INK2, spaceAfter=4),
        "code": ParagraphStyle("code", fontName="Courier", fontSize=8, leading=10.2,
                               leftIndent=8, textColor=INK, spaceAfter=4,
                               backColor=colors.HexColor("#f4f4f2"), borderPadding=4),
        "ref": ParagraphStyle("ref", fontName="Times-Roman", fontSize=base_size - 1,
                              leading=base_size + 2, leftIndent=18, firstLineIndent=-18,
                              spaceAfter=2, alignment=TA_LEFT, textColor=INK),
    }
    return st


def md(text: str) -> str:
    """Tiny inline markup: **bold**, *italic*, `code`."""
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", text)
    return text


def P(text, style):
    return Paragraph(md(text), style)


def bullets(items, st):
    return [Paragraph(md(t), st["bullet"], bulletText="•") for t in items]


def table(rows, col_widths, st, header_rows=1, zebra=True, font_size=None):
    cs, cb = st["cell"], st["cellB"]
    if font_size:
        cs = ParagraphStyle("c2", parent=cs, fontSize=font_size, leading=font_size + 1.8)
        cb = ParagraphStyle("c3", parent=cb, fontSize=font_size, leading=font_size + 1.8)
    data = [[Paragraph(md(str(c)), cb if r < header_rows else cs) for c in row]
            for r, row in enumerate(rows)]
    t = Table(data, colWidths=col_widths, repeatRows=header_rows)
    style = [("GRID", (0, 0), (-1, -1), 0.4, RULE),
             ("BACKGROUND", (0, 0), (-1, header_rows - 1), HEAD_BG),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
             ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    if zebra:
        for r in range(header_rows, len(rows)):
            if (r - header_rows) % 2 == 1:
                style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f8f8f6")))
    t.setStyle(TableStyle(style))
    return t


def figure(path, width, caption, st):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(path).getSize()
    img = Image(path, width=width, height=width * ih / iw)
    return KeepTogether([img, Spacer(1, 3), Paragraph(md(caption), st["cap"])])


def f3(x, sd=None, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    s = f"{x:.{nd}f}"
    if sd is not None and not (isinstance(sd, float) and math.isnan(sd)):
        s += f" ± {sd:.{nd}f}"
    return s


def pfmt(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "–"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def strip_dates(path):
    """Remove creation/modification timestamps from a finished PDF's metadata."""
    from pypdf import PdfReader, PdfWriter
    reader = PdfReader(path)
    writer = PdfWriter(clone_from=reader)
    meta = {k: v for k, v in (reader.metadata or {}).items() if k not in ("/CreationDate", "/ModDate")}
    writer._info = None
    writer.add_metadata(meta)
    with open(path, "wb") as f:
        writer.write(f)


__all__ = ["styles", "P", "bullets", "table", "figure", "f3", "pfmt", "md", "cm", "Spacer", "strip_dates"]
