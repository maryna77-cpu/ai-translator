"""In-place translation of a layout-faithful Word document.

The Word skeleton produced by ``pdf2docx`` keeps the source layout: one paragraph per
text block, real tables, inline figure images and per page sections.  This module

* walks the document (body paragraphs *and* table cells, in reading order) and
  collects the translatable units,
* writes the Persian translations back into the very same paragraphs, keeping the
  run formatting of the source (size, weight, colour) while switching the paragraph
  to right-to-left, mirroring alignment and swapping the embedded subset fonts for
  fonts that actually contain Persian glyphs,
* relaxes the "exact" line/row heights coming from the PDF so longer Persian text is
  never clipped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

from docx.document import Document as _Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph

PAGE_NUMBER_RE = re.compile(
    r"^(?:(?:page|p\.?|صفحه)\s*)?[-–—]?\s*\d{1,4}\s*(?:[-–—]|/\s*\d{1,4})?$", re.I
)
HEADING_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)*)\.?\s+\S")
CAPTION_RE = re.compile(r"^\s*(?:figure|table|fig\.|جدول|شکل)\s*\d+", re.I)

ALIGN_RTL = {"left": "right", "right": "left", "center": "center", "both": "both",
             "distribute": "distribute"}


@dataclass
class Unit:
    id: str
    kind: str
    text: str
    location: str
    paragraph: Paragraph


def _paragraph_text(paragraph: Paragraph) -> str:
    return "".join(node.text or "" for node in paragraph._p.iter(qn("w:t")))


def _has_drawing(paragraph: Paragraph) -> bool:
    p = paragraph._p
    return bool(p.findall(".//" + qn("w:drawing")) or p.findall(".//" + qn("w:pict")))


def _classify(text: str, in_table: bool) -> str:
    if in_table:
        return "cell"
    if CAPTION_RE.match(text):
        return "caption"
    if HEADING_RE.match(text) and len(text) <= 120:
        return "heading"
    if text.lstrip().lower().startswith(("note:", "note :")):
        return "note"
    return "paragraph"


def _iter_container(container, path: str, units: list[Unit], in_table: bool) -> None:
    """Walk paragraphs/tables of the body or of a table cell, in document order."""
    for child in container.iterchildren():
        tag = child.tag.split("}")[1]
        if tag == "p":
            paragraph = Paragraph(child, container)
            text = re.sub(r"\s+", " ", _paragraph_text(paragraph)).strip()
            # Some PDF text layers map the apostrophe glyph to U+00B6; normalise it.
            text = text.replace("¶", "'")
            if not text or _has_drawing(paragraph) or PAGE_NUMBER_RE.fullmatch(text):
                continue
            units.append(Unit(
                id=f"u{len(units) + 1}",
                kind=_classify(text, in_table),
                text=text,
                location=path,
                paragraph=paragraph,
            ))
        elif tag == "tbl":
            table = Table(child, container)
            for r, row in enumerate(table.rows, start=1):
                for c, cell in enumerate(row.cells, start=1):
                    _iter_container(cell._tc, f"{path}/t{r},{c}", units, True)


def collect_units(document: _Document) -> list[Unit]:
    """All translatable paragraphs of the document body, in reading order."""
    units: list[Unit] = []
    _iter_container(document.element.body, "body", units, False)
    return units


# ---------------------------------------------------------------------------
# write-back
# ---------------------------------------------------------------------------

def _set_rtl_paragraph(paragraph: Paragraph) -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    if p_pr.find(qn("w:bidi")) is None:
        bidi = OxmlElement("w:bidi")
        bidi.set(qn("w:val"), "1")
        p_pr.append(bidi)
    jc = p_pr.find(qn("w:jc"))
    if jc is not None:
        jc.set(qn("w:val"), ALIGN_RTL.get(jc.get(qn("w:val")), "right"))
    elif paragraph.alignment is None:
        new_jc = OxmlElement("w:jc")
        new_jc.set(qn("w:val"), "right")
        p_pr.append(new_jc)


def _relax_spacing(paragraph: Paragraph) -> None:
    p_pr = paragraph._p.find(qn("w:pPr"))
    if p_pr is None:
        return
    spacing = p_pr.find(qn("w:spacing"))
    if spacing is None:
        return
    if spacing.get(qn("w:lineRule")) == "exact":
        spacing.set(qn("w:lineRule"), "atLeast")
        try:
            line = int(float(spacing.get(qn("w:line")) or 0))
            if line:
                spacing.set(qn("w:line"), str(int(line * 1.18)))
        except ValueError:
            pass


def _style_run(run_element, latin_font: str, persian_font: str, rtl: bool) -> None:
    r_pr = run_element.find(qn("w:rPr"))
    if r_pr is None:
        r_pr = OxmlElement("w:rPr")
        run_element.insert(0, r_pr)
    fonts = r_pr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        r_pr.insert(0, fonts)
    fonts.set(qn("w:ascii"), latin_font)
    fonts.set(qn("w:hAnsi"), latin_font)
    fonts.set(qn("w:cs"), persian_font)
    if fonts.get(qn("w:eastAsia"), "").startswith("CIDFont"):
        fonts.set(qn("w:eastAsia"), latin_font)
    # PDF char-width scaling distorts reshaped Persian glyphs.
    for tag in ("w:w",):
        node = r_pr.find(qn(tag))
        if node is not None:
            r_pr.remove(node)
    if rtl:
        if r_pr.find(qn("w:rtl")) is None:
            rtl_el = OxmlElement("w:rtl")
            rtl_el.set(qn("w:val"), "1")
            r_pr.append(rtl_el)
        sz = r_pr.find(qn("w:sz"))
        if sz is not None and r_pr.find(qn("w:szCs")) is None:
            sz_cs = OxmlElement("w:szCs")
            sz_cs.set(qn("w:val"), sz.get(qn("w:val")))
            r_pr.append(sz_cs)


def apply_translation(paragraph: Paragraph, translation: str, latin_font: str, persian_font: str) -> None:
    """Replace the paragraph text with ``translation``, keeping layout and images."""
    runs = paragraph._p.findall(qn("w:r"))
    text_runs = [r for r in runs if r.find(qn("w:t")) is not None]
    if not text_runs:
        return
    first = text_runs[0]
    for run in text_runs[1:]:
        paragraph._p.remove(run)
    t_node = first.find(qn("w:t"))
    t_node.text = translation
    t_node.set(qn("xml:space"), "preserve")
    _style_run(first, latin_font, persian_font, rtl=True)
    _set_rtl_paragraph(paragraph)
    _relax_spacing(paragraph)


def restyle_residual_runs(document: _Document, latin_font: str, persian_font: str) -> None:
    """Swap embedded subset font names everywhere (also in untranslated runs)."""
    for run in document.element.body.iter(qn("w:r")):
        has_text = run.find(qn("w:t")) is not None
        _style_run(run, latin_font, persian_font, rtl=False)
        if not has_text:
            r_pr = run.find(qn("w:rPr"))
            if r_pr is not None:
                node = r_pr.find(qn("w:rtl"))
                if node is not None:
                    r_pr.remove(node)


def relax_table_rows(document: _Document) -> None:
    """Exact row heights from the PDF would clip longer Persian text."""
    for tr_height in document.element.body.iter(qn("w:trHeight")):
        if tr_height.get(qn("w:hRule")) == "exact":
            tr_height.set(qn("w:hRule"), "atLeast")


def iter_all_paragraphs(document: _Document) -> Iterator[Paragraph]:
    for child in document.element.body.iterchildren():
        tag = child.tag.split("}")[1]
        if tag == "p":
            yield Paragraph(child, document.element.body)
        elif tag == "tbl":
            table = Table(child, document.element.body)
            for row in table.rows:
                for cell in row.cells:
                    for paragraph in cell.paragraphs:
                        yield paragraph
