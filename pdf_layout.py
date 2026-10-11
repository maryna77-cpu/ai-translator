"""Layout analysis helpers: turn vector figures/diagrams into raster images.

The source PDFs of technical manuals usually draw figures (block diagrams, timing
diagrams, logos ...) with *vector* graphics.  ``pdf2docx`` cannot translate vector
art into Word faithfully - it either drops it or mis-detects the diagram lines as a
"stream table" full of loose labels.  This module therefore pre-processes the PDF:

1. detect the regions of the page that hold a figure (clustered vector primitives),
2. render each region to a high resolution PNG (exactly as it looks in the PDF),
3. redact (remove) the original vector art *and* the text inside the region,
4. insert the rendered PNG back at the very same position.

After this step the PDF contains real images for every figure, while genuine tables
and all body text stay untouched, so ``pdf2docx`` can build a faithful Word skeleton
whose tables can be translated and whose figures are pixel-perfect copies.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Any

import fitz  # PyMuPDF

log = logging.getLogger("document-translator")

# Redaction behaviour constants (PyMuPDF >= 1.23)
_REDACT_IMAGE_REMOVE = getattr(fitz, "PDF_REDACT_IMAGE_REMOVE", 2)
_REDACT_LINE_ART_IF_TOUCHED = getattr(fitz, "PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED", 1)
_REDACT_TEXT_REMOVE = getattr(fitz, "PDF_REDACT_TEXT_REMOVE", 0)


@dataclass
class FigureRegion:
    rect: fitz.Rect
    primitives: int = 0
    has_curves: bool = False
    merged: int = 1


@dataclass
class PageFigures:
    page_index: int
    regions: list[FigureRegion] = field(default_factory=list)
    dropped_tables: int = 0


DEFAULT_FIGURE_SETTINGS = {
    "figure_dpi": 240,              # resolution of the rasterized figures
    "cluster_tolerance": 6.0,       # pt gap tolerated inside one figure
    "merge_tolerance": 5.0,         # pt gap tolerated between figure clusters
    "figure_merge_tolerance": 26.0, # pt gap tolerated between diagram rows/labels
    "min_figure_width": 70.0,       # pt
    "min_figure_height": 45.0,      # pt
    "min_logo_width": 12.0,         # pt, small curve clusters (logos) still count
    "header_band": 0.11,            # fraction of page height considered header
    "footer_band": 0.88,            # fraction of page height considered footer
    "rasterize_logos": True,
}


def _primitive_rect(item: tuple) -> fitz.Rect | None:
    """Bounding rect of a single drawing item (``re``/``l``/``c``/``qu``)."""
    kind = item[0]
    try:
        if kind == "re":
            return fitz.Rect(item[1])
        if kind == "l":
            p1, p2 = item[1], item[2]
            return fitz.Rect(min(p1.x, p2.x), min(p1.y, p2.y), max(p1.x, p2.x), max(p1.y, p2.y))
        if kind == "qu":
            return fitz.Rect(item[1].rect)
        if kind == "c":
            points = []
            for entry in item[1:]:
                if isinstance(entry, (fitz.Point, tuple, list)):
                    points.append(entry)
            if not points:
                return None
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            return fitz.Rect(min(xs), min(ys), max(xs), max(ys))
    except Exception:
        return None
    return None


def _is_white(fill) -> bool:
    if fill is None:
        return False
    try:
        return all(abs(c - 1.0) < 0.02 for c in fill)
    except TypeError:
        return False


def _collect_primitives(page: fitz.Page, cfg: dict) -> list[dict[str, Any]]:
    """All vector primitives of the page, minus page furniture and white fills."""
    page_rect = page.rect
    width, height = page_rect.width, page_rect.height
    header_cut = height * float(cfg.get("header_band", 0.11))
    footer_cut = height * float(cfg.get("footer_band", 0.88))

    prims: list[dict[str, Any]] = []
    for drawing in page.get_drawings():
        fill = drawing.get("fill")
        for item in drawing["items"]:
            rect = _primitive_rect(item)
            if rect is None or rect.is_empty or rect.is_infinite:
                continue
            kind = item[0]
            # White background fills carry no information about figure extents.
            if kind == "re" and _is_white(fill):
                continue
            # Full page backgrounds / frames.
            if rect.width >= 0.95 * width and rect.height >= 0.95 * height:
                continue
            thin_h = rect.height <= 2.5
            thin_v = rect.width <= 2.5
            # Header/footer separator rules: very thin and (almost) page wide.
            if (thin_h and rect.width >= 0.6 * width and (rect.y0 < header_cut or rect.y0 > footer_cut)):
                continue
            # Full height vertical rules are page frames, figure borders are shorter.
            if thin_v and rect.height >= 0.92 * height:
                continue
            # Microscopic specks are noise.
            if rect.width < 1.2 and rect.height < 1.2 and kind != "c":
                continue
            prims.append({"rect": rect, "kind": kind})
    return prims


def _gap(a: fitz.Rect, b: fitz.Rect) -> tuple[float, float]:
    dx = max(0.0, max(a.x0, b.x0) - min(a.x1, b.x1))
    dy = max(0.0, max(a.y0, b.y0) - min(a.y1, b.y1))
    return dx, dy


def _cluster_primitives(prims: list[dict[str, Any]], tolerance: float) -> list[list[dict[str, Any]]]:
    """Union-find clustering of primitives whose rects are within ``tolerance``."""
    parent = list(range(len(prims)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(len(prims)):
        for j in range(i + 1, len(prims)):
            dx, dy = _gap(prims[i]["rect"], prims[j]["rect"])
            if dx <= tolerance and dy <= tolerance:
                union(i, j)

    groups: dict[int, list[dict[str, Any]]] = {}
    for i, prim in enumerate(prims):
        groups.setdefault(find(i), []).append(prim)
    return list(groups.values())


def _cluster_bbox(group: list[dict[str, Any]]) -> fitz.Rect:
    rect = fitz.Rect(group[0]["rect"])
    for prim in group[1:]:
        rect |= prim["rect"]
    return rect


def _merge_clusters(clusters: list[list[dict[str, Any]]], tolerance: float,
                    forbidden: list[fitz.Rect] | None = None,
                    max_area: float | None = None) -> list[list[dict[str, Any]]]:
    """Merge clusters that overlap, contain each other, or nearly touch.

    Merges whose resulting bbox would swallow one of the ``forbidden`` rects (real
    tables) are refused so that tables never end up inside a figure region.
    """
    boxes = [_cluster_bbox(c) for c in clusters]
    parent = list(range(len(clusters)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def allowed(box: fitz.Rect) -> bool:
        if max_area is not None and box.get_area() > max_area:
            return False
        for rect in forbidden or ():
            if _contained(rect, box, 0.85):
                return False
        return True

    changed = True
    while changed:
        changed = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if find(i) == find(j):
                    continue
                a, b = boxes[find(i)], boxes[find(j)]
                dx, dy = _gap(a, b)
                if not (a.contains(b) or b.contains(a) or (dx <= tolerance and dy <= tolerance)):
                    continue
                merged = a | b
                if not allowed(merged):
                    continue
                parent[find(j)] = find(i)
                boxes[find(i)] = merged
                changed = True
    groups: dict[int, list[dict[str, Any]]] = {}
    for i, cluster in enumerate(clusters):
        groups.setdefault(find(i), []).extend(cluster)
    return list(groups.values())


def _contained(inner: fitz.Rect, outer: fitz.Rect, ratio: float = 0.9) -> bool:
    common = inner & outer
    if common.is_empty:
        return False
    return common.get_area() >= ratio * inner.get_area()


def _classify_tables(page: fitz.Page) -> tuple[list[fitz.Rect], list[fitz.Rect]]:
    """Split detected tables into protected tables and tentative (maybe diagram) ones.

    Table finders also react to the box/arrow grids inside diagrams.  Genuine data
    tables practically always have three or more rows, while one/two row detections
    are usually diagram boxes, bullet rows or header strips.  Those tentative tables
    are absorbed into a figure region when they turn out to lie inside one.
    """
    protected: list[fitz.Rect] = []
    tentative: list[fitz.Rect] = []
    try:
        found = page.find_tables()
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("Table detection failed on page %d: %s", page.number + 1, exc)
        return protected, tentative
    for table in found.tables:
        rect = fitz.Rect(table.bbox)
        if rect.get_area() <= 0:
            continue
        (protected if table.row_count >= 3 else tentative).append(rect)
    return protected, tentative


def detect_figure_regions(page: fitz.Page, cfg: dict) -> PageFigures:
    """Detect the figure (diagram/logo) regions of one page."""
    settings = {**DEFAULT_FIGURE_SETTINGS, **(cfg or {})}
    result = PageFigures(page_index=page.number)
    page_rect = page.rect
    height = page_rect.height
    header_cut = height * float(settings["header_band"])
    footer_cut = height * float(settings["footer_band"])

    prims = _collect_primitives(page, settings)
    if not prims:
        return result

    real_tables, fake_tables = _classify_tables(page)

    tolerance = float(settings["cluster_tolerance"])
    merge_tolerance = float(settings["merge_tolerance"])
    page_area = page_rect.get_area()
    clusters = _cluster_primitives(prims, tolerance)
    # Ruling of real tables is not figure material (and must not bridge merges).
    clusters = [c for c in clusters if not any(_contained(_cluster_bbox(c), t, 0.85) for t in real_tables)]
    if not clusters:
        return result
    # Stage 1: tight merge (touching strokes, closed boxes).
    clusters = _merge_clusters(clusters, merge_tolerance, forbidden=real_tables,
                               max_area=0.8 * page_area)
    # Stage 2: loose merge, diagram rows are separated by label text only.
    clusters = _merge_clusters(clusters, float(settings["figure_merge_tolerance"]),
                               forbidden=real_tables, max_area=0.8 * page_area)

    candidates: list[FigureRegion] = []
    for cluster in clusters:
        rect = _cluster_bbox(cluster)
        kinds = {p["kind"] for p in cluster}
        has_curves = "c" in kinds
        in_furniture_band = rect.y1 <= header_cut or rect.y0 >= footer_cut
        is_logo = has_curves and rect.width >= float(settings["min_logo_width"]) and len(cluster) >= 2
        big_enough = (
            rect.width >= float(settings["min_figure_width"])
            and rect.height >= float(settings["min_figure_height"])
        )
        if not (big_enough or (is_logo and settings.get("rasterize_logos", True))):
            continue
        # A lone shaded rect wrapping prose (note boxes) is not a figure.
        if len(cluster) <= 2 and not has_curves:
            continue
        # Tables win: a cluster that sits inside a real table is cell decoration.
        if any(_contained(rect, t, 0.85) for t in real_tables):
            continue
        if in_furniture_band and not is_logo:
            continue
        candidates.append(FigureRegion(rect=rect, primitives=len(cluster), has_curves=has_curves))

    # Diagram boxes that the table finder reported inside a figure belong to it.
    kept_tables: list[fitz.Rect] = list(real_tables)
    for table in fake_tables:
        if any(_contained(table, c.rect, 0.85) for c in candidates):
            result.dropped_tables += 1
        else:
            kept_tables.append(table)

    # Grow each region so that labels drawn as text next to the line art are inside.
    words = page.get_text("words")
    for region in candidates:
        grown = fitz.Rect(region.rect)
        for w in words:
            span = fitz.Rect(w[:4])
            # Labels and captions that touch the figure become part of the raster;
            # anything only grazing the border stays translatable text.
            if span.height > 22 or span.width > 300:
                continue
            common = span & grown
            if not common.is_empty and common.get_area() >= 0.15 * span.get_area():
                grown |= span
        # Small safety margin so strokes on the border are not clipped.
        grown += (-2.0, -2.0, 2.0, 2.0)
        region.rect = grown & page_rect

    # Drop regions that ended up inside a kept table after growing, then de-duplicate
    # nested regions (a diagram fragment fully inside an accepted figure).
    accepted: list[FigureRegion] = []
    for region in sorted(candidates, key=lambda r: -r.rect.get_area()):
        if any(_contained(region.rect, t, 0.9) for t in kept_tables):
            continue
        if any(_contained(region.rect, other.rect, 0.9) for other in accepted):
            continue
        accepted.append(region)

    result.regions = sorted(accepted, key=lambda r: (r.rect.y0, r.rect.x0))
    return result


def rasterize_figures(page: fitz.Page, figures: PageFigures, cfg: dict) -> int:
    """Replace the vector art of every figure region by a rendered PNG image."""
    if not figures.regions:
        return 0
    dpi = float((cfg or {}).get("figure_dpi", DEFAULT_FIGURE_SETTINGS["figure_dpi"]))
    zoom = dpi / 72.0

    images: list[tuple[fitz.Rect, bytes]] = []
    for region in figures.regions:
        pix = page.get_pixmap(clip=region.rect, matrix=fitz.Matrix(zoom, zoom), alpha=False)
        images.append((fitz.Rect(region.rect), pix.tobytes("png")))

    for rect, _ in images:
        page.add_redact_annot(rect)
    page.apply_redactions(
        images=_REDACT_IMAGE_REMOVE,
        graphics=_REDACT_LINE_ART_IF_TOUCHED,
        text=_REDACT_TEXT_REMOVE,
    )
    for rect, png in images:
        page.insert_image(rect, stream=png, overlay=True)
    return len(images)


def prepare_pdf_stream(pdf_path: str, cfg: dict, page_range: tuple[int, int] | None = None) -> tuple[bytes, list[PageFigures]]:
    """Return a copy of the PDF whose figures are raster images, plus per page info."""
    settings = {**DEFAULT_FIGURE_SETTINGS, **(cfg or {})}
    doc = fitz.open(pdf_path)
    stats: list[PageFigures] = []
    start, end = 0, len(doc)
    if page_range:
        start, end = page_range
    for index in range(start, min(end, len(doc))):
        page = doc[index]
        figures = detect_figure_regions(page, settings)
        count = rasterize_figures(page, figures, settings)
        if count:
            log.info(
                "Page %d: rasterized %d figure region(s), %d false table(s) absorbed.",
                index + 1, count, figures.dropped_tables,
            )
        stats.append(figures)
    buffer = io.BytesIO()
    doc.save(buffer, garbage=3, deflate=True)
    doc.close()
    return buffer.getvalue(), stats
