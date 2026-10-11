"""Offline end-to-end test of the layout-preserving pipeline (no API calls).

Run:  python tests/test_layout_pipeline.py
Uses a fake provider that returns a marked pseudo-translation so the whole
PDF -> figure rasterizing -> pdf2docx -> translate-in-place -> DOCX flow is
exercised and the resulting Word structure is asserted.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docx import Document
from docx.oxml.ns import qn

from document_translator import translate_document
from providers.base import BaseProvider, ProviderResult

REPO = Path(__file__).resolve().parents[1]


class FakeProvider(BaseProvider):
    name = "fake"

    def generate(self, system_prompt, user_prompt, *, temperature, max_output_tokens):
        payload = json.loads(user_prompt)
        out = []
        for item in payload["target"]:
            out.append({"id": item["id"], "translation": "﴿FA﴾ " + item["text"]})
        return ProviderResult(content=json.dumps(out, ensure_ascii=False))


def main():
    pdf = REPO / "SAI.pdf"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.docx"
        cfg = {"translation": {"document_batch_max_items": 8, "document_batch_max_chars": 4000},
               "layout": {"figure_dpi": 150}}
        translate_document(pdf, out, cfg, "fake", provider=FakeProvider(), pages=(0, 10))
        assert out.exists(), "output docx missing"

        doc = Document(str(out))
        body = doc.element.body
        paragraphs = 0
        translated = 0
        images = 0
        tables = 0
        rtl_paras = 0
        for child in body.iterchildren():
            tag = child.tag.split("}")[1]
            if tag == "p":
                paragraphs += 1
                text = "".join(n.text or "" for n in child.iter(qn("w:t")))
                images += len(child.findall(".//" + qn("w:drawing")))
                if text.startswith("﴿FA﴾"):
                    translated += 1
                    assert child.find(qn("w:pPr")) is not None and \
                        child.find(qn("w:pPr")).find(qn("w:bidi")) is not None, "missing bidi"
                    rtl_paras += 1
                if "The SAI is mainly composed" in text:
                    assert text.startswith("﴿FA﴾"), "source paragraph left untranslated"
            elif tag == "tbl":
                tables += 1
                for t in child.iter(qn("w:t")):
                    if (t.text or "").startswith("﴿FA﴾"):
                        translated += 1
        assert images >= 8, f"expected figure images, got {images}"
        assert tables >= 5, f"expected tables, got {tables}"
        assert translated >= 100, f"expected many translated units, got {translated}"
        assert rtl_paras >= 50
        # exact row heights must be relaxed
        exact_rows = [e for e in body.iter(qn("w:trHeight")) if e.get(qn("w:hRule")) == "exact"]
        assert not exact_rows, "exact row heights left in place"
        # CID subset fonts must be gone
        fonts = {f.get(qn("w:ascii")) for f in body.iter(qn("w:rFonts"))}
        assert not any((f or "").startswith("CIDFont") for f in fonts), fonts
        print(f"OK: paragraphs={paragraphs} translated={translated} images={images} tables={tables} fonts={fonts}")


if __name__ == "__main__":
    main()
