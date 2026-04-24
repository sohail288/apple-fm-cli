#!/usr/bin/env python3
"""
Generate a small PDF in sample_corpus/ for the RAG ingest demo.
Requires: pip install fpdf2  (or full examples/rag_demo requirements.txt)
"""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF  # type: ignore[import-untyped]

TEXT = (
    "Acme - Legal & Records (DRAFT)\n\n"
    "NDA and vendor agreement originals must be stored in the IronVault archive room, "
    "basement B2, for seven (7) years after contract end. Scanned copies may live in the "
    "compliance S3 bucket (encrypted) but cannot replace the physical originals until "
    "Legal signs Form LV-9.\n\n"
    "The whistleblower hotline is 1-800-ACME-LAW, operated by an outside firm. Do not "
    "route PII through that number; use the internal ethics portal for personnel matters."
)


def main() -> None:
    base = Path(__file__).resolve().parent
    out = base / "sample_corpus" / "04_legal" / "legal_handbook_excerpt.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("helvetica", size=11)
    for line in TEXT.split("\n"):
        if not line:
            pdf.ln(4)
            continue
        pdf.multi_cell(0, 6, line)
        pdf.ln(2)
    pdf.output(str(out))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
