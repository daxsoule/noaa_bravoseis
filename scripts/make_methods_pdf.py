#!/usr/bin/env python3
"""
make_methods_pdf.py — Render methods_section_draft_source.md to PDF via WeasyPrint.

Usage:
    uv run python make_methods_pdf.py
"""

from pathlib import Path
import markdown
from weasyprint import HTML

SRC = Path(__file__).parent.parent / "outputs" / "methods_section_draft_source.md"
OUT = Path(__file__).parent.parent / "outputs" / "methods_section_draft.pdf"

CSS = """
@page {
    size: letter;
    margin: 2.5cm 2cm;
    @bottom-right { content: counter(page); font-size: 9pt; color: #666; }
}
body {
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.5;
    color: #222;
}
h1 { font-size: 18pt; margin-top: 1.5em; page-break-after: avoid; }
h2 { font-size: 15pt; margin-top: 1.3em; page-break-after: avoid; }
h3 { font-size: 13pt; margin-top: 1em; page-break-after: avoid; }
h4 { font-size: 11pt; margin-top: 0.8em; page-break-after: avoid; }
p { margin: 0.5em 0; }
code { font-family: "Courier New", monospace; font-size: 10pt; background: #f5f5f5; padding: 1px 3px; }
pre { background: #f5f5f5; padding: 8px 12px; font-size: 9pt; overflow-x: auto; }
table { border-collapse: collapse; margin: 0.8em 0; font-size: 10pt; }
th, td { border: 1px solid #ccc; padding: 4px 8px; }
th { background: #f0f0f0; font-weight: bold; }
blockquote { border-left: 3px solid #ccc; margin-left: 0; padding-left: 1em; color: #555; }
hr { border: none; border-top: 1px solid #ccc; margin: 1.5em 0; }
em { color: #555; }
"""


def main():
    print(f"Reading: {SRC}")
    md_text = SRC.read_text()

    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc", "smarty"],
    )

    full_html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<style>{CSS}</style>
</head><body>
{html_body}
</body></html>"""

    print(f"Rendering PDF...")
    HTML(string=full_html).write_pdf(str(OUT))
    size_mb = OUT.stat().st_size / 1e6
    print(f"Saved: {OUT} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
