#!/usr/bin/env python3
"""
make_methods_pdfs.py — Render individual methods documents to PDF with embedded figures.

Usage:
    uv run python scripts/make_methods_pdfs.py              # all docs
    uv run python scripts/make_methods_pdfs.py 01_detection  # single doc
"""

import sys
from pathlib import Path
import markdown
from weasyprint import HTML

METHODS_DIR = Path(__file__).parent.parent / "outputs" / "methods"
BASE_URL = str((Path(__file__).parent.parent / "outputs" / "methods").resolve()) + "/"

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
code {
    font-family: "Courier New", monospace;
    font-size: 10pt;
    background: #f5f5f5;
    padding: 1px 3px;
}
pre {
    background: #f5f5f5;
    padding: 8px 12px;
    font-size: 9pt;
    overflow-x: auto;
}
table { border-collapse: collapse; margin: 0.8em 0; font-size: 10pt; }
th, td { border: 1px solid #ccc; padding: 4px 8px; }
th { background: #f0f0f0; font-weight: bold; }
blockquote {
    border-left: 3px solid #ccc;
    margin-left: 0;
    padding-left: 1em;
    color: #555;
}
hr { border: none; border-top: 1px solid #ccc; margin: 1.5em 0; }
em { color: #555; }
img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 1em auto;
}
"""


def render_one(md_path: Path):
    """Render a single markdown file to PDF."""
    print(f"Reading: {md_path.name}")
    md_text = md_path.read_text()

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

    out_path = md_path.with_suffix(".pdf")
    HTML(string=full_html, base_url=BASE_URL).write_pdf(str(out_path))
    size_mb = out_path.stat().st_size / 1e6
    print(f"  -> {out_path.name} ({size_mb:.1f} MB)")
    return out_path


def main():
    if len(sys.argv) > 1:
        # Render specific doc(s)
        for name in sys.argv[1:]:
            if not name.endswith(".md"):
                name += ".md"
            md_path = METHODS_DIR / name
            if not md_path.exists():
                print(f"ERROR: {md_path} not found")
                sys.exit(1)
            render_one(md_path)
    else:
        # Render all .md files in methods dir
        md_files = sorted(METHODS_DIR.glob("*.md"))
        if not md_files:
            print("No .md files found in outputs/methods/")
            sys.exit(1)
        for md_path in md_files:
            render_one(md_path)

    print("Done.")


if __name__ == "__main__":
    main()
