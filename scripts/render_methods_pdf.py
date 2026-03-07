"""
render_methods_pdf.py — Convert methods_section_draft_source.md to a
publication-quality PDF using markdown → HTML → WeasyPrint.
"""

import markdown
from weasyprint import HTML
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "outputs" / "methods_section_draft_source.md"
OUT = REPO / "outputs" / "methods_section_draft.pdf"

CSS = """
@page {
    size: letter;
    margin: 1in;
}
body {
    font-family: "DejaVu Serif", Georgia, "Times New Roman", serif;
    font-size: 11pt;
    line-height: 1.45;
    color: #1a1a1a;
    max-width: 100%;
}
h1 {
    font-size: 16pt;
    border-bottom: 2px solid #333;
    padding-bottom: 4pt;
    margin-top: 24pt;
}
h2 {
    font-size: 14pt;
    border-bottom: 1px solid #999;
    padding-bottom: 3pt;
    margin-top: 20pt;
}
h3 {
    font-size: 12pt;
    margin-top: 16pt;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 12pt 0;
    font-size: 10pt;
}
th, td {
    border: 1px solid #999;
    padding: 4pt 8pt;
    text-align: left;
}
th {
    background-color: #f0f0f0;
    font-weight: bold;
}
blockquote {
    border-left: 3px solid #4a86c8;
    margin: 14pt 0;
    padding: 8pt 12pt;
    background-color: #f7f9fc;
    font-size: 10pt;
}
blockquote strong {
    color: #2a5a8a;
}
code {
    font-family: "DejaVu Sans Mono", "Courier New", monospace;
    font-size: 9pt;
    background-color: #f4f4f4;
    padding: 1pt 3pt;
    border-radius: 2pt;
}
pre {
    background-color: #f4f4f4;
    padding: 8pt;
    border-radius: 4pt;
    font-size: 9pt;
    overflow-x: auto;
}
hr {
    border: none;
    border-top: 1px solid #ccc;
    margin: 20pt 0;
}
em {
    color: #555;
}
"""


def render():
    md_text = SRC.read_text()

    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "smarty"],
    )

    full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>{CSS}</style>
</head>
<body>
{html_body}
</body>
</html>"""

    HTML(string=full_html).write_pdf(str(OUT))
    print(f"PDF written to {OUT}")
    print(f"  Size: {OUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    render()
