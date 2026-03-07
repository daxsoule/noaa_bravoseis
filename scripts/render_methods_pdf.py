"""
render_methods_pdf.py — Convert methods_section_draft_source.md to a
publication-quality PDF with embedded figures using markdown → HTML → WeasyPrint.
"""

import re
import markdown
from weasyprint import HTML
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "outputs" / "methods_section_draft_source.md"
OUT = REPO / "outputs" / "methods_section_draft.pdf"
FIG_ROOT = REPO / "outputs" / "figures"

# Map short figure names used in the markdown to actual file paths.
# Built by searching known figure directories.
FIGURE_DIRS = [
    FIG_ROOT / "exploratory" / "detection",
    FIG_ROOT / "exploratory" / "onsets",
    FIG_ROOT / "exploratory" / "seismic_onsets",
    FIG_ROOT / "exploratory" / "clustering",
    FIG_ROOT / "exploratory" / "association",
    FIG_ROOT / "exploratory" / "location",
    FIG_ROOT / "exploratory",
    FIG_ROOT / "paper",
]


def _build_figure_index():
    """Build a dict mapping short names (as used in markdown) to absolute paths."""
    index = {}
    for d in FIGURE_DIRS:
        if not d.exists():
            continue
        for f in d.glob("*.png"):
            # Key by the relative path from FIG_ROOT (e.g. "paper/event_montage_tphase.png")
            rel = f.relative_to(FIG_ROOT)
            index[str(rel)] = f
            # Also key by just the filename for simple references
            index[f.name] = f
    return index


def _embed_figures(html: str, fig_index: dict) -> str:
    """Find figure references in blockquotes and insert <img> tags.

    The markdown produces blockquotes like:
        <blockquote>
        <p><strong>Figure: Recording Timeline</strong> (<code>recording_timeline.png</code>)</p>
        <p><strong>Temporary Caption:</strong> ...</p>
        </blockquote>

    For multi-file references like (<code>file1.png</code>, <code>file2.png</code>, <code>file3.png</code>),
    we embed all images.

    We insert an <img> tag right after the figure title line.
    """
    # Match <code>some_path.png</code> patterns inside blockquotes
    def replace_blockquote(bq_match):
        bq_html = bq_match.group(0)

        # Find all .png references in <code> tags within this blockquote
        png_refs = re.findall(r'<code>([^<]*\.png)</code>', bq_html)
        if not png_refs:
            return bq_html

        # Build image tags
        img_tags = []
        for ref in png_refs:
            # Try exact match first, then just filename
            path = fig_index.get(ref) or fig_index.get(Path(ref).name)
            if path and path.exists():
                img_tags.append(
                    f'<img src="file://{path}" '
                    f'style="max-width:100%; margin:8pt 0; display:block;" />'
                )
            else:
                img_tags.append(
                    f'<p style="color:#c00; font-style:italic;">'
                    f'[Figure not found: {ref}]</p>'
                )

        # Insert images after the first <p> (the title line)
        # Find the end of the first </p> in the blockquote
        first_p_end = bq_html.find('</p>')
        if first_p_end == -1:
            return bq_html

        insert_pos = first_p_end + len('</p>')
        images_html = '\n'.join(img_tags)
        return bq_html[:insert_pos] + '\n' + images_html + bq_html[insert_pos:]

    # Process each blockquote
    result = re.sub(
        r'<blockquote>.*?</blockquote>',
        replace_blockquote,
        html,
        flags=re.DOTALL,
    )
    return result


CSS = """
@page {
    size: letter;
    margin: 0.85in;
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
blockquote img {
    max-width: 100%;
    max-height: 7.5in;
    margin: 8pt 0;
    object-fit: contain;
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
    fig_index = _build_figure_index()
    print(f"Figure index: {len(fig_index)} entries")

    md_text = SRC.read_text()

    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "smarty"],
    )

    # Embed figures
    html_body = _embed_figures(html_body, fig_index)

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

    # Debug: write HTML for inspection
    html_path = OUT.with_suffix('.html')
    html_path.write_text(full_html)
    print(f"HTML written to {html_path}")

    HTML(string=full_html, base_url=str(REPO)).write_pdf(str(OUT))
    print(f"PDF written to {OUT}")
    print(f"  Size: {OUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    render()
