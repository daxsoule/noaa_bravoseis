"""
render_methods_pdf.py — Convert methods_section_draft_source.md to a
publication-quality PDF with embedded figures using markdown → HTML → WeasyPrint.
"""

import re
import base64
import io
import markdown
from weasyprint import HTML
from pathlib import Path
from PIL import Image

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


# Figures with baked-in captions that should be cropped.
# Value is the fraction of the image height to keep (crop bottom).
CROP_FIGURES = {
    "recording_timeline.png": 0.78,
    "detection_rate_timeline.png": 0.80,
    "duration_vs_peak_freq.png": 0.80,
    "example_detection_20190417_0919.png": 0.82,
    "event_montage_tphase.png": 0.82,
    "event_montage_icequake.png": 0.82,
    "event_montage_vessel.png": 0.82,
    "icequake_seaice_6panel.png": 0.85,
    "sound_speed_profile.png": 0.85,
}


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


def _crop_and_encode(path):
    """Crop baked-in caption from bottom of figure if needed, return data URI."""
    fname = path.name
    keep_frac = CROP_FIGURES.get(fname)

    if keep_frac is not None:
        img = Image.open(path)
        w, h = img.size
        cropped = img.crop((0, 0, w, int(h * keep_frac)))
        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    else:
        return f"file://{path}"


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

        # Insert each image right after the </p> that contains its reference.
        # This handles blockquotes with multiple figures (markdown sometimes
        # merges consecutive blockquotes into one).
        # Only embed for refs in a <p> that also contains "Figure:" —
        # skip refs that appear in caption text (e.g. "available in ...").
        for ref in png_refs:
            ref_code = f'<code>{ref}</code>'
            pos = bq_html.find(ref_code)
            if pos == -1:
                continue

            # Find the enclosing <p>...</p> for this ref
            p_start = bq_html.rfind('<p>', 0, pos)
            p_end = bq_html.find('</p>', pos)
            if p_start == -1 or p_end == -1:
                continue
            enclosing_p = bq_html[p_start:p_end]

            # Only embed if this <p> is a figure title line
            if 'Figure:' not in enclosing_p and 'Figure ' not in enclosing_p:
                continue

            path = fig_index.get(ref) or fig_index.get(Path(ref).name)
            if not path or not path.exists():
                err = (f'<p style="color:#c00; font-style:italic;">'
                       f'[Figure not found: {ref}]</p>')
                insert_pos = p_end + len('</p>')
                bq_html = bq_html[:insert_pos] + '\n' + err + bq_html[insert_pos:]
                continue

            src = _crop_and_encode(path)
            img_tag = (f'<img src="{src}" '
                       f'style="max-width:100%; margin:8pt 0; display:block;" />')
            insert_pos = p_end + len('</p>')
            bq_html = bq_html[:insert_pos] + '\n' + img_tag + bq_html[insert_pos:]

        return bq_html

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
    page-break-after: avoid;
}
h3 {
    font-size: 12pt;
    margin-top: 16pt;
    page-break-after: avoid;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 10pt 0;
    font-size: 10pt;
    page-break-inside: avoid;
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
    margin: 10pt 0;
    padding: 6pt 10pt;
    background-color: #f7f9fc;
    font-size: 10pt;
    page-break-inside: auto;
}
blockquote strong {
    color: #2a5a8a;
}
blockquote img {
    max-width: 100%;
    max-height: 7in;
    margin: 6pt 0;
    object-fit: contain;
    page-break-before: auto;
    page-break-after: auto;
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
    margin: 12pt 0;
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
