#!/usr/bin/env python3
"""
interactive_umap.py — Generate interactive UMAP HTML files with Plotly.

Creates one standalone HTML per band with:
- Lasso / box select
- Dropdown to color by cluster or any spectral feature
- "Characterize Selection" button that shows full feature statistics
- Name regions, copy summary to clipboard for pasting back to Claude
- Download all saved regions as JSON

Usage:
    uv run python scripts/interactive_umap.py
    uv run python scripts/interactive_umap.py --band low

Spec: specs/002-event-discrimination/
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go

# === Paths ===
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
DATA_DIR = OUTPUT_DIR / "data"
FIG_DIR = OUTPUT_DIR / "figures" / "exploratory" / "clustering"

# === Feature columns ===
STAT_FEATURES = [
    "peak_freq_hz", "duration_s", "snr", "bandwidth_hz",
    "spectral_slope", "freq_modulation", "spectral_centroid_hz",
    "peak_power_db", "decay_time_s", "rise_time_s",
]

COLOR_FEATURES = [
    "peak_freq_hz", "duration_s", "snr", "bandwidth_hz",
    "spectral_slope", "freq_modulation", "spectral_centroid_hz",
    "peak_power_db", "decay_time_s",
]

BANDS = ["low", "mid", "high"]

PLOT_DIV_ID = "umap-plot"


def build_post_script(band, stat_features, event_data):
    """Build JS that runs immediately after Plotly.newPlot(), inside its scope.

    The post_script receives the plot div as `gd` (graph div).
    """
    # Note: post_script uses single braces {} — it is NOT an f-string template.
    # We inject data via string concatenation.
    return """
    // --- Selection tracking ---
    var BAND = """ + json.dumps(band) + """;
    var FEATURES = """ + json.dumps(stat_features) + """;
    var EVENT_DATA = """ + json.dumps(event_data) + """;
    var gd = document.getElementById('umap-plot');

    // Button is always enabled — reads selection state on click
    document.getElementById('characterize-btn').disabled = false;
    document.getElementById('characterize-btn').style.background = '#4a90d9';
    document.getElementById('characterize-btn').style.borderColor = '#3a7bc8';
    document.getElementById('characterize-btn').style.cursor = 'pointer';

    // Read selected points directly from Plotly's internal trace state
    function getSelectedEventIds() {
        var ids = [];
        var traces = gd.data || [];
        for (var t = 0; t < traces.length; t++) {
            var trace = traces[t];
            var sp = trace.selectedpoints;
            if (sp && sp.length > 0 && trace.text) {
                for (var j = 0; j < sp.length; j++) {
                    var eid = trace.text[sp[j]];
                    if (eid) ids.push(eid);
                }
            }
        }
        return ids;
    }

    window.characterizeSelection = function() {
        // Debug: show what Plotly has in its trace state
        var traces = gd.data || [];
        var debugInfo = 'Traces: ' + traces.length + '\\n';
        for (var t = 0; t < traces.length; t++) {
            var sp = traces[t].selectedpoints;
            debugInfo += 'Trace ' + t + ' (' + (traces[t].name||'?') + '): ';
            debugInfo += 'selectedpoints=' + (sp ? sp.length : 'null');
            debugInfo += ', text=' + (traces[t].text ? traces[t].text.length : 'null');
            debugInfo += '\\n';
        }
        // Also check _fullData
        var fullTraces = gd._fullData || [];
        debugInfo += '\\n_fullData traces: ' + fullTraces.length + '\\n';
        for (var t = 0; t < Math.min(fullTraces.length, 5); t++) {
            var sp = fullTraces[t].selectedpoints;
            debugInfo += 'FullTrace ' + t + ': selectedpoints=' + (sp ? sp.length : 'null') + '\\n';
        }

        var selectedIds = getSelectedEventIds();
        debugInfo += '\\nTotal selected IDs: ' + selectedIds.length;

        if (selectedIds.length === 0) {
            // Also try _fullData
            for (var t = 0; t < fullTraces.length; t++) {
                var trace = fullTraces[t];
                var sp = trace.selectedpoints;
                if (sp && sp.length > 0 && trace.text) {
                    for (var j = 0; j < sp.length; j++) {
                        var eid = trace.text[sp[j]];
                        if (eid) selectedIds.push(eid);
                    }
                }
            }
            debugInfo += '\\nAfter _fullData fallback: ' + selectedIds.length;
        }

        if (selectedIds.length === 0) {
            alert('Debug info:\\n\\n' + debugInfo + '\\n\\nNo points found. Try lasso-selecting first.');
            return;
        }

        document.getElementById('selection-count').textContent =
            selectedIds.length.toLocaleString() + ' points selected';
        document.getElementById('selection-count').style.color = '#4a90d9';

        var idSet = new Set(selectedIds);
        window._lastSelectedIds = selectedIds;
        var stats = computeStats(idSet);
        showModal(stats, idSet.size);
    };

    function computeStats(selectedIds) {
        var stats = {};
        FEATURES.forEach(function(feat) {
            var vals = [];
            EVENT_DATA.event_id.forEach(function(eid, i) {
                if (selectedIds.has(eid)) {
                    var v = EVENT_DATA[feat][i];
                    if (v !== null && !isNaN(v)) vals.push(v);
                }
            });
            if (vals.length > 0) {
                vals.sort(function(a, b) { return a - b; });
                var sum = vals.reduce(function(a, b) { return a + b; }, 0);
                var mean = sum / vals.length;
                var mid = Math.floor(vals.length / 2);
                var median = vals.length % 2 ? vals[mid] : (vals[mid - 1] + vals[mid]) / 2;
                var p5 = vals[Math.floor(vals.length * 0.05)];
                var p95 = vals[Math.floor(vals.length * 0.95)];
                stats[feat] = {
                    n: vals.length, min: vals[0], p5: p5,
                    median: median, mean: mean, p95: p95,
                    max: vals[vals.length - 1],
                };
            }
        });
        return stats;
    }

    function fmt(v) {
        if (Math.abs(v) >= 100) return v.toFixed(1);
        if (Math.abs(v) >= 1) return v.toFixed(2);
        return v.toFixed(4);
    }

    function showModal(stats, nSelected) {
        document.getElementById('modal-title').textContent =
            'Selection: ' + nSelected.toLocaleString() + ' events (' + BAND.toUpperCase() + ' band)';

        var html = '<table><tr><th>Feature</th><th>Min</th><th>P5</th><th>Median</th><th>Mean</th><th>P95</th><th>Max</th></tr>';
        FEATURES.forEach(function(feat) {
            if (stats[feat]) {
                var s = stats[feat];
                html += '<tr><td>' + feat + '</td><td>' + fmt(s.min) + '</td><td>' + fmt(s.p5) +
                        '</td><td>' + fmt(s.median) + '</td><td>' + fmt(s.mean) +
                        '</td><td>' + fmt(s.p95) + '</td><td>' + fmt(s.max) + '</td></tr>';
            }
        });
        html += '</table>';

        document.getElementById('modal-stats').innerHTML = html;
        document.getElementById('region-modal').style.display = 'block';
        document.getElementById('modal-overlay').style.display = 'block';
        document.getElementById('region-name').focus();

        window._lastStats = stats;
        window._lastN = nSelected;
    }

    window.closeModal = function() {
        document.getElementById('region-modal').style.display = 'none';
        document.getElementById('modal-overlay').style.display = 'none';
    };

    function buildClipboardText(regionName, stats, nSelected) {
        var lines = [];
        lines.push('## UMAP Region: ' + regionName);
        lines.push('Band: ' + BAND + ' | Events: ' + nSelected.toLocaleString());
        lines.push('');
        lines.push('| Feature | Min | P5 | Median | Mean | P95 | Max |');
        lines.push('|---------|-----|-----|--------|------|-----|-----|');
        FEATURES.forEach(function(feat) {
            if (stats[feat]) {
                var s = stats[feat];
                lines.push('| ' + feat + ' | ' + fmt(s.min) + ' | ' + fmt(s.p5) +
                            ' | ' + fmt(s.median) + ' | ' + fmt(s.mean) +
                            ' | ' + fmt(s.p95) + ' | ' + fmt(s.max) + ' |');
            }
        });
        lines.push('');
        return lines.join('\\n');
    }

    var savedRegions = JSON.parse(localStorage.getItem('umap_regions_' + BAND) || '{}');
    window.updateRegionsPanel = function() {
        var list = document.getElementById('regions-list');
        var names = Object.keys(savedRegions);
        if (names.length === 0) {
            list.innerHTML = 'None yet.';
            return;
        }
        var html = '<ul style="padding-left:16px; margin:4px 0;">';
        names.forEach(function(name) {
            var r = savedRegions[name];
            html += '<li><strong>' + name + '</strong>: ' + r.n_events.toLocaleString() + ' events</li>';
        });
        html += '</ul>';
        list.innerHTML = html;
    };
    updateRegionsPanel();

    window.copyAndSave = function() {
        var name = document.getElementById('region-name').value.trim();
        if (!name) { alert('Please enter a region name'); return; }

        var text = buildClipboardText(name, window._lastStats, window._lastN);

        savedRegions[name] = {
            band: BAND, n_events: window._lastN,
            event_ids: window._lastSelectedIds || [],
            stats: window._lastStats, timestamp: new Date().toISOString(),
        };
        localStorage.setItem('umap_regions_' + BAND, JSON.stringify(savedRegions));
        updateRegionsPanel();

        // Show selectable text area so user can manually copy
        var copyBox = document.getElementById('copy-text-area');
        copyBox.value = text;
        copyBox.style.display = 'block';
        copyBox.select();

        // Try clipboard API, but don't rely on it
        navigator.clipboard.writeText(text).then(function() {
            showToast('Copied to clipboard!');
        }).catch(function() {
            try {
                document.execCommand('copy');
                showToast('Copied to clipboard!');
            } catch(e) {
                showToast('Select the text below and Ctrl+C to copy');
            }
        });
    };

    function showToast(msg) {
        var toast = document.getElementById('copy-toast');
        toast.textContent = msg;
        toast.style.display = 'block';
        setTimeout(function() { toast.style.display = 'none'; }, 2000);
    }

    window.downloadRegions = function() {
        var json = JSON.stringify(savedRegions, null, 2);
        try {
            var blob = new Blob([json], {type: 'application/json'});
            var a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = 'umap_regions_' + BAND + '.json';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        } catch(e) {
            // Fallback: open JSON in new tab for manual save-as
            var w = window.open('', '_blank');
            w.document.write('<pre>' + json.replace(/</g, '&lt;') + '</pre>');
            w.document.title = 'umap_regions_' + BAND + '.json';
            showToast('JSON opened in new tab — use Ctrl+S to save');
        }
    };

    window.showRegionsJSON = function() {
        var area = document.getElementById('regions-json-area');
        if (area.style.display === 'none') {
            area.value = JSON.stringify(savedRegions, null, 2);
            area.style.display = 'block';
            area.select();
        } else {
            area.style.display = 'none';
        }
    };

    window.clearRegions = function() {
        if (confirm('Clear all saved regions for ' + BAND.toUpperCase() + ' band?')) {
            savedRegions = {};
            localStorage.removeItem('umap_regions_' + BAND);
            updateRegionsPanel();
        }
    };

    console.log('UMAP Explorer ready. Lasso-select points then click Characterize Selection.');
    """


# HTML wrapper with modal, button, and panels
PAGE_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>UMAP Explorer - {band_upper} band</title>
  <style>
    body {{ margin: 0; padding: 10px; font-family: 'Courier New', monospace; }}

    #characterize-btn {{
      position: fixed; top: 10px; right: 10px; z-index: 6000;
      padding: 14px 24px; font-size: 16px; font-family: monospace; font-weight: bold;
      background: #4a90d9; color: white; border: 2px solid #3a7bc8; border-radius: 8px;
      cursor: pointer; box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }}
    #selection-count {{
      position: fixed; top: 62px; right: 10px; z-index: 6000;
      font-family: monospace; font-size: 13px; color: #666;
      background: white; padding: 4px 10px; border-radius: 4px; border: 1px solid #ddd;
    }}

    #region-modal {{
      display: none; position: fixed; top: 50%; left: 50%;
      transform: translate(-50%, -50%);
      background: white; border: 2px solid #333; border-radius: 8px;
      padding: 24px; z-index: 10000; max-width: 700px; width: 90%;
      max-height: 80vh; overflow-y: auto;
      box-shadow: 0 8px 32px rgba(0,0,0,0.3);
      font-family: 'Courier New', monospace; font-size: 13px;
    }}
    #modal-overlay {{
      display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(0,0,0,0.4); z-index: 9999;
    }}
    #region-modal h3 {{
      margin-top: 0; font-size: 16px;
      border-bottom: 1px solid #ccc; padding-bottom: 8px;
    }}
    #region-modal table {{
      border-collapse: collapse; width: 100%; margin: 10px 0;
    }}
    #region-modal th, #region-modal td {{
      text-align: right; padding: 3px 8px; border-bottom: 1px solid #eee;
    }}
    #region-modal th:first-child, #region-modal td:first-child {{
      text-align: left;
    }}
    #region-modal input[type=text] {{
      width: 250px; padding: 6px; font-family: monospace; font-size: 13px;
      border: 1px solid #999; border-radius: 4px;
    }}
    .modal-btn {{
      padding: 8px 16px; margin: 4px; border: 1px solid #666; border-radius: 4px;
      cursor: pointer; font-family: monospace; font-size: 13px; background: #f0f0f0;
    }}
    .modal-btn:hover {{ background: #ddd; }}
    .modal-btn.primary {{ background: #4a90d9; color: white; border-color: #3a7bc8; }}
    .modal-btn.primary:hover {{ background: #3a7bc8; }}

    #saved-regions-panel {{
      position: fixed; bottom: 10px; right: 10px;
      background: white; border: 1px solid #ccc; border-radius: 6px;
      padding: 12px; z-index: 5000; font-family: monospace; font-size: 12px;
      max-width: 350px; max-height: 200px; overflow-y: auto;
      box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }}
    #copy-toast {{
      display: none; position: fixed; top: 20px; right: 20px;
      background: #4a90d9; color: white; padding: 12px 20px;
      border-radius: 6px; z-index: 20000; font-family: monospace;
    }}
  </style>
</head>
<body>
  <div style="margin-bottom: 5px; color: #666; font-size: 12px;">
    1. Lasso-select a region &nbsp; 2. Click "Characterize Selection" &nbsp;
    3. Name it &nbsp; 4. Copy to clipboard &nbsp; 5. Paste into Claude
  </div>

  <button id="characterize-btn" onclick="characterizeSelection()">
    Characterize Selection
  </button>
  <div id="selection-count">No points selected</div>

  <div id="modal-overlay" onclick="closeModal()"></div>
  <div id="region-modal">
    <h3 id="modal-title">Selection Summary</h3>
    <div id="modal-stats"></div>
    <div style="margin-top: 12px;">
      <label>Region name: </label>
      <input type="text" id="region-name" placeholder="e.g. texas_peninsula">
    </div>
    <div style="margin-top: 12px;">
      <button class="modal-btn primary" onclick="copyAndSave()">Copy to Clipboard & Save</button>
      <button class="modal-btn" onclick="closeModal()">Close</button>
    </div>
    <textarea id="copy-text-area" style="display:none; width:100%; height:180px; margin-top:12px; font-family:monospace; font-size:11px; border:1px solid #999; border-radius:4px; padding:6px;" readonly onclick="this.select()"></textarea>
  </div>

  <div id="saved-regions-panel">
    <strong>Saved Regions</strong>
    <div id="regions-list">None yet.</div>
    <div style="margin-top: 8px;">
      <button class="modal-btn" onclick="downloadRegions()" style="padding:4px 10px; font-size:11px;">Download JSON</button>
      <button class="modal-btn" onclick="showRegionsJSON()" style="padding:4px 10px; font-size:11px;">Show JSON</button>
      <button class="modal-btn" onclick="clearRegions()" style="padding:4px 10px; font-size:11px;">Clear All</button>
    </div>
    <textarea id="regions-json-area" style="display:none; width:100%; height:200px; margin-top:8px; font-family:monospace; font-size:10px; border:1px solid #999; border-radius:4px; padding:6px;" readonly onclick="this.select()"></textarea>
  </div>

  <div id="copy-toast"></div>

  {plotly_div}
</body>
</html>"""


def load_data():
    """Load UMAP coordinates and merge with features."""
    umap_df = pd.read_parquet(DATA_DIR / "umap_coordinates.parquet")
    feat_df = pd.read_parquet(DATA_DIR / "event_features.parquet")
    merged = umap_df.merge(feat_df, on=["event_id", "mooring", "detection_band"],
                           how="left")
    return merged


def make_band_html(df, band, outpath):
    """Build an interactive Plotly HTML for one band."""
    band_df = df[df["detection_band"] == band].reset_index(drop=True)
    n_events = len(band_df)

    max_points = 30000
    if n_events > max_points:
        rng = np.random.RandomState(42)
        idx = rng.choice(n_events, max_points, replace=False)
        idx.sort()
        band_df = band_df.iloc[idx].reset_index(drop=True)
        sampled = True
        print(f"  Subsampled {n_events:,} -> {max_points:,} for interactivity")
    else:
        sampled = False

    x = band_df["umap_1"].values
    y = band_df["umap_2"].values
    event_ids = band_df["event_id"].values
    labels = band_df["cluster_id_numeric"].values

    # --- Build Plotly figure ---
    noise_mask = labels == -1
    cluster_ids = sorted(set(labels) - {-1})
    fig = go.Figure()

    if noise_mask.any():
        fig.add_trace(go.Scatter(
            x=x[noise_mask], y=y[noise_mask], mode="markers",
            marker=dict(size=2, color="lightgray", opacity=0.3),
            name="noise", text=event_ids[noise_mask],
            hovertemplate="event: %{text}<br>(%{x:.2f}, %{y:.2f})",
            visible=True,
        ))

    for cid in cluster_ids:
        mask = labels == cid
        fig.add_trace(go.Scatter(
            x=x[mask], y=y[mask], mode="markers",
            marker=dict(size=2, opacity=0.5),
            name=f"{band}_{cid} (n={mask.sum():,})",
            text=event_ids[mask],
            hovertemplate="event: %{text}<br>(%{x:.2f}, %{y:.2f})",
            visible=True,
        ))

    n_cluster_traces = 1 + len(cluster_ids)

    for feat in COLOR_FEATURES:
        vals = band_df[feat].values.copy()
        finite = vals[np.isfinite(vals)]
        vmin, vmax = (np.percentile(finite, [2, 98]) if len(finite) > 0
                      else (0, 1))
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers",
            marker=dict(size=2, color=vals, colorscale="Viridis",
                        cmin=vmin, cmax=vmax,
                        colorbar=dict(title=feat, len=0.6), opacity=0.4),
            name=feat, text=event_ids,
            hovertemplate=(f"{feat}: %{{marker.color:.3g}}<br>"
                           "event: %{text}<br>(%{x:.2f}, %{y:.2f})"),
            visible=False,
        ))

    n_feature_traces = len(COLOR_FEATURES)

    # Dropdown buttons
    buttons = []
    vis_clusters = [True] * n_cluster_traces + [False] * n_feature_traces
    buttons.append(dict(label="Clusters (HDBSCAN)", method="update",
                        args=[{"visible": vis_clusters}, {"showlegend": True}]))
    for i, feat in enumerate(COLOR_FEATURES):
        vis = [False] * n_cluster_traces + [False] * n_feature_traces
        vis[n_cluster_traces + i] = True
        buttons.append(dict(label=feat, method="update",
                            args=[{"visible": vis}, {"showlegend": False}]))

    subtitle = (f" (showing {len(band_df):,} of {n_events:,})"
                if sampled else "")
    fig.update_layout(
        title=dict(text=f"{band.upper()} band — {n_events:,} events{subtitle}",
                   font=dict(size=16)),
        xaxis_title="UMAP 1", yaxis_title="UMAP 2",
        dragmode="lasso", template="plotly_white",
        legend=dict(font=dict(size=9)),
        updatemenus=[dict(type="dropdown", direction="down",
                          x=0.01, xanchor="left", y=1.12, yanchor="top",
                          buttons=buttons, showactive=True, active=0)],
        annotations=[dict(text="Color by:", x=0.0, xref="paper",
                          xanchor="right", y=1.11, yref="paper",
                          yanchor="top", showarrow=False, font=dict(size=12))],
    )

    # Build event data for JS stats
    event_data = {"event_id": event_ids.tolist()}
    for feat in STAT_FEATURES:
        vals = band_df[feat].values
        event_data[feat] = [None if np.isnan(v) else round(float(v), 4)
                            for v in vals]

    # Use post_script so JS runs inside Plotly's own scope with `gd` available
    ps = build_post_script(band, STAT_FEATURES, event_data)

    plotly_div = fig.to_html(
        full_html=False,
        include_plotlyjs=True,
        div_id=PLOT_DIV_ID,
        config={"scrollZoom": True},
        post_script=[ps],
    )

    full_html = PAGE_TEMPLATE.format(
        band_upper=band.upper(),
        plotly_div=plotly_div,
    )

    outpath.write_text(full_html)
    print(f"  Saved: {outpath} ({outpath.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate interactive UMAP HTML files")
    parser.add_argument("--band", type=str, default=None, choices=BANDS,
                        help="Single band (default: all)")
    args = parser.parse_args()

    bands = [args.band] if args.band else BANDS

    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df):,} events")

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    for band in bands:
        n = (df["detection_band"] == band).sum()
        print(f"\n{band.upper()} band ({n:,} events)...")
        outpath = FIG_DIR / f"umap_{band}_interactive.html"
        make_band_html(df, band, outpath)

    print("\nDone. Open the HTML files in your browser.")
    print("Lasso-select -> click Characterize -> name -> copy -> paste here.")


if __name__ == "__main__":
    main()
