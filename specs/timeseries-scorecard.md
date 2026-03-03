# Time Series Scorecard — NOAA BRAVOSEIS

**Rubric**: `specs/rubrics/timeseries-evaluation-rubric.md`
**Sizing tier**: Paper (Title >= 14pt, Axis/Caption >= 10pt, Tick Labels >= 8pt, Line Weight >= 1pt, Min DPI 300)

**P** = Pass, **F** = Fail, **-** = Not applicable

| Figure | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | Notes |
|--------|---|---|---|---|---|---|---|---|---|----|----|----|----|----|----|----|----|----|----|-------|
| Recording Timeline | P | - | P | P | P | P | P | P | P | P | P | P | P | P | P | P | P | - | P | See notes below |

## Recording Timeline (`recording_timeline.png`)

Scored 2026-03-02.

| # | Criterion | Score | Notes |
|---|-----------|-------|-------|
| 1 | Axis Labels | **P** | X-axis: "Date (UTC)" bold 10pt. Y-axis: no title, but tick labels are self-documenting mooring IDs (standard for Gantt charts). |
| 2 | Dual-Axis Clarity | **-** | Single axis. |
| 3 | Date Formatting | **P** | `MonthLocator` + `%b\n%Y`. Clean, no overlap, 14 ticks across 13-month span. |
| 4 | Classification Legend | **P** | All 6 moorings identified by Okabe-Ito color patch + name. Deploy/recover marker included. Framed, alpha=0.9. |
| 5 | Event Annotation | **P** | Deploy and recovery dates marked with black pipe (`|`) markers at each mooring row. Identified in legend. No per-marker text labels (would crowd 12 markers across 6 rows), but legend entry is clear. |
| 6 | Title | **P** | "BRAVOSEIS Hydrophone Recording Timeline" — 14pt bold sans-serif. |
| 7 | Figure Caption | **P** | Renderer-based justified caption in dedicated axes region. 10pt sans-serif. Bold "Temporary Caption:" prefix. States data format, duty cycle, and total file count. |
| 8 | Temporal Aggregation | **P** | Caption states "one 4-hour DAT file (1000 Hz, 24-bit)". Each bar = one file. |
| 9 | Y-Axis Range | **P** | Constrained to 6 mooring rows with 0.5 padding above/below. |
| 10 | X-Axis Padding | **P** | 5-day padding on each side of the data extent. |
| 11 | Data Gaps | **P** | ~40-hour off periods clearly visible as white space between bars. No interpolation. |
| 12 | Grid | **P** | Vertical grid at month boundaries, alpha=0.3, linewidth=0.5. |
| 13 | Spine Weight | **P** | 1.5pt on all spines (meets >=1pt paper tier). |
| 14 | Line Weight | **P** | Gantt bars (not lines). Bars are filled with alpha=0.85, clearly visible. Deploy/recover markers at markeredgewidth=2.0. |
| 15 | Resolution/Format | **P** | PNG at 300 DPI. |
| 16 | Colorblind Safety | **P** | Okabe-Ito 6-color palette. |
| 17 | Data Provenance | **P** | Caption names instrument type ("autonomous hydrophone moorings"), mooring IDs (BRA28–BRA33), deployment location (Bransfield Strait), and date range (Jan 2019–Feb 2020). Stats box shows per-mooring file counts. |
| 18 | Multi-Panel Labels | **-** | Single panel. |
| 19 | Layout Spacing | **P** | Plot area explicitly positioned `[0.12, 0.35, 0.82, 0.55]`. Caption in dedicated axes below. No clipping. |

**Result: 17/17 applicable criteria pass.**
