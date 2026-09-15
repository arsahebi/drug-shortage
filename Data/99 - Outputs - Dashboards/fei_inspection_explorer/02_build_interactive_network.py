"""
02_build_interactive_network.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PURPOSE
  Builds a standalone interactive HTML network visualization of cross-FEI
  regulatory relationships using pyvis + NetworkX.

  Each NODE is an FEI (facility):
    Size  → regulatory severity score (larger = more / worse events)
    Color → worst regulatory outcome (crimson = WL, red = OAI, amber = VAI, etc.)
    Shape → star if Warning Letter, diamond if OAI, circle otherwise
    Hover → full event timeline + summary stats popup

  Each EDGE is a cross-FEI relationship:
    Red thick  → Warning Letter cross-site mention or repeat multi-site link
    Gray dashed → same parent company (inferred from legal name)

WHEN TO RUN
  Optional — run after 01_build_combined_dataset.py.
  Produces a separate standalone network file; not required by 03 or the LLM pipeline.
  Useful for exploring facility relationships outside the main dashboard.

REQUIRED FOR COMBINED DATASET?  NO — standalone visualization only.

INPUTS (from this folder, produced by 01)
  fei_events_timeline.csv
  fei_node_summary.csv
  fei_edge_list.csv

OUTPUTS
  fei_network.html  — open in Chrome / Firefox / Safari (no server needed)

DEPENDENCIES
  pip install pyvis networkx
  (auto-installs if missing)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

try:
    from pyvis.network import Network
    import networkx as nx
    HAS_PYVIS = True
except ImportError:
    HAS_PYVIS = False
    print("pyvis not found. Installing...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyvis", "networkx", "-q"])
    from pyvis.network import Network
    import networkx as nx
    HAS_PYVIS = True

# ── Paths ──────────────────────────────────────────────────────────────────
OUT       = Path(__file__).parent
EVENTS    = OUT / "fei_events_timeline.csv"
NODES     = OUT / "fei_node_summary.csv"
EDGES_F   = OUT / "fei_edge_list.csv"
HTML_OUT  = OUT / "fei_network.html"

# ── Colour palette ──────────────────────────────────────────────────────────
NODE_COLORS = {
    "Warning Letter":       "#8B0000",   # dark crimson
    "OAI":                  "#C0392B",   # bright red
    "Class I Recall":       "#9B59B6",   # purple
    "VAI":                  "#E67E22",   # amber
    "NAI":                  "#27AE60",   # green
    "Import Refusal Only":  "#007A87",   # teal
    "No Regulatory Events": "#BDC3C7",   # gray
}
EDGE_COLORS = {
    "wl_cross_site":    "#8B0000",
    "wl_repeat_multi":  "#C0392B",
    "same_company":     "#BDC3C7",
}

EVENT_COLORS = {
    "Warning Letter":   "#8B0000",
    "Inspection":       {"OAI": "#C0392B", "VAI": "#E67E22", "NAI": "#27AE60"},
    "483":              "#E6AF22",
    "Recall":           {"Class I": "#8B008B", "Class II": "#9B59B6", "Class III": "#BDC3C7"},
    "Import Refusal":   "#007A87",
}

# ══════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════
print("Loading combined dataset...")
events_df = pd.read_csv(EVENTS, parse_dates=["event_date"])
nodes_df  = pd.read_csv(NODES)
edges_df  = pd.read_csv(EDGES_F)

nodes_df["fei"] = nodes_df["fei"].astype(int)
events_df["fei"] = events_df["fei"].astype(int)

print(f"  Nodes: {len(nodes_df)} FEIs")
print(f"  Events: {len(events_df)} total")
print(f"  Edges: {len(edges_df)} cross-FEI links")


# ══════════════════════════════════════════════════════════════════════════
# 2. BUILD TOOLTIP HTML FOR EACH FEI
# ══════════════════════════════════════════════════════════════════════════
def event_row_color(row):
    etype = row["event_type"]
    sub   = str(row.get("event_subtype", ""))
    if etype == "Inspection":
        if "OAI" in sub: return "#C0392B"
        if "VAI" in sub: return "#E67E22"
        return "#27AE60"
    if etype == "Warning Letter": return "#8B0000"
    if etype == "483":            return "#E6AF22"
    if etype == "Recall":
        if "Class I" in sub:  return "#8B008B"
        if "Class II" in sub: return "#9B59B6"
        return "#BDC3C7"
    if etype == "Import Refusal": return "#007A87"
    return "#555555"


def make_tooltip(node_row, fei_events):
    fei     = int(node_row["fei"])
    firm    = str(node_row["firm_name"])[:55]
    country = str(node_row["country"])
    worst   = str(node_row["worst_outcome"])

    # Summary badges
    badges = []
    if node_row["n_warning_letters"] > 0:
        badges.append(f'<span style="background:#8B0000;color:white;padding:1px 5px;border-radius:3px;font-size:10px">WL×{int(node_row["n_warning_letters"])}</span>')
    if node_row["n_oai"] > 0:
        badges.append(f'<span style="background:#C0392B;color:white;padding:1px 5px;border-radius:3px;font-size:10px">OAI×{int(node_row["n_oai"])}</span>')
    if node_row["n_class_I_recalls"] > 0:
        badges.append(f'<span style="background:#8B008B;color:white;padding:1px 5px;border-radius:3px;font-size:10px">RecallI×{int(node_row["n_class_I_recalls"])}</span>')
    if node_row["n_vai"] > 0:
        badges.append(f'<span style="background:#E67E22;color:white;padding:1px 5px;border-radius:3px;font-size:10px">VAI×{int(node_row["n_vai"])}</span>')
    if node_row["n_import_refusals"] > 0:
        badges.append(f'<span style="background:#007A87;color:white;padding:1px 5px;border-radius:3px;font-size:10px">Refusal×{int(min(node_row["n_import_refusals"], 999))}</span>')

    # Stats table
    stats = [
        ("Inspections", f"{int(node_row['n_inspections'])} (OAI:{int(node_row['n_oai'])} VAI:{int(node_row['n_vai'])} NAI:{int(node_row['n_nai'])})"),
        ("Form 483s",   str(int(node_row["n_483s"]))),
        ("Warning Letters", str(int(node_row["n_warning_letters"]))),
        ("Recalls",     f"{int(node_row['n_recalls'])} (Class I: {int(node_row['n_class_I_recalls'])})"),
        ("Import Refusals", str(int(node_row["n_import_refusals"]))),
        ("First Event", str(node_row["first_event_date"])[:10]),
        ("Last Event",  str(node_row["last_event_date"])[:10]),
    ]

    # Timeline — show up to 30 most recent events
    recent = fei_events.sort_values("event_date", ascending=False).head(30)
    timeline_rows = ""
    for _, ev in recent.iterrows():
        clr = event_row_color(ev)
        date_str = str(ev["event_date"])[:10]
        etype    = str(ev["event_type"])
        sub      = str(ev["event_subtype"])[:20]
        details  = str(ev["key_details"])[:90]
        timeline_rows += (
            f'<tr style="border-bottom:1px solid #eee">'
            f'<td style="padding:2px 5px;color:#666;white-space:nowrap">{date_str}</td>'
            f'<td style="padding:2px 5px;color:{clr};font-weight:bold;white-space:nowrap">{etype}</td>'
            f'<td style="padding:2px 5px;color:{clr};white-space:nowrap">{sub}</td>'
            f'<td style="padding:2px 5px;color:#444;font-size:10px">{details}</td>'
            f'</tr>'
        )

    n_total = len(fei_events)
    show_note = f'<div style="color:#999;font-size:9px;margin-top:3px">Showing {min(30, n_total)} of {n_total} events (most recent first)</div>' if n_total > 30 else ""

    html = f"""
<div style="font-family:Arial,sans-serif;width:550px;background:white;border-radius:6px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.2)">
  <div style="background:#1F3564;padding:10px 14px">
    <div style="color:white;font-size:14px;font-weight:bold">FEI {fei}</div>
    <div style="color:#BDC3C7;font-size:12px">{firm} &nbsp;·&nbsp; {country}</div>
    <div style="margin-top:5px">{'&nbsp;'.join(badges)}</div>
  </div>
  <div style="padding:8px 14px;background:#f8f9fa">
    <table style="font-size:11px;border-collapse:collapse;width:100%">
      {''.join(f"<tr><td style='padding:2px 6px;color:#888;width:120px'>{k}</td><td style='padding:2px 6px;font-weight:bold'>{v}</td></tr>" for k, v in stats)}
    </table>
  </div>
  <div style="padding:4px 14px 8px 14px">
    <div style="font-size:11px;font-weight:bold;color:#1F3564;margin-bottom:4px">Event Timeline</div>
    <div style="max-height:220px;overflow-y:auto">
    <table style="font-size:11px;border-collapse:collapse;width:100%">
      <thead>
        <tr style="background:#E8ECF0">
          <th style="padding:3px 5px;text-align:left">Date</th>
          <th style="padding:3px 5px;text-align:left">Type</th>
          <th style="padding:3px 5px;text-align:left">Sub-type</th>
          <th style="padding:3px 5px;text-align:left">Details</th>
        </tr>
      </thead>
      <tbody>{timeline_rows}</tbody>
    </table>
    </div>
    {show_note}
  </div>
</div>"""
    return html


# ══════════════════════════════════════════════════════════════════════════
# 3. BUILD NETWORKX GRAPH
# ══════════════════════════════════════════════════════════════════════════
print("Building graph...")
G = nx.Graph()

# Node size range
sev_scores = nodes_df["severity_score"].fillna(0)
min_sev = sev_scores.min()
max_sev = max(sev_scores.max(), 1)

def scale_size(sev, lo=18, hi=65):
    return lo + (sev - min_sev) / (max_sev - min_sev) * (hi - lo)

for _, node_row in nodes_df.iterrows():
    fei  = int(node_row["fei"])
    firm = str(node_row["firm_name"])
    worst = str(node_row["worst_outcome"])
    color = NODE_COLORS.get(worst, "#BDC3C7")
    size  = scale_size(float(node_row["severity_score"]))
    shape = "star" if node_row["has_wl"] else ("diamond" if node_row["has_oai"] else "dot")

    # Short label: abbreviated firm name + country code
    country = str(node_row["country"])
    country_abbr = {"India": "IN", "United States": "US", "China": "CN",
                    "Canada": "CA", "Germany": "DE", "Italy": "IT",
                    "Hungary": "HU", "Bangladesh": "BD", "Israel": "IL",
                    "Austria": "AT", "Japan": "JP", "Ireland": "IE",
                    "Slovenia": "SI"}.get(country, country[:2].upper())

    # Abbreviated firm name (first word or first 12 chars)
    first_word = firm.split()[0] if firm else str(fei)
    label = f"{first_word[:14]}\n({country_abbr})"

    # Build tooltip
    fei_events = events_df[events_df["fei"] == fei]
    tooltip    = make_tooltip(node_row, fei_events)

    G.add_node(
        fei,
        label=label,
        title=tooltip,
        color=color,
        size=size,
        shape=shape,
        borderWidth=2,
        borderWidthSelected=4,
    )

# Add edges
for _, edge_row in edges_df.iterrows():
    fei_a = int(edge_row["fei_a"])
    fei_b = int(edge_row["fei_b"])
    if not G.has_node(fei_a) or not G.has_node(fei_b):
        continue
    etype  = str(edge_row["edge_type"])
    color  = EDGE_COLORS.get(etype, "#BDC3C7")
    width  = 4 if "wl" in etype else 1.5
    dashes = (etype == "same_company")
    label  = str(edge_row.get("label", ""))
    desc   = str(edge_row.get("description", ""))

    G.add_edge(
        fei_a, fei_b,
        color=color,
        width=width,
        dashes=dashes,
        label=label if "wl" in etype else "",
        title=f'<div style="font-family:Arial;font-size:11px;padding:4px 8px">'
              f'<b style="color:{color}">{etype.replace("_"," ").title()}</b><br>{desc}</div>',
        arrows="",
    )

print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


# ══════════════════════════════════════════════════════════════════════════
# 4. RENDER WITH PYVIS
# ══════════════════════════════════════════════════════════════════════════
print("Rendering interactive HTML...")

net = Network(
    height="900px",
    width="100%",
    bgcolor="#F0F2F6",
    font_color="#1A1A2E",
    notebook=False,
    directed=False,
)

# Transfer graph
net.from_nx(G)

# Physics configuration — Barnes-Hut for good spread with 129 nodes
net.set_options("""
{
  "nodes": {
    "font": {"size": 11, "face": "Arial", "strokeWidth": 2, "strokeColor": "#ffffff"},
    "shadow": {"enabled": true, "size": 6, "x": 2, "y": 2, "color": "rgba(0,0,0,0.2)"}
  },
  "edges": {
    "smooth": {"type": "continuous"},
    "font": {"size": 9, "align": "middle"},
    "shadow": false
  },
  "physics": {
    "enabled": true,
    "barnesHut": {
      "gravitationalConstant": -8000,
      "centralGravity": 0.25,
      "springLength": 180,
      "springConstant": 0.03,
      "damping": 0.09,
      "avoidOverlap": 0.6
    },
    "maxVelocity": 60,
    "minVelocity": 0.75,
    "stabilization": {"iterations": 250, "updateInterval": 10}
  },
  "interaction": {
    "hover": true,
    "tooltipDelay": 100,
    "hideEdgesOnDrag": true,
    "navigationButtons": true,
    "keyboard": {"enabled": true}
  }
}
""")

# ══════════════════════════════════════════════════════════════════════════
# 5. INJECT CUSTOM HTML — legend, title bar, filter controls
# ══════════════════════════════════════════════════════════════════════════
net.save_graph(str(HTML_OUT))

# Read the saved HTML and inject our custom header + legend
with open(HTML_OUT, "r", encoding="utf-8") as f:
    html_content = f.read()

LEGEND_HTML = """
<div id="custom-header" style="
    position:fixed; top:0; left:0; right:0; z-index:1000;
    background:linear-gradient(135deg,#1F3564,#007A87);
    color:white; padding:10px 20px;
    display:flex; align-items:center; justify-content:space-between;
    box-shadow:0 2px 8px rgba(0,0,0,0.3); font-family:Arial,sans-serif">
  <div>
    <div style="font-size:18px;font-weight:bold">FDA Facility Cross-Site Regulatory Network</div>
    <div style="font-size:11px;opacity:0.85;margin-top:2px">
      129 FEIs · 14 APIs · Sources: Inspections, 483s, Warning Letters, Recalls, Import Refusals
    </div>
  </div>
  <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center">
    <!-- Node legend -->
    <div style="font-size:10px;opacity:0.7;margin-right:4px">NODES:</div>
    <span style="background:#8B0000;padding:3px 8px;border-radius:12px;font-size:11px">★ Warning Letter</span>
    <span style="background:#C0392B;padding:3px 8px;border-radius:12px;font-size:11px">◆ OAI</span>
    <span style="background:#E67E22;padding:3px 8px;border-radius:12px;font-size:11px">● VAI</span>
    <span style="background:#27AE60;padding:3px 8px;border-radius:12px;font-size:11px">● NAI</span>
    <span style="background:#007A87;padding:3px 8px;border-radius:12px;font-size:11px">● Refusal</span>
    <span style="background:#BDC3C7;color:#333;padding:3px 8px;border-radius:12px;font-size:11px">● No Data</span>
    <div style="font-size:10px;opacity:0.7;margin-left:8px;margin-right:4px">EDGES:</div>
    <span style="border-bottom:3px solid #C0392B;padding:0 8px;font-size:11px">— WL Cross-Site</span>
    <span style="border-bottom:2px dashed #BDC3C7;color:#BDC3C7;padding:0 8px;font-size:11px">— Same Company</span>
  </div>
</div>
<div style="height:55px"></div>
"""

# Inject legend after <body>
html_content = html_content.replace("<body>", "<body>\n" + LEGEND_HTML, 1)

# Also add a note about node size
SIZE_NOTE = """
<div style="
    position:fixed; bottom:12px; left:12px; z-index:999;
    background:rgba(31,53,100,0.85); color:white;
    border-radius:6px; padding:6px 12px; font-family:Arial;font-size:11px">
  <b>Node size</b> = regulatory severity score<br>
  <b>Hover</b> = full event timeline &nbsp;·&nbsp;
  <b>Drag</b> = rearrange &nbsp;·&nbsp;
  <b>Scroll</b> = zoom
</div>
"""
html_content = html_content.replace("</body>", SIZE_NOTE + "\n</body>", 1)

with open(HTML_OUT, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"\n✓ Saved: {HTML_OUT}")
print("  Open in Chrome / Safari / Firefox for full interactivity.")
print(f"\nNetwork summary:")
print(f"  Total nodes (FEIs):     {G.number_of_nodes()}")
print(f"  Connected nodes:        {sum(1 for n in G.nodes if G.degree(n) > 0)}")
print(f"  Isolated nodes:         {sum(1 for n in G.nodes if G.degree(n) == 0)}")
print(f"  WL cross-site edges:    {sum(1 for _,_,d in G.edges(data=True) if 'wl' in str(d.get('title','')).lower())}")
print(f"  Same-company edges:     {sum(1 for _,_,d in G.edges(data=True) if 'same' in str(d.get('title','')).lower())}")
