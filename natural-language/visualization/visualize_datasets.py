"""Generate HTML visualizations of paraphrase and prefix-paraphrase datasets
with color-coded prefix/suffix boundaries."""

import hashlib, json, html, os

from collections import defaultdict
from datasets import load_from_disk
from transformers import AutoTokenizer

# ── Config ──────────────────────────────────────────────────────────────────
SEQ_LEN = 100
PREFIX_TOKENS = 50
SUFFIX_TOKENS = 50
OPENAI_MODEL = "gpt-5-mini-2025-08-07"
NUM_PARAPHRASES = 10

BASE = os.path.join(os.path.dirname(__file__), "..")
PARAPHRASE_PATH        = os.path.join(BASE, "wikitext_paraphrase")
PREFIX_PARAPHRASE_PATH = os.path.join(BASE, "wikitext_prefix_paraphrase")
CACHE_PATH             = os.path.join(BASE, "paraphrase_cache.json")
OUTPUT_HTML            = os.path.join(os.path.dirname(__file__), "datasets_visualization.html")

tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)


# ── Helpers ─────────────────────────────────────────────────────────────────
def color_span(text, cls):
    return f'<span class="{cls}">{html.escape(text)}</span>'


# ── Load datasets ───────────────────────────────────────────────────────────
ds_para   = load_from_disk(PARAPHRASE_PATH)
ds_prefix = load_from_disk(PREFIX_PARAPHRASE_PATH)

with open(CACHE_PATH) as f:
    cache = json.load(f)


# ── Group rows by source_idx ───────────────────────────────────────────────
def group_by_source(ds):
    groups = defaultdict(list)
    for row in ds:
        groups[row["source_idx"]].append(row)
    for g in groups.values():
        g.sort(key=lambda r: r["paraphrase_idx"])
    return dict(sorted(groups.items()))

para_groups   = group_by_source(ds_para)
prefix_groups = group_by_source(ds_prefix)


# ── Prefix boundary is always at token 50 ──────────────────────────────────


# ── HTML generation ─────────────────────────────────────────────────────────
def render_paraphrase_group(source_idx, rows):
    """Render a paraphrase dataset group (entire sequence is the paraphrase)."""
    parts = []
    parts.append(f'<div class="group">')
    parts.append(f'<h3>Source {source_idx}</h3>')
    for row in rows:
        pid = row["paraphrase_idx"]
        label = "original" if pid == 0 else f"paraphrase {pid}"
        text = tokenizer.decode(row["input_ids"])
        parts.append(
            f'<div class="row">'
            f'<span class="label">[{label}]</span> '
            f'{color_span(text, "full-para")}'
            f'</div>'
        )
    parts.append('</div>')
    return "\n".join(parts)


def render_prefix_paraphrase_group(source_idx, rows):
    """Render a prefix-paraphrase dataset group with prefix/suffix coloring."""
    parts = []
    parts.append(f'<div class="group">')
    parts.append(f'<h3>Source {source_idx}</h3>')
    for row in rows:
        pid = row["paraphrase_idx"]
        label = "original" if pid == 0 else f"paraphrase {pid}"
        ids = row["input_ids"]
        prefix_ids = ids[:PREFIX_TOKENS]
        suffix_ids = ids[PREFIX_TOKENS:]
        prefix_text = tokenizer.decode(prefix_ids)
        suffix_text = tokenizer.decode(suffix_ids)
        parts.append(
            f'<div class="row">'
            f'<span class="label">[{label}]</span> '
            f'{color_span(prefix_text, "prefix")}'
            f'<span class="token-count">[{len(prefix_ids)}t]</span>'
            f'{color_span(suffix_text, "suffix")}'
            f'<span class="token-count">[{len(suffix_ids)}t]</span>'
            f'</div>'
        )
    parts.append('</div>')
    return "\n".join(parts)


# ── Assemble HTML ───────────────────────────────────────────────────────────
para_html = "\n".join(
    render_paraphrase_group(sid, rows)
    for sid, rows in para_groups.items()
)
prefix_html = "\n".join(
    render_prefix_paraphrase_group(sid, rows)
    for sid, rows in prefix_groups.items()
)

full_html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Dataset Visualization: Paraphrase vs Prefix-Paraphrase</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: "SF Mono", "Cascadia Code", "Fira Code", monospace;
    font-size: 13px;
    line-height: 1.6;
    background: #0d1117;
    color: #c9d1d9;
    padding: 24px;
  }}
  h1 {{
    font-size: 22px;
    color: #f0f6fc;
    margin-bottom: 8px;
  }}
  h2 {{
    font-size: 17px;
    color: #8b949e;
    margin: 32px 0 12px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #21262d;
  }}
  h3 {{
    font-size: 14px;
    color: #58a6ff;
    margin: 16px 0 6px 0;
  }}
  .legend {{
    margin: 12px 0 24px 0;
    padding: 12px 16px;
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 6px;
    display: inline-flex;
    gap: 24px;
    font-size: 13px;
  }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; }}
  .legend-swatch {{
    display: inline-block;
    width: 14px; height: 14px;
    border-radius: 3px;
  }}
  .tab-bar {{
    display: flex;
    gap: 0;
    margin-bottom: 0;
  }}
  .tab {{
    padding: 10px 24px;
    cursor: pointer;
    background: #161b22;
    color: #8b949e;
    border: 1px solid #21262d;
    border-bottom: none;
    border-radius: 6px 6px 0 0;
    font-family: inherit;
    font-size: 14px;
    font-weight: 600;
  }}
  .tab.active {{
    background: #0d1117;
    color: #f0f6fc;
    border-bottom: 2px solid #58a6ff;
  }}
  .tab-content {{
    display: none;
    border: 1px solid #21262d;
    border-radius: 0 6px 6px 6px;
    padding: 20px;
    background: #0d1117;
  }}
  .tab-content.active {{ display: block; }}
  .group {{
    margin-bottom: 20px;
    padding: 12px 16px;
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 6px;
  }}
  .row {{
    margin: 4px 0;
    padding: 4px 0;
    word-wrap: break-word;
  }}
  .label {{
    color: #8b949e;
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  .full-para {{
    color: #c9d1d9;
  }}
  .prefix {{
    background: rgba(56, 139, 253, 0.15);
    color: #79c0ff;
    border-radius: 2px;
    padding: 1px 0;
  }}
  .suffix {{
    background: rgba(238, 159, 79, 0.15);
    color: #e0a46e;
    border-radius: 2px;
    padding: 1px 0;
  }}
  .stats {{
    color: #8b949e;
    font-size: 12px;
    margin-bottom: 16px;
  }}
  .token-count {{
    color: #6e7681;
    font-size: 10px;
    margin: 0 4px;
  }}
</style>
</head>
<body>

<h1>Dataset Visualization</h1>
<div class="legend">
  <div class="legend-item">
    <span class="legend-swatch" style="background:rgba(56,139,253,0.35)"></span>
    Prefix (paraphrased)
  </div>
  <div class="legend-item">
    <span class="legend-swatch" style="background:rgba(238,159,79,0.35)"></span>
    Suffix (from separate source)
  </div>
  <div class="legend-item">
    <span class="legend-swatch" style="background:#21262d"></span>
    Full paraphrase (no split)
  </div>
</div>

<div class="tab-bar">
  <div class="tab active" onclick="switchTab(0)">Prefix-Paraphrase Dataset</div>
  <div class="tab" onclick="switchTab(1)">Paraphrase Dataset</div>
</div>

<div class="tab-content active" id="tab-0">
  <h2>Prefix-Paraphrase Dataset</h2>
  <div class="stats">{len(ds_prefix)} rows &middot; {len(prefix_groups)} source sequences &middot; {NUM_PARAPHRASES} variants each &middot; {PREFIX_TOKENS} prefix tokens + {SUFFIX_TOKENS} suffix tokens &middot; prefix shown in <span class="prefix">blue</span>, suffix in <span class="suffix">orange</span></div>
  {prefix_html}
</div>

<div class="tab-content" id="tab-1">
  <h2>Paraphrase Dataset</h2>
  <div class="stats">{len(ds_para)} rows &middot; {len(para_groups)} source sequences &middot; {NUM_PARAPHRASES} variants each &middot; entire sequence is paraphrased</div>
  {para_html}
</div>

<script>
function switchTab(idx) {{
  document.querySelectorAll('.tab').forEach((t, i) => {{
    t.classList.toggle('active', i === idx);
  }});
  document.querySelectorAll('.tab-content').forEach((c, i) => {{
    c.classList.toggle('active', i === idx);
  }});
}}
</script>

</body>
</html>
"""

with open(OUTPUT_HTML, "w") as f:
    f.write(full_html)

print(f"Wrote {OUTPUT_HTML}")
print(f"  Paraphrase dataset:        {len(ds_para)} rows, {len(para_groups)} groups")
print(f"  Prefix-paraphrase dataset: {len(ds_prefix)} rows, {len(prefix_groups)} groups")
