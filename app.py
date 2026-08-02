"""
app.py
======
Streamlit front-end for the CVD palette optimiser.

Run with:
    streamlit run app.py

Expects cvd_module.py, fitness.py, ga.py and fastmath.py to sit next to this
file. The algorithm itself is untouched — this file only drives it.

Everything technical (thresholds, fidelity budget, GA settings) lives in the
TUNING block below and never reaches the screen. Change it here.
"""

import threading
import time

import numpy as np
import streamlit as st

import fastmath
from cvd_module import lab_to_srgb, simulate_palette_lab, srgb_to_lab
from fitness import find_problematic_pairs
import ga as ga_module

# Same CIEDE2000 numbers, fed to colour-science in batches instead of one
# pair at a time. Roughly halves GA runtime; see fastmath.py.
fastmath.enable()


# ═════════════════════════════════════════════════════════════
# TUNING — developer settings. Not exposed in the UI.
# ═════════════════════════════════════════════════════════════
CONFLICT_THRESHOLD = 10.0   # ΔE below which two colours count as confusable
FIDELITY_TAU = 12.0         # max ΔE any one colour may drift from the user's pick
CVD_TYPES = ["protan", "deutan", "tritan"]
POP_SIZE = 60
N_GENERATIONS = 60
RANDOM_SEED = 7             # None for a different result each run
MAX_COLOURS = 5

# Deliberately conflicting starter palette (red-yellow-green ramp)
DEFAULT_HEX = ["#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#91cf60"]


# ─────────────────────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Palette check", page_icon="◼", layout="wide")

CSS = """
<style>
  .stApp { background: #FFFFFF; }
  html, body, [class*="css"] { color: #000000; }
  header[data-testid="stHeader"] { background: transparent; }
  .block-container { padding-top: 2.2rem; padding-bottom: 5rem; max-width: 1120px; }
  h1, h2, h3, h4, p, li, label, span, div { color: #000000; }

  .masthead { border-bottom: 2px solid #000; padding-bottom: 0.7rem; margin-bottom: 1.7rem; }
  .masthead .title { font-size: 2.15rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1.05; margin: 0; }
  .masthead .sub { font-size: 0.95rem; margin-top: 0.4rem; }

  .eyebrow {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.7rem; letter-spacing: 0.16em; text-transform: uppercase;
    border-bottom: 1px solid #000; padding-bottom: 0.35rem; margin: 2.3rem 0 1.1rem 0;
  }

  /* ── Simulation grid ───────────────────────────────── */
  .grid { border-collapse: collapse; width: 100%; table-layout: fixed; }
  .grid th, .grid td { border: 0; padding: 0; }
  .grid thead th {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase;
    text-align: left; padding: 0 0 0.5rem 0; border-bottom: 1px solid #000;
  }
  .grid tbody th {
    font-size: 0.86rem; text-align: left; vertical-align: middle;
    width: 190px; padding-right: 16px; line-height: 1.3;
  }
  .grid tbody th .vsub {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.66rem; letter-spacing: 0.06em; opacity: 0.55; display: block;
  }
  .grid tbody td { padding: 11px 6px 15px 0; vertical-align: top; }
  .grid tbody tr + tr th, .grid tbody tr + tr td { border-top: 1px solid #E0E0E0; }

  .sw { height: 58px; border: 1px solid #000; width: 100%; }
  .sw.flag { box-shadow: inset 0 0 0 3px #FFF, inset 0 0 0 4px #000; }
  .cap {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.68rem; letter-spacing: 0.02em; padding-top: 5px;
  }
  .warn { font-size: 0.68rem; padding-top: 3px; font-weight: 600; line-height: 1.3; }

  /* ── Verdict banners ───────────────────────────────── */
  .verdict { border: 2px solid #000; padding: 1rem 1.2rem; margin: 0.3rem 0; }
  .verdict.bad { background: #000; }
  .verdict.bad .vtitle, .verdict.bad .vbody { color: #FFF; }
  .vtitle {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.74rem; letter-spacing: 0.14em; text-transform: uppercase; font-weight: 700;
  }
  .vbody { font-size: 0.97rem; margin-top: 0.35rem; line-height: 1.45; }

  /* ── Conflict list ─────────────────────────────────── */
  .conflict {
    display: flex; align-items: center; gap: 14px;
    padding: 11px 0; border-bottom: 1px solid #E0E0E0;
  }
  .chip { width: 32px; height: 32px; border: 1px solid #000; flex: none; }
  .chip.pair { margin-left: -14px; }
  .ctext { font-size: 0.92rem; }
  .ctag {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase;
    margin-left: auto; border: 1px solid #000; padding: 3px 9px; white-space: nowrap;
  }
  .ctag.hard { background: #000; color: #FFF; }

  /* ── Buttons ───────────────────────────────────────── */
  .stButton > button {
    background: #000; color: #FFF; border: 2px solid #000; border-radius: 0;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.78rem; letter-spacing: 0.12em; text-transform: uppercase;
    padding: 0.6rem 1.5rem; font-weight: 600;
  }
  .stButton > button:hover { background: #FFF; color: #000; border-color: #000; }
  .stButton > button:focus { color: #FFF; border-color: #000; box-shadow: none; }
  .stButton > button:focus:hover { color: #000; }

  .stProgress > div > div > div > div { background-color: #000; }
  [data-testid="stColorPicker"] label p {
    font-family: ui-monospace, monospace; font-size: 0.7rem; letter-spacing: 0.08em;
  }
  div[role="radiogroup"] label p { font-size: 0.9rem; }
  div[data-testid="stSidebar"] { display: none; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Labels
# ─────────────────────────────────────────────────────────────
VISION_ROWS = [
    ("Normal vision", "", None),
    ("Protanopia", "reduced red", "protan"),
    ("Deuteranopia", "reduced green", "deutan"),
    ("Tritanopia", "reduced blue", "tritan"),
]
CVD_NAME = {"protan": "protanopia", "deutan": "deuteranopia", "tritan": "tritanopia"}


# ─────────────────────────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────────────────────────
def hex_to_rgb01(hex_str):
    h = hex_str.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def rgb01_to_hex(rgb):
    r, g, b = (int(round(float(np.clip(c, 0, 1)) * 255)) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def lab_to_hex(lab):
    return rgb01_to_hex(lab_to_srgb(np.atleast_2d(lab))[0])


def name(idx):
    """1-based, human-facing colour label."""
    return f"Colour {idx + 1}"


def severity(delta):
    """Plain-language stand-in for the raw distance."""
    ratio = delta / CONFLICT_THRESHOLD
    if ratio < 0.3:
        return "Nearly identical", True
    if ratio < 0.65:
        return "Very similar", True
    return "Easily confused", False


# ─────────────────────────────────────────────────────────────
# Cached analysis
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def palette_lab_from_hex(hex_list):
    return srgb_to_lab(np.array([hex_to_rgb01(h) for h in hex_list]))


@st.cache_data(show_spinner=False)
def simulated_hexes(palette_lab, cvd):
    labs = palette_lab if cvd is None else simulate_palette_lab(palette_lab, cvd_type=cvd)
    return [lab_to_hex(l) for l in labs]


@st.cache_data(show_spinner=False)
def analyse(palette_lab):
    _, details = find_problematic_pairs(
        palette_lab,
        threshold=CONFLICT_THRESHOLD,
        cvd_types=CVD_TYPES,
        return_details=True,
    )
    return sorted(details, key=lambda d: d["deltaE"])


@st.cache_data(show_spinner=False)
def mutable_for(palette_lab):
    return ga_module.get_mutable_indices(
        palette_lab, threshold=CONFLICT_THRESHOLD, cvd_types=tuple(CVD_TYPES)
    )


# ─────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────
def join_names(indices):
    """'Colour 2, Colour 3 and Colour 4'"""
    labels = [name(i) for i in indices]
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + " and " + labels[-1]


def render_grid(palette_lab, details):
    """One column per colour, one row per kind of vision."""
    k = len(palette_lab)

    collisions = {cvd: {} for cvd in CVD_TYPES}
    for d in details:
        i, j = d["pair"]
        collisions[d["cvd"]].setdefault(i, set()).add(j)
        collisions[d["cvd"]].setdefault(j, set()).add(i)

    head = "".join(f"<th>{name(i)}</th>" for i in range(k))
    body = []

    for label, sub, cvd in VISION_ROWS:
        hexes = simulated_hexes(palette_lab, cvd)
        cells = []
        for i, hx in enumerate(hexes):
            partners = collisions.get(cvd, {}).get(i, set()) if cvd else set()
            flag = " flag" if partners else ""
            mark = ""
            if partners:
                joined = join_names(sorted(partners))
                mark = f'<div class="warn">↔ blurs with {joined}</div>'
            cells.append(
                f'<td><div class="sw{flag}" style="background:{hx}"></div>'
                f'<div class="cap">{hx}</div>{mark}</td>'
            )
        sub_html = f'<span class="vsub">{sub}</span>' if sub else ""
        body.append(f'<tr><th>{label}{sub_html}</th>{"".join(cells)}</tr>')

    st.markdown(
        f'<table class="grid"><thead><tr><th></th>{head}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table>',
        unsafe_allow_html=True,
    )


def render_verdict(details, optimised=False):
    if not details:
        body = (
            "Every colour stays distinct for all three kinds of colour blindness."
            if not optimised else
            "Every colour now stays distinct for all three kinds of colour blindness."
        )
        st.markdown(
            f'<div class="verdict"><div class="vtitle">◻ Good to go</div>'
            f'<div class="vbody">{body}</div></div>',
            unsafe_allow_html=True,
        )
        return False

    pairs = {d["pair"] for d in details}
    n = len(pairs)
    st.markdown(
        f'<div class="verdict bad">'
        f'<div class="vtitle">◼ {n} {"pair" if n == 1 else "pairs"} hard to tell apart</div>'
        f'<div class="vbody">Some viewers will see these colours as the same. '
        f'They are listed below, closest first.</div></div>',
        unsafe_allow_html=True,
    )
    return True


def render_conflicts(palette_lab, details):
    hexes = [lab_to_hex(l) for l in palette_lab]

    # One row per pair, listing every kind of vision it affects
    grouped = {}
    for d in details:
        entry = grouped.setdefault(d["pair"], {"worst": d["deltaE"], "cvds": []})
        entry["worst"] = min(entry["worst"], d["deltaE"])
        entry["cvds"].append(CVD_NAME[d["cvd"]])

    rows = []
    for pair, entry in sorted(grouped.items(), key=lambda kv: kv[1]["worst"]):
        i, j = pair
        word, hard = severity(entry["worst"])
        cvds = entry["cvds"]
        which = ", ".join(cvds[:-1]) + " and " + cvds[-1] if len(cvds) > 1 else cvds[0]
        rows.append(
            f'<div class="conflict">'
            f'<div style="display:flex"><div class="chip" style="background:{hexes[i]}"></div>'
            f'<div class="chip pair" style="background:{hexes[j]}"></div></div>'
            f'<div class="ctext"><b>{name(i)}</b> and <b>{name(j)}</b> — with {which}</div>'
            f'<div class="ctag{" hard" if hard else ""}">{word}</div>'
            f'</div>'
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def render_before_after(original_lab, optimized_lab, mutable_indices):
    cols = st.columns(len(original_lab))
    for i, col in enumerate(cols):
        before_hex = lab_to_hex(original_lab[i])
        after_hex = lab_to_hex(optimized_lab[i])
        changed = i in mutable_indices and before_hex != after_hex
        col.markdown(
            f'<div class="cap" style="letter-spacing:.1em;text-transform:uppercase;'
            f'padding-bottom:6px">{name(i)} · {"adjusted" if changed else "unchanged"}</div>'
            f'<div style="display:flex;gap:0">'
            f'<div class="sw" style="background:{before_hex};height:52px"></div>'
            f'<div class="sw" style="background:{after_hex};height:52px;border-left:0"></div>'
            f'</div>'
            f'<div class="cap">{before_hex} → {after_hex}</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────
# GA driver (threaded, so the progress bar can move)
# ─────────────────────────────────────────────────────────────
def optimise(palette_lab, mutable_indices, on_progress):
    counter = {"n": 0}
    total = max(1, POP_SIZE * N_GENERATIONS)
    real_fitness = ga_module.fitness
    memo = {}

    def counting_fitness(candidate, original, **kwargs):
        # Elites survive untouched and crossover often reproduces a parent
        # exactly, so roughly half of every generation is a repeat lookup.
        counter["n"] += 1
        key = np.asarray(candidate, dtype=np.float64).tobytes()
        if key not in memo:
            memo[key] = real_fitness(candidate, original, **kwargs)
        return memo[key]

    ga_module.fitness = counting_fitness
    box = {}

    def worker():
        try:
            if RANDOM_SEED is not None:
                np.random.seed(RANDOM_SEED)
            box["value"] = ga_module.run_ga(
                palette_lab,
                mutable_indices,
                cvd_types=list(CVD_TYPES),
                pop_size=POP_SIZE,
                n_generations=N_GENERATIONS,
                tau=FIDELITY_TAU,
                verbose=False,
            )
        except Exception as exc:          # surfaced to the user below
            box["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    while thread.is_alive():
        on_progress(min(counter["n"] / total, 0.99))
        time.sleep(0.08)
    thread.join()
    ga_module.fitness = real_fitness

    if "error" in box:
        raise box["error"]
    on_progress(1.0)
    return box["value"]


# ═════════════════════════════════════════════════════════════
# Screen
# ═════════════════════════════════════════════════════════════
st.markdown(
    '<div class="masthead"><div class="title">Palette check</div>'
    '<div class="sub">See how your colours look to people with colour blindness, '
    'and fix the ones that clash.</div></div>',
    unsafe_allow_html=True,
)


# ── 1. Pick colours ──────────────────────────────────────────
st.markdown('<div class="eyebrow" style="margin-top:0">Your colours</div>', unsafe_allow_html=True)

n_colours = st.radio(
    "How many colours?",
    options=list(range(2, MAX_COLOURS + 1)),
    index=MAX_COLOURS - 2,
    horizontal=True,
)

pickers = st.columns(MAX_COLOURS)
hex_list = []
for i in range(n_colours):
    with pickers[i]:
        hex_list.append(st.color_picker(name(i), DEFAULT_HEX[i], key=f"c{i}"))

palette_lab = palette_lab_from_hex(tuple(hex_list))

# Any change to the palette drops a previous result
if st.session_state.get("signature") != tuple(hex_list):
    st.session_state.pop("result", None)
    st.session_state["signature"] = tuple(hex_list)


# ── 2. Simulations ───────────────────────────────────────────
st.markdown('<div class="eyebrow">How others see them</div>', unsafe_allow_html=True)

details = analyse(palette_lab)
render_grid(palette_lab, details)


# ── 3. Verdict ───────────────────────────────────────────────
st.markdown('<div class="eyebrow">The verdict</div>', unsafe_allow_html=True)

has_conflicts = render_verdict(details)

if not has_conflicts:
    st.markdown(
        '<div class="cap" style="padding-top:14px;opacity:.6">'
        'Nothing to fix. Change a colour above to check a different set.</div>',
        unsafe_allow_html=True,
    )
else:
    render_conflicts(palette_lab, details)

    mutable_indices = mutable_for(palette_lab)
    keep = [i for i in range(n_colours) if i not in mutable_indices]
    moved = join_names(mutable_indices)
    kept = join_names(keep) if keep else ""

    st.markdown(
        f'<div style="padding-top:16px;font-size:0.95rem">'
        f'Fixing this means nudging <b>{moved}</b>.'
        + (f' {kept} stay exactly as you picked them.' if len(keep) > 1
           else f' {kept} stays exactly as you picked it.' if keep else '')
        + '</div>',
        unsafe_allow_html=True,
    )

    if st.button("Fix my colours"):
        bar = st.progress(0.0, text="Finding colours that stay distinct…")
        try:
            optimized_lab, _ = optimise(
                palette_lab, mutable_indices,
                on_progress=lambda f: bar.progress(f, text="Finding colours that stay distinct…"),
            )
            st.session_state["result"] = {
                "palette": optimized_lab,
                "mutable": mutable_indices,
            }
        except Exception:
            st.session_state.pop("result", None)
            st.error("Something went wrong while adjusting the colours. Please try again.")
        finally:
            bar.empty()

    # ── 4. Result ────────────────────────────────────────────
    result = st.session_state.get("result")
    if result is not None:
        optimized_lab = result["palette"]
        after = analyse(optimized_lab)

        st.markdown('<div class="eyebrow">Your fixed palette</div>', unsafe_allow_html=True)
        render_verdict(after, optimised=True)
        if after:
            render_conflicts(optimized_lab, after)

        st.markdown('<div class="eyebrow">Before and after</div>', unsafe_allow_html=True)
        render_before_after(palette_lab, optimized_lab, result["mutable"])

        st.markdown('<div class="eyebrow">How others see the fixed palette</div>',
                    unsafe_allow_html=True)
        render_grid(optimized_lab, after)

        st.markdown('<div class="eyebrow">Copy your colours</div>', unsafe_allow_html=True)
        st.code("  ".join(lab_to_hex(l) for l in optimized_lab), language=None)