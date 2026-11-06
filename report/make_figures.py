"""Figures and summary tables for the project-work report (v1.6 figure numbers; file stems are historical).

Reads only stored results (no training, no new quantity):
  pw01 analysis_001 dimension_candidates.json      (reference lines of Figure 2, fig1_active_units)
  pw02a full_001 au_table.json / au_per_run.json / results.json and raw/*.npz (stored conf_mu / conf_logvar,
        Figure 3, fig4_variance; `converged` flags of the crossed runs in Figures 2 and 6)
  pw03a full_001 results.json and raw/scores_bw-*.npz (stored per-unit x per-fact F1 and sign)
  pw04a descriptive pass on pw02a full_001 results.json (Figure 6, fig5_legality)
  pw05 analysis_001 results.json                     (per-unit fact lists and signs, Figure 5, fig2_unit_anatomy)
  pw05 presentation/q3_pairs.json                   (cross-check of the Figure 5 row labels)
  pw06 full_001 results.json / runs/ / runs_index.json (Figure 4, fig3_encoding)
  pw07 analysis_001 results.json                     (post hoc: KL per active unit, tables.md; legality with the
        derived facts recomputed, the hollow circles of Figure 6 and tables.md)
  pw01 analysis_001 legal_states_866.npy and pw04a descriptive raw/samples_*.npz (Figure 1, fig6_examples)
Writes report/figures/fig1_active_units.{png,pdf} (Figure 2), fig2_unit_anatomy (Figure 5), fig3_encoding (Figure 4),
fig4_variance (Figure 3), fig5_legality (Figure 6), fig6_examples (Figure 1) and tables.md.
Every plotted value is asserted against the JSON it comes from (see the ASSERT comments).

Every figure is drawn at the full 16.4 cm text width and included at width: 100%, so the drawn size is the print
size: ordinary text prints at 8.5 pt, sub- and superscripts at 8.5 x 0.78 = 6.6 pt.

Run from the repository root:  uv run python report/make_figures.py
Figure 1 only (leaves the other files untouched):  uv run python report/make_figures.py --only fig6
"""
import json
import statistics
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib._mathtext
import matplotlib.patheffects
import matplotlib.pyplot as plt
import matplotlib.transforms
from matplotlib.colors import ListedColormap

if __name__ == "__main__" and {"-h", "--help"} & set(sys.argv[1:]):  # usage only
    print(__doc__)
    sys.exit(0)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pwlib import blocksworld as bw

EXP = ROOT / "experiments"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

PW02A = EXP / "pw02a_active_units_amended/results/full_001"
PW03A = EXP / "pw03a_fact_alignment_amended/results/full_001"
PW04D = EXP / "pw04a_sample_legality_amended/descriptive/results/pw02a_full_001/results.json"
PW01 = EXP / "pw01_state_space_ground_truth/results/analysis_001/dimension_candidates.json"
PW05 = EXP / "pw05_unit_anatomy/results/analysis_001/results.json"
PW06 = EXP / "pw06_encoding_and_latent_size/results/full_001"
PW07 = EXP / "pw07_unit_kl_and_matched_legality/results/analysis_001/results.json"

TEXT_WIDTH_IN = 16.4 / 2.54  # A4 with 2.3 cm margins
FIG_W = TEXT_WIDTH_IN  # drawn at the text width and included at width: 100%, so print scale = 1
FS = 8.5  # the one text size of every figure (pt, = print size)

# Okabe-Ito (colour-blind safe); black text and axes; neutral greys for secondary marks only
OI_BLUE, OI_VERMILLION, OI_GREEN, OI_ORANGE = "#0072B2", "#D55E00", "#009E73", "#E69F00"
GREY_REF, GREY_GRID, GREY_SEP = "#7f7f7f", "#d9d9d9", "#7f7f7f"
# "did not converge" cross: Okabe-Ito reddish purple (used by no series). Figure 2 draws it on the run's marker,
# small and with a thin white halo so the marker shape stays visible; Figure 6 draws one above the run (CROSS6)
OI_PURPLE = "#CC79A7"
CROSS = dict(marker="x", s=20, color=OI_PURPLE, lw=0.9,
             path_effects=[matplotlib.patheffects.withStroke(linewidth=1.4, foreground="white")])
CROSS6 = dict(marker="x", s=16, color=OI_PURPLE, lw=0.9)
SEED_S = 15  # seed-marker area (pt^2) of the strip plots

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": FS,
    "axes.labelsize": FS, "axes.titlesize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
    "legend.fontsize": FS, "text.color": "black", "axes.edgecolor": "black", "axes.labelcolor": "black",
    "xtick.color": "black", "ytick.color": "black", "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "legend.frameon": False, "savefig.dpi": 300, "pdf.fonttype": 42,
})
# Sub- and superscripts at 8.5 pt x 0.78 = 6.6 pt in print.
# matplotlib reads this module constant at render time.
matplotlib._mathtext.SHRINK_FACTOR = 0.78


def load(p):
    with open(p) as f:
        return json.load(f)


def panel_title(letter, title):
    """One-line panel title, set left: bold letter, then the title."""
    return r"$\mathbf{" + letter + "}$ " + title


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}")  # no bbox="tight": the saved size is the designed print size
    plt.close(fig)


# ---------------------------------------------------------------- Figure 1 (fig6_examples): input and decoded samples
FIG6_STATE = {"A": "B", "B": "C", "C": "T", "D": "T", "E": "H"}  # positions of the drawn state (looked up below)
FIG6_RUN = "bw-gauss_vae-A2-b1-lr0.0003_s1"  # reference configuration, seed 1
FIG6_SAMPLES = EXP / f"pw04a_sample_legality_amended/descriptive/results/pw02a_full_001/raw/samples_{FIG6_RUN}.npz"


def fig6_examples():
    """(a) one legal state from the pw01 866-state table, drawn, with its true facts; (b) three mode-decoded prior
    samples of the reference configuration (seed 1) that break the domain rules: in file order, the first sample
    that breaks exactly one rule, for each of the first three different rules met (pw01 checker + pw04 taxonomy)."""
    from matplotlib.patches import Rectangle

    from pwlib import legality as LG

    def fmt(f):  # report notation: on(A, B), on_table(C)
        return f.replace(",", ", ")

    # (a) the state must be a row of the stored pw01 table, and the fact order must be pwlib's FACTS
    legal = np.load(EXP / "pw01_state_space_ground_truth/results/analysis_001/legal_states_866.npy")
    assert legal.shape == (866, bw.N_FACTS)
    v = bw.state_from_positions(FIG6_STATE)
    rows = np.flatnonzero((legal == v).all(1))
    assert len(rows) == 1 and bw.is_legal(legal[rows[0]]), rows  # ASSERT: drawn state is one of the 866
    v = legal[rows[0]]
    true_a = [bw.FACTS[j] for j in np.flatnonzero(v)]

    # (b) samples exactly as pw04a stored them: packed mode-decoded bits (mean > 0.5) and V1-V6 flags
    d = np.load(FIG6_SAMPLES)
    S = np.unpackbits(d["prior_mode_bits"], axis=1)[:, :bw.N_FACTS]
    V = LG.violations(S)
    assert (V == np.unpackbits(d["prior_mode_violations"], axis=1)[:, :len(LG.VIOLATIONS)].astype(bool)).all()
    assert (LG.checker_flags(S) == d["prior_mode_legal"]).all() and (LG.checker_flags(S) == ~V.any(1)).all()
    picks, seen = [], set()
    for i in np.flatnonzero(V.sum(1) == 1):
        k = int(V[i].argmax())
        if k not in seen:
            seen.add(k)
            picks.append((int(i), k))
        if len(picks) == 3:
            break
    idx = bw.INDEX

    def explain(s, k):
        """(offending facts present, missing facts, plain-words rule) for the one broken rule k."""
        on = lambda x, y: bool(s[idx[f"on({x},{y})"]])
        held = [x for x in bw.BLOCKS if s[idx[f"holding({x})"]]]
        if LG.VIOLATIONS[k] == "V1_place":
            for x in bw.BLOCKS:
                pl = [f for f in [f"on_table({x})", f"holding({x})"] + [f"on({x},{y})" for y in bw.BLOCKS if y != x]
                      if s[idx[f]]]
                if len(pl) == 0:
                    return [], [f"{x} has no place"], f"{x} is not on the table, on a block or held"
                if len(pl) > 1:
                    return pl, [], f"{x} is in {len(pl)} places at once"
        if LG.VIOLATIONS[k] == "V3_hand":
            if len(held) > 1:
                return [f"holding({x})" for x in held], [], "the hand holds more than one block"
            if held and s[bw.HAND_EMPTY]:
                return [f"holding({held[0]})", "hand_empty"], [], f"the hand is empty but holds {held[0]}"
            # only rule V3 is broken, so every block has exactly one place: hand_empty is the only one-fact repair
            return [], ["hand_empty"], "the hand is neither empty nor holding a block"
        if LG.VIOLATIONS[k] == "V4_clear":
            for y in bw.BLOCKS:
                cov = [x for x in bw.BLOCKS if x != y and on(x, y)]
                exp = int(y not in held and not cov)
                if s[idx[f"clear({y})"]] != exp:
                    if exp:
                        return [], [f"clear({y})"], f"nothing is on {y}, but {y} is not clear"
                    why = [f"on({x},{y})" for x in cov] + ([f"holding({y})"] if y in held else [])
                    what = f"{cov[0]} is on {y}" if cov else f"{y} is held"
                    return [f"clear({y})"] + why, [], f"{y} is clear, but {what}"
        raise ValueError(f"no plain-words label for {LG.VIOLATIONS[k]}")  # fail loudly on an unhandled rule

    # Full text width, 8.5 pt text. (a) drawing then its fact list; (b) three sample columns spread over the rest.
    H = 1.336  # = the v3.3 print height
    fig = plt.figure(figsize=(FIG_W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, H)
    ax.axis("off")
    LS, TOP = 0.13, H - 0.025  # fact line step (in), top of the panel titles
    Y_RULE, Y_FACT = TOP - 0.21, TOP - 0.50  # top of the first rule line / of the first fact line in (b)

    # (a) drawing: stacks built from the on_table / on facts, the held block from holding
    ax.text(0.02, TOP, panel_title("(a)", "A legal state and its true facts"), va="top")
    s_, y0, x0, pitch = 0.21, 0.08, 0.10, 0.32
    above = {y: x for x in bw.BLOCKS for y in bw.BLOCKS if x != y and v[idx[f"on({x},{y})"]]}
    bases = [x for x in bw.BLOCKS if v[idx[f"on_table({x})"]]]
    held = [x for x in bw.BLOCKS if v[idx[f"holding({x})"]]]

    def block(x, xl, yb):
        ax.add_patch(Rectangle((xl, yb), s_, s_, facecolor="#dbe9f6", edgecolor=OI_BLUE, lw=0.9))
        ax.text(xl + s_ / 2, yb + s_ / 2, x, ha="center", va="center")

    for n, b in enumerate(bases):
        xl, yb, cur = x0 + n * pitch, y0, b
        while cur is not None:
            block(cur, xl, yb)
            yb, cur = yb + s_, above.get(cur)
    x_end = x0 + len(bases) * pitch
    if held:
        xl, yb = x_end + 0.06, y0 + 3 * s_ + 0.05
        block(held[0], xl, yb)
        g, top_y = {"color": "black", "lw": 1.0, "solid_capstyle": "butt"}, TOP - 0.2
        cx, yt = xl + s_ / 2, yb + s_ + 0.05
        ax.plot([cx, cx], [yt, top_y], **g)  # arm
        ax.plot([xl - 0.05, xl + s_ + 0.05], [yt, yt], **g)  # palm
        for xf in (xl - 0.05, xl + s_ + 0.05):  # fingers
            ax.plot([xf, xf], [yt, yb + 0.06], **g)
        x_end = xl + s_ + 0.05
    ax.plot([0.02, x_end + 0.06], [y0, y0], color="black", lw=1.0)  # table
    xf = x_end + 0.12
    for n, f in enumerate(true_a):
        ax.text(xf, Y_RULE - n * LS, fmt(f), va="top")
    X_SEP = xf + 0.70
    ax.plot([X_SEP, X_SEP], [0.03, H - 0.03], color=GREY_GRID, lw=0.8)

    # (b) the three samples: rule in black roman, true facts, offending ones in vermillion bold, missing ones after them
    XB, COL2, W_LAST = X_SEP + 0.08, 0.76, 1.43  # (b) left edge, offset of the derived-fact column, last column width
    ax.text(XB, TOP, panel_title("(b)", "Decoded prior samples that break a constraint (2×256, β = 1)"), va="top")
    for c, (i, k) in enumerate(picks):
        s = S[i]
        bad, missing, rule = explain(s, k)
        assert all(s[idx[f]] for f in bad)
        xc = XB + c * (FIG_W - 0.02 - W_LAST - XB) / 2
        ax.text(xc, Y_RULE, textwrap.fill(rule, 29), va="top", linespacing=1.05)
        # two columns: position facts (on_table, on, holding) | derived facts (clear, hand_empty); missing facts
        # go last in the first column, below both lists, where their longer text has room
        cols = [[], []]
        for j in np.flatnonzero(s):
            f = bw.FACTS[j]
            cols[f.startswith("clear(") or f == "hand_empty"].append((fmt(f), f in bad))
        cols[0] += [("", False)] * (max(map(len, cols)) - len(cols[0])) if missing else []
        cols[0] += [(f"{fmt(f)} missing" if f in bw.FACTS else f, True) for f in missing]
        for dx, col in zip((0.0, COL2), cols):
            for n, (t, hi) in enumerate(col):
                ax.text(xc + dx, Y_FACT - n * LS, t, va="top",
                        color=OI_VERMILLION if hi else "black", fontweight="bold" if hi else "normal")
    print("Figure 1 (a) table row", int(rows[0]), true_a)
    for i, k in picks:
        print("Figure 1 (b) sample", i, LG.VIOLATIONS[k], explain(S[i], k), [bw.FACTS[j] for j in np.flatnonzero(S[i])])
    save(fig, "fig6_examples")


if __name__ == "__main__" and sys.argv[1:] == ["--only", "fig6"]:
    fig6_examples()
    sys.exit(0)


au_rows = load(PW02A / "au_table.json")
au_ids = load(PW02A / "au_per_run.json")
pw02 = load(PW02A / "results.json")
pw03 = load(PW03A / "results.json")
pw04 = load(PW04D)
pw05 = load(PW05)["metrics"]["per_run"]
pw05_by_name = {o["name"]: o for o in pw05}
pw07 = load(PW07)  # post hoc: KL per active unit (A) and legality on equal terms (B)
targets = load(PW01)["candidates"]
K_BITS = targets["bits_min_binary_code"]["value"]
K_AFF = targets["affine_dim_legal_states"]["value"]
assert (K_BITS, K_AFF) == (10, 25), (K_BITS, K_AFF)  # ASSERT: reference lines = pw01 candidates

cells = defaultdict(list)
for r in au_rows:
    cells[r["cell_id"]].append(r)
for c in cells:
    cells[c].sort(key=lambda r: r["seed"])


def au(cell, gated_only=False):
    rs = cells[cell]
    return [r["au"] for r in rs if (r["competence"] or not gated_only)]


def comp(cell):
    return [bool(r["competence"]) for r in cells[cell]]


tables = []

# ---------------------------------------------------------------- Figure 2 (fig1_active_units): active units by network
archs = ["A1", "A2", "A4"]
arch_label = {"A1": "1×64", "A2": "2×256", "A4": "2×1024"}  # plain ×, as in the text (no mathtext spacing)
# (table label, legend label, cell template, colour, marker, seeds); all markers filled
series = [
    ("Blocksworld, lr 3e-4", r"Blocksworld, lr $3{\cdot}10^{-4}$", "bw-gauss_vae-{a}-b1-lr0.0003", OI_BLUE, "o", 5),
    ("Blocksworld, lr 3e-3", r"Blocksworld, lr $3{\cdot}10^{-3}$", "bw-gauss_vae-{a}-b1-lr0.003", OI_BLUE, "^", 5),
    ("code10 control", "code10", "code10-gauss_vae-{a}-b1-lr0.0003", OI_VERMILLION, "s", 3),
    ("prod5 control", "prod5", "prod5-gauss_vae-{a}-b1-lr0.0003", OI_GREEN, "D", 3),
]
# Figure 2 values and tables.md "AU at beta=1" (median, min, max over seeds; v1.6 Table 6 5.1 rows), typed here
# so that a drift in either fails loudly
TABLE2 = {
    "bw-gauss_vae-{a}-b1-lr0.0003": {"A1": (16, 15, 16), "A2": (15, 14, 15), "A4": (15, 15, 16)},
    "bw-gauss_vae-{a}-b1-lr0.003": {"A1": (15, 15, 16), "A2": (10, 9, 11), "A4": (15, 14, 15)},
    "code10-gauss_vae-{a}-b1-lr0.0003": {"A1": (10, 10, 10), "A2": (10, 10, 10), "A4": (10, 10, 10)},
    "prod5-gauss_vae-{a}-b1-lr0.0003": {"A1": (16, 15, 17), "A2": (17, 16, 19), "A4": (23, 22, 23)},
}
for _, _, tmpl, _, _, n_seeds in series:
    for a in archs:
        cid = tmpl.format(a=a)
        rs = cells[cid]
        vals = au(cid)
        # ASSERT: seed count, every plotted run passes competence (so no marker encodes a failure),
        # au_table == number of active ids in au_per_run == pw05 n_active, and TABLE2 = median (min, max).
        assert len(rs) == n_seeds and all(comp(cid)), cid
        for r in rs:
            assert r["au"] == len(au_ids[r["run"]]) == pw05_by_name[r["run"]]["n_active"], r["run"]
        assert (statistics.median(vals), min(vals), max(vals)) == TABLE2[tmpl][a], (cid, vals)
# ASSERT: the runs crossed in Figure 2 (pw02a au_table `converged: false`) are exactly these three
FIG2_NOT_CONVERGED = {"bw-gauss_vae-A1-b1-lr0.0003_s2", "bw-gauss_vae-A1-b1-lr0.0003_s5", "bw-gauss_vae-A2-b1-lr0.003_s5"}
fig2_crossed = {r["run"] for _, _, tmpl, *_ in series for a in archs for r in cells[tmpl.format(a=a)]
                if r["converged"] is False}
assert fig2_crossed == FIG2_NOT_CONVERGED, fig2_crossed
assert all(r["converged"] is True for _, _, tmpl, *_ in series for a in archs for r in cells[tmpl.format(a=a)]
           if r["run"] not in FIG2_NOT_CONVERGED)

fig, ax = plt.subplots(figsize=(FIG_W, 2.092), layout="constrained")  # = the v3.3 print height
offs = [-0.33, -0.11, 0.11, 0.33]  # spaced so the 2x256 lr 3e-3 triangles clear code10
SEED_DX = 0.036
for (_, lab, tmpl, col, mk, _), off in zip(series, offs):
    for i, a in enumerate(archs):
        vals = au(tmpl.format(a=a))
        xs = [i + off + (j - (len(vals) - 1) / 2) * SEED_DX for j in range(len(vals))]
        ax.scatter(xs, vals, s=SEED_S, marker=mk, color=col, edgecolor="none", zorder=3)
        ax.plot([i + off - 0.085, i + off + 0.085], [statistics.median(vals)] * 2, color="black", lw=1.1,
                zorder=2, solid_capstyle="butt")
        for x, r in zip(xs, cells[tmpl.format(a=a)]):
            if r["run"] in FIG2_NOT_CONVERGED:
                ax.scatter([x], [r["au"]], zorder=5, **CROSS)
# reference lines: the affine label top left, the bits label bottom right, clear of the 2x256 triangles
for y, lab, x, ha in [(K_AFF, f"affine dimension {K_AFF}", -0.47, "left"),
                      (K_BITS, rf"$\lceil \log_2 866 \rceil = {K_BITS}$, bits of the binary index", 2.47, "right")]:
    ax.axhline(y, color=GREY_REF, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.text(x, y - 0.7, lab, ha=ha, va="top")
# key drawn by hand at a fixed row pitch (a legend stretches the rows whose labels carry superscripts),
# bottom left, below the 1x64 data
KEY_ROWS = [[(lab, dict(marker=mk, color=col, s=SEED_S, edgecolor="none")) for _, lab, _, col, mk, _ in series[:3]],
            [(series[3][1], dict(marker=series[3][4], color=series[3][3], s=SEED_S, edgecolor="none")),
             ("did not converge", CROSS)]]
for c, (col_x, entries) in enumerate(zip((0, 100), KEY_ROWS)):
    for r, (lab, style) in enumerate(entries):
        dy = 4 + (2 - r) * 11.5  # points above the anchor
        mk_tr = matplotlib.transforms.offset_copy(ax.transAxes, fig=fig, x=col_x + 4, y=dy, units="points")
        tx_tr = matplotlib.transforms.offset_copy(ax.transAxes, fig=fig, x=col_x + 10, y=dy - 2.6, units="points")
        ax.scatter([0.01], [0.0], transform=mk_tr, clip_on=False, zorder=5, **style)
        ax.text(0.01, 0.0, lab, transform=tx_tr, ha="left", va="baseline")
ax.set_xticks(range(3), [arch_label[a] for a in archs])
ax.tick_params(axis="x", length=0)
ax.set_xlim(-0.5, 2.5)
ax.set_ylim(0, 27)
ax.set_yticks(range(0, 26, 5))  # integer ticks only
ax.set_ylabel("active units (of 32)")
save(fig, "fig1_active_units")

t = ["| Cell (beta = 1) | A1 | A2 | A4 |", "|---|---|---|---|"]
for lab, _, tmpl, *_ in series:
    t.append(f"| {lab} | " + " | ".join(
        f"{statistics.median(au(tmpl.format(a=a))):g} ({min(au(tmpl.format(a=a)))}-{max(au(tmpl.format(a=a)))})"
        for a in archs) + " |")
tables.append("## Table: AU at beta=1, median (min-max) over seeds\n\n" + "\n".join(t))

# ---------------------------------------------------------------- Figure 5 (fig2_unit_anatomy): unit x fact anatomy
# Columns: the 30 position facts grouped by subject block X: on_table(X), on(X,Y) for the four Y, holding(X).
COLS = []
for X in bw.BLOCKS:
    COLS += [f"on_table({X})"] + [f"on({X},{Y})" for Y in bw.BLOCKS if Y != X] + [f"holding({X})"]
assert len(COLS) == 30 and len(set(COLS)) == 30 and set(COLS) <= set(bw.FACTS)
COL = {f: i for i, f in enumerate(COLS)}
COL_TICK = ["t" if f.startswith("on_table") else "h" if f.startswith("holding") else f[5] for f in COLS]
F1_MIN = 0.9  # pw05 / pw03a detection threshold


def representative(cell_id):
    """Rule: the seed whose number of two-fact units is closest to the cell median; ties -> lowest seed."""
    runs = sorted((o for o in pw05 if o["cell_id"] == cell_id and "q3" in o), key=lambda o: o["seed"])
    n2 = [sum(u["n_facts"] == 2 for u in o["q3"]["units"]) for o in runs]
    med = statistics.median(n2)
    best = min(range(len(runs)), key=lambda i: (abs(n2[i] - med), runs[i]["seed"]))
    return runs[best], n2, med


def unit_matrix(run):
    """Rows = active units in display order; +1 / -1 = fact detected at F1 >= 0.9 by high / low z_j; 0 = blank.
    Row order: two-fact units by (leftmost, rightmost) column, then one-fact units by column, then units that
    detect no position fact."""
    units = run["q3"]["units"]
    # ASSERT: the rows are exactly the active units of pw02a, and no unit detects a non-position fact only
    assert [u["unit"] for u in units] == au_ids[run["name"]], run["name"]
    assert all(f in COL for u in units for f in u["facts"]), run["name"]

    def key(u):
        cs = sorted(COL[f] for f in u["facts"])
        return ({2: 0, 1: 1, 0: 2}[len(cs)], cs)
    units = sorted(units, key=key)
    M = np.zeros((len(units), 30), dtype=int)
    for r, u in enumerate(units):
        for f, s in zip(u["facts"], u["signs"]):
            assert s in (1, -1)
            M[r, COL[f]] = s
    return M, units


bw_run, bw_n2, bw_med = representative("bw-gauss_vae-A2-b1-lr0.0003")
p5_run, p5_n2, p5_med = representative("prod5-gauss_vae-A2-b1-lr0.0003")
# ASSERT: pw05 two-fact totals per cell (README Q3 table: 55 and 31) and the chosen seeds
assert sum(bw_n2) == 55 and sum(p5_n2) == 31, (bw_n2, p5_n2)
assert (bw_run["seed"], p5_run["seed"]) == (1, 1), (bw_run["seed"], p5_run["seed"])
# ASSERT: the Blocksworld fact lists and signs equal the stored pw03a per-unit x per-fact F1 / sign matrices
with np.load(PW03A / "raw" / f"scores_{bw_run['name']}.npz") as z:
    f1, sgn = z["f1_conf"], z["sign"]
for u in bw_run["q3"]["units"]:
    hit = np.flatnonzero(f1[u["unit"]] >= F1_MIN)
    assert [bw.FACTS[i] for i in hit] == u["facts"], u
    assert [int(sgn[u["unit"], i]) for i in hit] == u["signs"], u

M_bw, units_bw = unit_matrix(bw_run)
unit_matrix(p5_run)  # prod5 is no longer drawn (v1.7: panel (b) dropped, the contrast is in the text); asserts kept
# ASSERT: every Blocksworld two-fact unit of this seed is antipodal (opposite signs), as pw05 reports
assert all(u["signs"][0] != u["signs"][1] for u in units_bw if u["n_facts"] == 2)

def short(f):
    """Column notation: on(X,Y) -> XY, on_table(X) -> tX, holding(X) -> hX."""
    if f.startswith("on_table("):
        return "t" + f[9]
    if f.startswith("holding("):
        return "h" + f[8]
    assert f.startswith("on(") and len(f) == 7, f
    return f[3] + f[5]


def unit_label(u):
    """Row label: X<->Y for a unit pairing on(X,Y) with on(Y,X) (alphabetical); else facts joined by '/'."""
    fs = u["facts"]
    if len(fs) == 2 and all(f.startswith("on(") for f in fs) and short(fs[0]) == short(fs[1])[::-1]:
        a, b = sorted(short(fs[0]))
        return f"{a}\u2194{b}"
    return "/".join(short(f) for f in fs)


labels_bw = [unit_label(u) for u in units_bw]
# ASSERT: the two-fact labels of the reference seed equal the pw05 presentation list (q3_pairs.json)
q3_named = load(EXP / "pw05_unit_anatomy/presentation/q3_pairs.json")["named_pairs_per_run"][bw_run["name"]]


def q3_label(s):
    fs = [x[:-1] for x in s.split("/")]
    assert [x[-1] for x in s.split("/")] in (["+", "-"], ["-", "+"]), s
    return unit_label({"facts": fs})


assert sorted(q3_label(s) for s in q3_named) == sorted(lab for lab, u in zip(labels_bw, units_bw) if u["n_facts"] == 2), \
    (q3_named, labels_bw)
assert len(q3_named) == 11 and sum("\u2194" in lab for lab in labels_bw) == 8, labels_bw

# One panel (v1.7: the prod5 panel was dropped), no title (the caption names the run). Header = block names over
# column letters, with the sign key stacked at their right, above the row labels. Height = the v3.3 print height.
LEFT, LABEL_W, RIGHT_PAD = 0.30, 1.02, 0.03
CELL_W = (FIG_W - LEFT - LABEL_W - RIGHT_PAD) / 30  # inches per column (cells are wider than tall)
HEAD, BOTTOM = 0.32, 0.03  # header = block names + column letters
fig_h = 1.946
CELL_H = (fig_h - HEAD - BOTTOM) / len(M_bw)  # inches per row
fig = plt.figure(figsize=(FIG_W, fig_h))
C_LOW = "#9ECAE9"  # light blue: the low end of the unit's range; dark Okabe-Ito blue: the high end
cmap = ListedColormap([C_LOW, "white", OI_BLUE])  # -1, 0, +1
M, row_labels = M_bw, [lab or "none" for lab in labels_bw]  # a unit that detects no position fact: "none"
h = len(M) * CELL_H
y0 = fig_h - HEAD - h
ax = fig.add_axes([LEFT / FIG_W, y0 / fig_h, 30 * CELL_W / FIG_W, h / fig_h])
ax.pcolormesh(np.arange(31), np.arange(len(M) + 1), M, cmap=cmap, vmin=-1.5, vmax=1.5,
              edgecolors=GREY_GRID, linewidth=0.2)
for b in range(1, 5):
    ax.axvline(6 * b, color=GREY_SEP, lw=0.8)
ax.set_xlim(0, 30)
ax.set_ylim(len(M), 0)
for sp in ax.spines.values():
    sp.set_visible(True)
    sp.set_linewidth(0.6)
ax.xaxis.tick_top()
ax.set_xticks(np.arange(30) + 0.5, COL_TICK)
ax.tick_params(axis="x", length=0, pad=1.5)
ax.set_yticks([])
for r, lab in enumerate(row_labels):
    ax.annotate(lab, xy=(30, r + 0.5), xycoords="data", xytext=(4, 0), textcoords="offset points",
                ha="left", va="center", annotation_clip=False)
for b, X in enumerate(bw.BLOCKS):
    ax.annotate(f"block {X}", xy=(6 * b + 3, 0), xycoords="data", xytext=(0, 11.5), textcoords="offset points",
                ha="center", va="bottom")
# sign key: the cmap colours of +1 (dark blue) and -1 (light blue), level with the block names and the letters
for dy, col, lab in [(11.5, OI_BLUE, "high end of range"), (1.5, C_LOW, "low end of range")]:
    ax.annotate("\u25a0", xy=(30, 0), xycoords="data", xytext=(4, dy), textcoords="offset points",
                ha="left", va="bottom", color=col, annotation_clip=False)
    ax.annotate(lab, xy=(30, 0), xycoords="data", xytext=(13, dy), textcoords="offset points",
                ha="left", va="bottom", annotation_clip=False)
ax.set_ylabel("active units")
save(fig, "fig2_unit_anatomy")

# ---------------------------------------------------------------- Figure 4 (fig3_encoding): encoding and copied fair bits (pw06)
pw06 = load(PW06 / "results.json")
pw06_index = {r["run"]: r for r in load(PW06 / "runs_index.json")["runs"]}
pw06_runs = defaultdict(list)
for r in pw06["per_run"]:
    pw06_runs[r["label"]].append(r)
# (pw06 label, x tick label, expected count per seed 1-5, expected reconstruction-check pass)
ENC = [
    ("E36-L32", "36 facts", [15, 15, 14, 15, 15], True),
    ("E30-L32", "30 position facts", [11, 12, 11, 11, 11], True),
    ("E10a-L32", "index, order 1", [9, 6, 6, 6, 8], False),
    ("E10b-L32", "index, order 2", [7, 7, 7, 6, 7], False),
]
COPIES = [(f"code10-c{c}", str(c), [0] * 5 if c == 1 else [10] * 5, c != 1) for c in (1, 2, 3, 4)]
fig3_rows = []  # (label, seed, count, exact, bitwise, recon_ok) for the report (Figure 4)


def pw06_cell(label, expected, expect_pass):
    rs = sorted(pw06_runs[label], key=lambda r: r["seed"])
    # ASSERT: 5 seeds, beta = 1, 32 latents, numerically fine, every count equals the expected per-seed list and the
    # cell summary, recon_ok is the stated rule on the stored values and equals the per-run file, and the expected
    # pass/fail holds for every seed.
    assert [r["seed"] for r in rs] == [1, 2, 3, 4, 5], label
    assert [r["count"] for r in rs] == expected == pw06["cells"][label]["count"]["values"], (label, rs)
    for r in rs:
        assert r["latent"] == 32 and "-b1-" in r["cell_id"] and r["status"] == "ok" and r["numerical_ok"], r["run"]
        assert pw06_index[r["run"]]["status"] == "ok", r["run"]
        conf = load(PW06 / "runs" / f"{r['run']}.json")["metrics"]["conf"]
        assert (conf["exact"], conf["bitwise"]) == (r["recon_exact"], r["recon_bitwise"]), r["run"]
        assert r["recon_ok"] == (r["recon_exact"] >= 0.95 and r["recon_bitwise"] >= 0.99) == expect_pass, r["run"]
        fig3_rows.append((label, r["seed"], r["count"], r["recon_exact"], r["recon_bitwise"], r["recon_ok"]))
    assert pw06["cells"][label]["recon_pass"] == (5 if expect_pass else 0), label
    return rs


def exact_range(rs):
    """Exact-state reconstruction range over seeds, rounded outward to two decimals."""
    lo = np.floor(min(r["recon_exact"] for r in rs) * 100 + 1e-9) / 100
    hi = np.ceil(max(r["recon_exact"] for r in rs) * 100 - 1e-9) / 100
    return f"{lo:.2f}" if lo == hi else f"{lo:.2f}\u2013{hi:.2f}"


fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 2.202), sharey=True, layout="constrained",  # = v3.3 print height
                         gridspec_kw={"width_ratios": [4, 2.4]})
# (axis, letter, title, categories, colour, marker, annotate, seed spacing); panel (b) spaced wider
# so that the five seeds at the same count stay countable
panels3 = [
    (axes[0], "(a)", "Same 866 states, four encodings", ENC, OI_BLUE, "o", True, 0.1),
    (axes[1], "(b)", "code10, each bit copied c times", COPIES, OI_VERMILLION, "s", False, 0.14),
]
fig3_ranges = {}
for ax, letter, title, cats, col, mk, annotate, dx3 in panels3:
    for i, (label, _, expected, expect_pass) in enumerate(cats):
        rs = pw06_cell(label, expected, expect_pass)
        vals = [r["count"] for r in rs]
        xs = [i + (j - 2) * dx3 for j in range(5)]
        for x, r in zip(xs, rs):
            ax.scatter([x], [r["count"]], s=SEED_S, marker=mk, zorder=3, linewidths=0.9,
                       facecolor=col if r["recon_ok"] else "white", edgecolor=col)
        ax.plot([i - 0.28, i + 0.28], [statistics.median(vals)] * 2, color="black", lw=1.1, zorder=2,
                solid_capstyle="butt")
        fig3_ranges[label] = exact_range(rs)
    # panel (a): the tick label ends with the exact-state reconstruction range over seeds, rounded outward
    ax.set_xticks(range(len(cats)), [c[1] + (f"\nexact {fig3_ranges[c[0]]}" if annotate else "") for c in cats])
    ax.set_xlim(-0.55, len(cats) - 0.45)
    ax.tick_params(axis="x", length=0)
    ax.set_title(panel_title(letter, title), loc="left")
axes[0].set_ylim(-0.8, 17.5)
axes[0].set_yticks(range(0, 16, 5))
axes[0].set_ylabel("active units (of 32)")
axes[1].set_xlabel("copies per bit c")
axes[1].axhline(K_BITS, color=GREY_REF, lw=0.8, ls=(0, (4, 3)), zorder=1)
axes[1].text(-0.5, K_BITS + 0.4, f"{K_BITS} fair bits", ha="left", va="bottom")
axes[0].legend(handles=[
    plt.Line2D([], [], ls="none", marker="o", ms=SEED_S ** 0.5, mfc="black", mec="black", mew=0.9,
               label="reconstructs the data"),
    plt.Line2D([], [], ls="none", marker="o", ms=SEED_S ** 0.5, mfc="white", mec="black", mew=0.9,
               label="does not reconstruct the data"),
], loc="upper right", handletextpad=0.3, borderaxespad=0.2)
save(fig, "fig3_encoding")

# ---------------------------------------------------------------- Figure 3 (fig4_variance): per-latent posterior variances (pw02a)
from pwlib import metrics as pm_metrics

VAR_CELLS = [  # (cell, panel title prefix, colour, seeds)
    ("bw-gauss_vae-A2-b1-lr0.0003", "Blocksworld", OI_BLUE, 5),
    ("code10-gauss_vae-A2-b1-lr0.0003", "code10", OI_VERMILLION, 3),
    ("prod5-gauss_vae-A2-b1-lr0.0003", "prod5", OI_GREEN, 3),
]
fig4_rows = []  # Figure 3: (run, stored AU, count > 0.01, max |Var mu + E sigma^2 - 1|, smallest active Var mu, largest inactive)
fig4_plot = []
for cid, name, col, n_seeds in VAR_CELLS:
    stats = []
    for sd in range(1, n_seeds + 1):
        run = f"{cid}_s{sd}"
        with np.load(PW02A / "raw" / f"{run}.npz") as z:
            mu, logvar, w = z["conf_mu"], z["conf_logvar"], z["conf_weights"]
        d = pm_metrics.asperti_diagnostic(mu, logvar, w)
        n_act, mask, A = pm_metrics.active_units(mu, w)
        assert np.allclose(A, d["var_mu"]) and pm_metrics.AU_THRESHOLD == 0.01
        # ASSERT: count above 0.01 = stored AU (au_table, au_per_run ids); variance law within 0.1 for every latent
        assert n_act == len(au_ids[run]) == next(r["au"] for r in cells[cid] if r["seed"] == sd), run
        assert list(np.flatnonzero(mask)) == au_ids[run], run
        dev = float(np.abs(d["sum"] - 1).max())
        assert dev < 0.1, (run, dev)
        fig4_rows.append((run, len(au_ids[run]), n_act, round(dev, 4), float(A[mask].min()), float(A[~mask].max())))
        stats.append((sd, n_act, d["var_mu"], d["mean_sigma2"]))
    med = statistics.median(st[1] for st in stats)
    rep_seed = min(stats, key=lambda st: (abs(st[1] - med), st[0]))  # count closest to the cell median, lowest seed
    fig4_plot.append((name, col, rep_seed))

# key inside panel (a), in the empty band between 10^-2 and 10^-4 left of the inactive latents
fig, axes = plt.subplots(1, 3, figsize=(FIG_W, 2.122), sharey=True, layout="constrained")  # = v3.3 print height
for ax, letter, (name, col, (sd, n_act, vmu, s2)) in zip(axes, ["(a)", "(b)", "(c)"], fig4_plot):
    order = np.argsort(-vmu, kind="stable")
    x = np.arange(1, 33)
    ax.scatter(x, s2[order], s=10, marker="o", facecolor="white", edgecolor=col, linewidths=0.7, zorder=3)
    ax.scatter(x, vmu[order], s=12, marker="o", color=col, edgecolor="none", zorder=4)
    ax.axhline(0.01, color=GREY_REF, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.set_yscale("log")
    ax.set_xlim(0, 33)
    ax.set_xticks([1, 8, 16, 24, 32])
    ax.set_title(panel_title(letter, f"{name}, seed {sd}"), loc="left")
    ax.text(1, 3e-5, f"{n_act} active", ha="left", va="center")
axes[0].set_ylim(1e-5, 3)
axes[0].set_ylabel("variance")
axes[1].set_xlabel(r"latent, sorted by $\mathrm{Var}_x(\mu_u)$")
axes[0].legend(handles=[  # the threshold first, next to its line; the two shorter rows stay left of the latents
    plt.Line2D([], [], color=GREY_REF, lw=1.0, ls=(0, (4, 3)), label="threshold 0.01"),
    plt.Line2D([], [], ls="none", marker="o", ms=12 ** 0.5, mfc="black", mec="none",
               label=r"$\mathrm{Var}_x(\mu_u)$"),
    plt.Line2D([], [], ls="none", marker="o", ms=10 ** 0.5, mfc="white", mec="black", mew=0.7,
               label=r"$\mathbb{E}_x(\sigma_u^2)$"),
], loc="upper left", bbox_to_anchor=(0.0, 0.53), borderaxespad=0.2, handletextpad=0.4, labelspacing=0.25,
    handlelength=1.6)
save(fig, "fig4_variance")

# ---------------------------------------------------------------- Figure 6 (fig5_legality): sample legality
# As decoded: pw04a descriptive pass. On equal terms (clear and hand_empty recomputed from each sample's 30 position
# facts, as the per-block baseline derives them; post hoc): pw07 B_matched_legality.
B2_LINE = pw04["metrics"]["baselines"]["exact"]["B2"]
assert f"{B2_LINE:.3f}" == "0.540", B2_LINE
leg_cells = defaultdict(list)
for v in pw04["metrics"]["per_model"].values():
    leg_cells[v["cell_id"]].append(v)
# v1.6 Table 6 row "5.6 Legality" and tables.md "descriptive legality" (prior range, aggregate-posterior range,
# runs with posterior >= 0.95, runs with prior > B2); first typed for report v1.3 Table 5
TABLE5 = {
    ("A1", "1"): [("0.109", "0.167", "0.546", "0.636", 0, 0), ("0.320", "0.337", "0.725", "0.738", 0, 0)],
    ("A2", "1"): [("0.413", "0.464", "0.811", "0.845", 0, 0), ("0.462", "0.550", "0.863", "0.899", 0, 2)],
    ("A4", "1"): [("0.493", "0.540", "0.858", "0.872", 0, 0), ("0.539", "0.578", "0.888", "0.895", 0, 4)],
    ("A1", "0.25"): [("0.067", "0.085", "0.878", "0.890", 0, 0), ("0.341", "0.385", "0.935", "0.943", 0, 0)],
    ("A2", "0.25"): [("0.343", "0.397", "0.944", "0.967", 4, 0), ("0.423", "0.472", "0.969", "0.972", 5, 0)],
    ("A4", "0.25"): [("0.412", "0.452", "0.962", "0.969", 5, 0), ("0.437", "0.522", "0.969", "0.975", 5, 0)],
}
TC_CELL, TC_ROW = "bw-tc_vae-A2-b2-lr0.0003", ("0.454", "0.464", "0.738", "0.754", 0, 0)
fig5_rows = []  # Figure 6: (cell, seed, prior as decoded, prior on equal terms, aggregate posterior as decoded)
B7 = pw07["B_matched_legality"]
B7_RUN = {r["name"]: r for r in B7["per_run"]}
# ASSERT: pw07's baseline is the pw04a B2 line drawn here, recomputed from the training marginals
assert B7["baseline"]["pw04a_stored"] == B7["baseline"]["B2_train_marginals"] == B2_LINE, B7["baseline"]
assert all(B7["checks"].values()), B7["checks"]
# runs that fail the convergence criterion (pw02a au_table `converged: false`): beta = 1 as in Figure 2, plus beta = 0.25
FIG6_NOT_CONVERGED = FIG2_NOT_CONVERGED | {"bw-gauss_vae-A2-b0.25-lr0.0003_s2", "bw-gauss_vae-A4-b0.25-lr0.0003_s1",
                                           "bw-gauss_vae-A4-b0.25-lr0.0003_s5", "bw-gauss_vae-A4-b0.25-lr0.003_s3",
                                           "bw-gauss_vae-A4-b0.25-lr0.003_s5"}
fig6_crossed = set()


def legality(cid, expected, n_seeds):
    vs = sorted(leg_cells[cid], key=lambda v: v["seed"])
    assert [v["seed"] for v in vs] == list(range(1, n_seeds + 1)), cid
    p = [v["prior_mode"]["rate"] for v in vs]
    q = [v["aggpost_mode"]["rate"] for v in vs]
    lo = lambda x: f"{np.floor(x * 1000 + 1e-9) / 1000:.3f}"
    hi = lambda x: f"{np.ceil(x * 1000 - 1e-9) / 1000:.3f}"
    got = (lo(min(p)), hi(max(p)), lo(min(q)), hi(max(q)),
           sum(x >= 0.95 for x in q), sum(x > B2_LINE for x in p))
    assert got == expected, (cid, got, expected)  # ASSERT: plotted values reproduce the typed ranges and counts
    r_eq, conv = [], []
    for v in vs:
        name = f"{cid}_s{v['seed']}"
        b7 = B7_RUN[name]
        # ASSERT: pw07's as-decoded rates are pw04a's, and its equal-terms rate is the one plotted (hollow circle)
        assert b7["cell_id"] == cid and b7["seed"] == v["seed"], name
        assert abs(b7["prior"]["legal"] - v["prior_mode"]["rate"]) < 1e-12, name
        assert abs(b7["aggpost"]["legal"] - v["aggpost_mode"]["rate"]) < 1e-12, name
        assert b7["prior"]["below_baseline"]["rederived"] == (b7["prior"]["legal_rederived"] < B2_LINE), name
        r_eq.append(b7["prior"]["legal_rederived"])
        conv.append(next(r["converged"] for r in cells[cid] if r["seed"] == v["seed"]))
        if conv[-1] is False:
            fig6_crossed.add(name)
        fig5_rows.append((cid, v["seed"], v["prior_mode"]["rate"], r_eq[-1], v["aggpost_mode"]["rate"]))
    return p, q, r_eq, conv


C_PRIOR, M_PRIOR, C_POST, M_POST = OI_BLUE, "o", OI_BLUE, "s"  # all Blocksworld: one colour, shapes separate
LR_DX, SEED_DX5 = 0.25, 0.075
Y_CROSS = 1.035  # height of the "did not converge" crosses, just above the data area


def legality_markers(ax, xs, p, q, r_eq, conv):
    """Prior as decoded (filled circle), prior on equal terms (hollow circle, same x), aggregate posterior (filled
    square); one reddish-purple cross above the x of each run that did not converge, clear of every data mark."""
    ax.scatter(xs, p, s=SEED_S, marker=M_PRIOR, color=C_PRIOR, edgecolor="none", zorder=3)
    ax.scatter(xs, r_eq, s=SEED_S, marker=M_PRIOR, facecolor="white", edgecolor=C_PRIOR, linewidths=0.7, zorder=3)
    ax.scatter(xs, q, s=SEED_S, marker=M_POST, color=C_POST, edgecolor="none", zorder=3)
    for x, c in zip(xs, conv):
        if c is False:
            ax.scatter([x], [Y_CROSS], zorder=5, **CROSS6)


def below_ticks(ax, x, text, pts):
    """Text centred at data x, `pts` points below the x axis (second line under the tick labels)."""
    ax.annotate(text, xy=(x, 0), xycoords=ax.get_xaxis_transform(), xytext=(0, -pts), textcoords="offset points",
                ha="center", va="top")


below_eq = defaultdict(list)  # beta -> runs whose prior samples on equal terms fall below the baseline
fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 2.114), sharey=True, layout="constrained",  # = v3.3 print height
                         gridspec_kw={"width_ratios": [3.65, 2.9]})
for ax, letter, beta in [(axes[0], "(a)", "1"), (axes[1], "(b)", "0.25")]:
    ticks, ticklabels = [], []
    for i, a in enumerate(archs):
        for j, lr in enumerate(["0.0003", "0.003"]):
            x0 = i + (j - 0.5) * 2 * LR_DX
            cid = f"bw-gauss_vae-{a}-b{beta}-lr{lr}"
            p, q, r_eq, conv = legality(cid, TABLE5[(a, beta)][j], 5)
            below_eq[beta] += [(cid, k + 1) for k, x in enumerate(r_eq) if x < B2_LINE]
            xs = [x0 + (k - 2) * SEED_DX5 for k in range(5)]
            legality_markers(ax, xs, p, q, r_eq, conv)
            ticks.append(x0)
            ticklabels.append(["lower", "higher"][j])  # the learning rate; values in the caption
        below_ticks(ax, i, arch_label[a], 14)
    if beta == "1":
        p, q, r_eq, conv = legality(TC_CELL, TC_ROW, 3)
        below_eq["tc"] += [(TC_CELL, k + 1) for k, x in enumerate(r_eq) if x < B2_LINE]
        xs = [2.95 + (k - 1) * SEED_DX5 for k in range(3)]
        legality_markers(ax, xs, p, q, r_eq, conv)
        ticks.append(2.95)
        ticklabels.append("β-TCVAE")
        below_ticks(ax, 2.95, r"2×256, β$_\mathrm{TC}$ = 2", 14)
        ax.axvline(2.62, color=GREY_GRID, lw=0.8, zorder=0)
        ax.set_xlim(-0.45, 3.25)
    else:
        ax.set_xlim(-0.45, 2.45)
    ax.set_xticks(ticks, ticklabels)
    ax.tick_params(axis="x", length=0)
    for y in (B2_LINE, 0.95):
        ax.axhline(y, color=GREY_REF, lw=0.8, ls=(0, (4, 3)) if y < 0.9 else (0, (1, 2)), zorder=1)
    ax.set_title(panel_title(letter, f"β = {beta}"), loc="left")
axes[0].set_ylim(0, 1.07)
axes[0].set_yticks([0, 0.25, 0.5, 0.75, 1])
axes[0].set_ylabel("legal fraction of samples")
# shared key above both panels; the tick row is the learning rate (lr, Section 3.2), with a row header left of
# panel (a) level with the ticks; the row below names the network
axes[0].annotate("lr", xy=(0, 0), xycoords="axes fraction", xytext=(-4, -3.5), textcoords="offset points",
                 ha="right", va="top")
fig.legend(handles=[
    plt.Line2D([], [], ls="none", marker=M_PRIOR, ms=SEED_S ** 0.5, mfc=C_PRIOR, mec="none", label="prior, as decoded"),
    plt.Line2D([], [], ls="none", marker=M_PRIOR, ms=SEED_S ** 0.5, mfc="white", mec=C_PRIOR, mew=0.7,
               label="prior, on equal terms"),
    plt.Line2D([], [], ls="none", marker=M_POST, ms=SEED_S ** 0.5, mfc=C_POST, mec="none",
               label="aggregate posterior, as decoded"),
    plt.Line2D([], [], ls="none", marker="x", ms=4, mec=OI_PURPLE, mew=0.9, label="did not converge"),
    plt.Line2D([], [], color=GREY_REF, lw=0.8, ls=(0, (4, 3)), label=f"per-block baseline {B2_LINE:.2f}"),
    plt.Line2D([], [], color=GREY_REF, lw=0.8, ls=(0, (1, 2)), label="planned level 0.95"),
], loc="outside upper center", ncol=3, handletextpad=0.3, columnspacing=1.5)
# ASSERT: crossed runs = au_table `converged: false` among the plotted runs; equal-terms counts below the baseline
# (pw07 B counts): 10 at beta = 1, all ten 1x64 runs; 11 at beta = 0.25; none for the beta-TCVAE
assert fig6_crossed == FIG6_NOT_CONVERGED, fig6_crossed
cnt = B7["counts"]
assert len(below_eq["1"]) == 10 == cnt["beta1"]["prior_below_rederived"], below_eq["1"]
assert {c for c, _ in below_eq["1"]} == {"bw-gauss_vae-A1-b1-lr0.0003", "bw-gauss_vae-A1-b1-lr0.003"}, below_eq["1"]
assert len(below_eq["0.25"]) == 11 == cnt["beta025"]["prior_below_rederived"], below_eq["0.25"]
assert len(below_eq["tc"]) == 0 == cnt["tc_vae_b2"]["prior_below_rederived"], below_eq["tc"]
save(fig, "fig5_legality")

# ---------------------------------------------------------------- tables.md (values of the typed Tables 2-4 and 6)
betas = [0.25, 1, 4]
t = ["| Arch | lr | beta 0.25 | beta 1 | beta 4 |", "|---|---|---|---|---|"]
for a in archs:
    for lr in ["0.0003", "0.003"]:
        row = []
        for b in betas:
            cid = f"bw-gauss_vae-{a}-b{b:g}-lr{lr}"
            vals, cs = au(cid), comp(cid)
            row.append(f"{statistics.median(vals):g} ({min(vals)}-{max(vals)}){'' if all(cs) else ' *'}")
        t.append(f"| {a} | {float(lr):g} | " + " | ".join(row) + " |")
tables.append("## Table: AU over beta (median (min-max); * = fails competence)\n\n" + "\n".join(t))

srcs = ["w64", "w128", "w128s2", "w128s3", "w256"]
h24 = pw02["metrics"]["h24"]
t = ["| Source | b_lin learned | b_lin fixed | b_raw learned | K_probe | K_pca99 |", "|---|---|---|---|---|---|"]
for s in srcs:
    t.append(f"| {s} | {h24[s]['lin_learned']['values']} | {h24[s]['lin_fixed']['values']} | "
             f"{h24[s]['raw_learned']['values']} | {h24[s]['K_probe']} | {h24[s]['K_pca99']} |")
tables.append("## Table: activation substrates, AU per seed\n\n" + "\n".join(t))

pm = pw03["metrics"]["per_model"]
byc = defaultdict(list)
for v in pm.values():
    byc[v.get("cell_id") or "control"].append(v)
rows = [
    ("top-k SAE, k = 5", "bw-topk_sae-k5-lr0.0003"),
    ("top-k SAE, k = 10", "bw-topk_sae-k10-lr0.0003"),
    ("beta-VAE A2, beta 1, lr 3e-4 (H3.2 comparator)", "bw-gauss_vae-A2-b1-lr0.0003"),
    ("beta-TCVAE A2, beta_TC 2 (H3.2 comparator)", "bw-tc_vae-A2-b2-lr0.0003"),
    ("beta-VAE A1, beta 1, lr 3e-4", "bw-gauss_vae-A1-b1-lr0.0003"),
    ("beta-VAE A1, beta 1, lr 3e-3", "bw-gauss_vae-A1-b1-lr0.003"),
    ("beta-VAE A2, beta 1, lr 3e-3", "bw-gauss_vae-A2-b1-lr0.003"),
    ("beta-VAE A4, beta 1, lr 3e-4", "bw-gauss_vae-A4-b1-lr0.0003"),
    ("beta-VAE A4, beta 1, lr 3e-3", "bw-gauss_vae-A4-b1-lr0.003"),
    ("plain autoencoder", "bw-plain_ae-A2-lr0.0003"),
]
t = ["| Model (raw states) | Cov30 per seed | MonoFrac per seed | DCI disentanglement per seed |", "|---|---|---|---|"]
for lab, cid in rows:
    vs = sorted(byc[cid], key=lambda v: v["seed"])
    cov = ", ".join(f"{v['cov30']:.2f}" for v in vs)
    mono = ", ".join(f"{v['mono_frac']:.2f}" for v in vs)
    dci = ", ".join(f"{v['dci']['disentanglement']:.2f}" for v in vs)
    t.append(f"| {lab} | {cov} | {mono} | {dci} |")
for name in ["control_fact_oracle", "control_pca25"]:
    v = pm[name]
    t.append(f"| {name} | {v['cov30']:.2f} | {v['mono_frac']:.2f} | {v['dci']['disentanglement']:.2f} |")
tables.append("## Table: fact alignment on raw states\n\n" + "\n".join(t))

# ---------------------------------------------------------------- legality table (descriptive; Figure 6, v1.6 Table 6)
lp = pw04["metrics"]["per_model"]
B2 = pw04["metrics"]["baselines"]["exact"]["B2"]
lc = defaultdict(list)
for v in lp.values():
    lc[v["cell_id"]].append(v)
t = ["| Cell | prior legality (mode) min-max | aggregate-posterior legality min-max | runs passing 0.95 |",
     "|---|---|---|---|"]
for cid in sorted(c for c in lc if c.startswith(("bw-gauss", "bw-tc"))):
    vs = lc[cid]
    p = [v["prior_mode"]["rate"] for v in vs]
    q = [v["aggpost_mode"]["rate"] for v in vs]
    t.append(f"| {cid} | {min(p):.3f}-{max(p):.3f} | {min(q):.3f}-{max(q):.3f} | "
             f"{sum(v['competence'] for v in vs)}/{len(vs)} |")
above = sum(v["prior_mode"]["rate"] > B2 for c in lc if c.startswith("bw-gauss") and "-b1-" in c for v in lc[c])
t.append(f"\nB2 = {B2:.4f}; beta=1 runs with prior legality above B2: {above}")
tables.append("## Table: descriptive legality (unregistered pass)\n\n" + "\n".join(t))


# ---------------------------------------------------------------- pw07 (post hoc): Tables 2 and 6, section 5
def stat(xs):
    """(median, min, max) of a list, as pw07 summarises its cells."""
    return statistics.median(xs), min(xs), max(xs)


def outward(lo, hi, nd):
    """A range rounded outward to nd decimals, as the report prints ranges."""
    f = 10 ** nd
    return f"{np.floor(lo * f + 1e-9) / f:.{nd}f}\u2013{np.ceil(hi * f - 1e-9) / f:.{nd}f}"


A7 = pw07["A_unit_kl"]
a7_runs = defaultdict(list)
for r in A7["per_run"]:
    a7_runs[r["cell_id"]].append(r)
au_by_run = {r["run"]: r for r in au_rows}
A7_KEYS = ("au", "rate_nats", "mean_kl_per_active_nats")
for cid, rs in a7_runs.items():
    for r in rs:  # ASSERT: pw07's count per run is the stored pw02a count, and the KL per active unit is rate-consistent
        assert r["au"] == au_by_run[r["name"]]["au"], r["name"]
        assert r["frac_rate_active"] > 0.99 and abs(r["kl_active_nats"] / r["au"] - r["mean_kl_per_active_nats"]) < 1e-9
    for key in A7_KEYS:  # ASSERT: pw07's per-cell summaries equal the summaries of its per-run values
        c = A7["per_cell"][cid][key]
        assert stat([r[key] for r in rs]) == (c["median"], c["min"], c["max"]), (cid, key)

# (tables.md label, cell): the beta = 1 cells of v1.6 Table 6 row "5.1 KL per active unit", in its order: Blocksworld by
# network, lower lr first, then code10 and prod5 by network
KL_CELLS = [(f"{lab}, {a}", tmpl.format(a=a)) for a in archs for lab, _, tmpl, *_ in series[:2]] + [
    (f"{lab}, {a}", tmpl.format(a=a)) for lab, _, tmpl, *_ in series[2:] for a in archs]
bw1 = [r for _, cid in KL_CELLS[:6] for r in a7_runs[cid]]
g7 = A7["global_"]
# ASSERT: the global beta = 1 ranges of pw07 are those of the 30 Blocksworld runs listed here
assert sorted(r["name"] for r in bw1) == sorted(r["run"] for _, _, tmpl, *_ in series[:2] for a in archs
                                              for r in cells[tmpl.format(a=a)]), "pw07 beta = 1 runs"
assert [min(r["rate_nats"] for r in bw1), max(r["rate_nats"] for r in bw1)] == g7["beta1_rate_range_nats"]
assert [min(r["au"] for r in bw1), max(r["au"] for r in bw1)] == g7["beta1_au_range"]
assert [min(r["mean_kl_per_active_nats"] for r in bw1),
        max(r["mean_kl_per_active_nats"] for r in bw1)] == g7["beta1_mean_kl_active_range_nats"]
assert [min(r["rate_minus_H_nats"] for r in bw1), max(r["rate_minus_H_nats"] for r in bw1)] == \
    g7["beta1_rate_minus_H_range_nats"]
assert min(r["frac_rate_active"] for r in bw1) == g7["beta1_frac_rate_active_min"]
assert max(r["kl_inactive_max_nats"] for r in bw1) == g7["beta1_max_inactive_unit_kl_nats"]
t = [("| Cell (beta = 1) | runs | active units, median (min-max) | rate (nats), median (min-max) | "
      "mean KL per active unit (nats), median (min-max) |"), "|---|---|---|---|---|"]
for lab, cid in KL_CELLS:
    c = A7["per_cell"][cid]
    t.append(f"| {lab} | {c['n_runs']} | {c['au']['median']:g} ({c['au']['min']:g}-{c['au']['max']:g}) | "
             f"{c['rate_nats']['median']:.4f} ({c['rate_nats']['min']:.4f}-{c['rate_nats']['max']:.4f}) | "
             f"{c['mean_kl_per_active_nats']['median']:.4f} ({c['mean_kl_per_active_nats']['min']:.4f}-"
             f"{c['mean_kl_per_active_nats']['max']:.4f}) |")
lo_r, hi_r = g7["beta1_rate_range_nats"]
lo_k, hi_k = g7["beta1_mean_kl_active_range_nats"]
lo_e, hi_e = g7["beta1_rate_minus_H_range_nats"]
t.append(f"\nOver the 30 Blocksworld runs at beta = 1: active units {g7['beta1_au_range'][0]}-{g7['beta1_au_range'][1]}; "
         f"rate {lo_r:.4f}-{hi_r:.4f} nats (outward {outward(lo_r, hi_r, 2)}, {outward(lo_r, hi_r, 1)}); "
         f"mean KL per active unit {lo_k:.4f}-{hi_k:.4f} (outward {outward(lo_k, hi_k, 2)}); "
         f"rate minus H(X) {lo_e:.4f}-{hi_e:.4f} (outward {outward(lo_e, hi_e, 1)}); "
         f"smallest share of the rate in the active units {g7['beta1_frac_rate_active_min']:.5f}; "
         f"largest KL of an inactive unit {g7['beta1_max_inactive_unit_kl_nats']:.5f} nats.")
tables.append("## Table: KL per active unit, beta = 1 (pw07 A, post hoc)\n\n" + "\n".join(t))

# pw06 follow-up (Table 2): counts and rates of pw06, KL per active unit of pw07
P6_ORDER = ["E36-L32", "E30-L32", "E10a-L32", "E10b-L32", "code10-c1", "code10-c2", "code10-c3", "code10-c4",
            "E36-L8", "E36-L10", "E36-L12", "E36-L16", "E36-L24", "E36-L64", "E10a-b0.25"]
assert sorted(P6_ORDER) == sorted(A7["per_cell_pw06"]), sorted(A7["per_cell_pw06"])
a7_p6 = defaultdict(list)
for r in A7["per_run_pw06"]:
    a7_p6[r["label"]].append(r)
t = [("| pw06 condition | active units, median (min-max) | rate (nats), median | mean KL per active unit (nats), "
      "median (min-max) | negative ELBO (nats), median | H(X) (nats) |"), "|---|---|---|---|---|---|"]
for lab in P6_ORDER:
    c, c6 = A7["per_cell_pw06"][lab], pw06["cells"][lab]
    # ASSERT: pw07's counts and median rate per condition are pw06's (Table 2), and its summaries match its runs
    assert [int(x) for x in c["au"]["values"]] == c6["count"]["values"], lab
    assert c["rate_nats"]["median"] == c6["rate_nats"], lab
    assert stat([r["rate_nats"] for r in a7_p6[lab]]) == (c["rate_nats"]["median"], c["rate_nats"]["min"],
                                                          c["rate_nats"]["max"]), lab
    kl = c["mean_kl_per_active_nats"]
    if kl is None:
        assert c6["count"]["values"] == [0] * 5, lab  # no active unit: the KL per active unit is undefined
        kl_txt = "n/a (no active unit)"
    else:
        assert stat([r["mean_kl_per_active_nats"] for r in a7_p6[lab]]) == (kl["median"], kl["min"], kl["max"]), lab
        kl_txt = f"{kl['median']:.4f} ({kl['min']:.4f}-{kl['max']:.4f})"
    t.append(f"| {lab} | {c6['count']['median']:g} ({c6['count']['min']}-{c6['count']['max']}) | "
             f"{c['rate_nats']['median']:.4f} | {kl_txt} | {c6['neg_elbo_nats']:.4f} | {c6['H_X_nats']:.4f} |")
tables.append("## Table: pw06 follow-up, KL per active unit (Table 2; pw07 A, post hoc)\n\n" + "\n".join(t))

# legality on equal terms (Figure 6 hollow circles, v1.6 Table 6 row 5.6, section 5.6)
pc7 = B7["per_cell"]
b7_cells = defaultdict(list)
for r in B7["per_run"]:
    b7_cells[r["cell_id"]].append(r)
L_KEYS = [("prior", "legal"), ("prior", "legal_rederived"), ("prior", "wf_all"), ("prior", "cross_given_wf"),
          ("prior", "implied_cross_given_wf_independent_blocks"), ("prior", "derived_given_cross"),
          ("aggpost", "legal"), ("aggpost", "legal_rederived"), ("aggpost", "wf_all")]
for cid, rs in b7_cells.items():
    for side, key in L_KEYS:  # ASSERT: pw07's per-cell summaries equal the summaries of its per-run values
        c = pc7[cid][side][key]
        assert stat([r[side][key] for r in rs]) == (c["median"], c["min"], c["max"]), (cid, side, key)
    for side in ("prior", "aggpost"):
        for m, k7 in (("original", "legal"), ("rederived", "legal_rederived")):
            assert pc7[cid][side][f"below_baseline_{m}"] == sum(r[side][k7] < B2_LINE for r in rs), (cid, side, m)


def group(beta, family="gauss_vae", arch=None):
    return [r for r in B7["per_run"] if r["beta"] == beta and r["family"] == family and arch in (None, r["arch"])]


# ASSERT: pw07 counts and ranges recomputed from its per-run values (the numbers of section 5.6 and v1.6 Table 6)
for key, rs in [("beta1", group(1.0)), ("beta025", group(0.25)), ("tc_vae_b2", group(2.0, "tc_vae"))]:
    c = cnt[key]
    assert c["n_runs"] == len(rs), key
    assert c["prior_below_original"] == sum(r["prior"]["legal"] < B2_LINE for r in rs), key
    assert c["prior_below_rederived"] == sum(r["prior"]["legal_rederived"] < B2_LINE for r in rs), key
    assert c["prior_rederived_ci_entirely_above"] == sum(r["prior"]["legal_rederived_wilson95"][0] > B2_LINE
                                                         for r in rs), key
    assert c["aggpost_below_original"] == sum(r["aggpost"]["legal"] < B2_LINE for r in rs) == 0, key
    assert c["aggpost_below_rederived"] == sum(r["aggpost"]["legal_rederived"] < B2_LINE for r in rs) == 0, key
assert (cnt["beta1"]["prior_below_original"], cnt["beta025"]["prior_below_original"],
        cnt["tc_vae_b2"]["prior_below_original"]) == (24, 30, 3), cnt
assert cnt["beta1"]["prior_rederived_ci_entirely_above"] == 20, cnt["beta1"]
vr = B7["verify_report"]
b1 = group(1.0)
b1_cells = sorted({r["cell_id"] for r in b1})


def rng(rs, side, key):
    return [min(r[side][key] for r in rs), max(r[side][key] for r in rs)]


def med_rng(side, key):
    return [min(pc7[c][side][key]["median"] for c in b1_cells), max(pc7[c][side][key]["median"] for c in b1_cells)]


assert rng(b1, "prior", "legal_rederived") == vr["rederived_prior_run_range_beta1"]
assert rng(b1, "prior", "legal") == vr["prior_legal_run_range_beta1"]
assert rng(b1, "aggpost", "legal") == vr["agg_legal_run_range_beta1"]
assert med_rng("prior", "cross_given_wf") == vr["cross_given_wf_cell_median_range"]
assert med_rng("prior", "implied_cross_given_wf_independent_blocks") == vr["implied_cross_cell_median_range"]
assert med_rng("prior", "derived_given_cross") == vr["derived_given_cross_cell_median_range"]
assert med_rng("prior", "wf_all") == vr["wf_cell_median_range"]
assert med_rng("aggpost", "wf_all") == vr["agg_wf_cell_median_range"]
assert med_rng("aggpost", "legal_rederived") == vr["rederived_agg_cell_median_range_beta1"]
assert min(r["aggpost"]["cross_given_wf"] for r in b1) == vr["agg_cross_given_wf_min_run"]
assert max(abs(r["prior"]["wf_minus_product"]) for r in b1) == vr["wf_minus_product_max_abs_run"]

t = [("| Runs | n | prior below baseline, as decoded | prior below, on equal terms | "
      "prior on equal terms, Wilson 95% interval entirely above | aggregate posterior below (as decoded / equal terms) |"),
     "|---|---|---|---|---|---|"]
for lab, key in [("Gaussian-latent beta-VAEs, beta = 1", "beta1"), ("Gaussian-latent beta-VAEs, beta = 0.25", "beta025"),
                 ("beta-TCVAE, beta_TC = 2, A2", "tc_vae_b2")]:
    c = cnt[key]
    t.append(f"| {lab} | {c['n_runs']} | {c['prior_below_original']} | {c['prior_below_rederived']} | "
             f"{c['prior_rederived_ci_entirely_above']} | {c['aggpost_below_original']} / {c['aggpost_below_rederived']} |")
t += ["", ("| Cell | prior as decoded min-max | prior on equal terms median (min-max) | prior well formed min-max | "
           "aggregate posterior as decoded min-max | aggregate posterior on equal terms median (min-max) | "
           "prior runs below baseline (decoded / equal terms) |"), "|---|---|---|---|---|---|---|"]


def cell_range(c, side, key, with_median=False):
    """'min-max' of a pw07 per-cell summary, optionally preceded by its median."""
    x = c[side][key]
    return (f"{x['median']:.4f} ({x['min']:.4f}-{x['max']:.4f})" if with_median
            else f"{x['min']:.4f}-{x['max']:.4f}")


for cid in sorted(pc7):
    c = pc7[cid]
    t.append(f"| {cid} | {cell_range(c, 'prior', 'legal')} | {cell_range(c, 'prior', 'legal_rederived', True)} | {cell_range(c, 'prior', 'wf_all')} | "
             f"{cell_range(c, 'aggpost', 'legal')} | {cell_range(c, 'aggpost', 'legal_rederived', True)} | "
             f"{c['prior']['below_baseline_original']} / {c['prior']['below_baseline_rederived']} |")
a1 = group(1.0, arch="A1")
big = [r for r in b1 if r["arch"] != "A1"]
tcr = group(2.0, "tc_vae")
x = rng(b1, "prior", "legal_rederived")
t.append("")
t.append(f"Baseline B2 = {B2_LINE:.5f} (pw07 = pw04a stored). Beta = 1, prior on equal terms per run: all "
         f"{x[0]:.4f}-{x[1]:.4f} (outward {outward(*x, 2)}); 1x64 {outward(*rng(a1, 'prior', 'legal_rederived'), 2)} "
         f"({rng(a1, 'prior', 'legal_rederived')[0]:.4f}-{rng(a1, 'prior', 'legal_rederived')[1]:.4f}); 2x256 and 2x1024 "
         f"{outward(*rng(big, 'prior', 'legal_rederived'), 2)} ({rng(big, 'prior', 'legal_rederived')[0]:.4f}-"
         f"{rng(big, 'prior', 'legal_rederived')[1]:.4f}). Aggregate posterior on equal terms per run: "
         f"{rng(b1, 'aggpost', 'legal_rederived')[0]:.4f}-{rng(b1, 'aggpost', 'legal_rederived')[1]:.4f} "
         f"(outward {outward(*rng(b1, 'aggpost', 'legal_rederived'), 2)}).")
t.append(f"1x64 at beta = 1, prior well formed per run: {rng(a1, 'prior', 'wf_all')[0]:.4f}-{rng(a1, 'prior', 'wf_all')[1]:.4f} "
         f"(outward {outward(*rng(a1, 'prior', 'wf_all'), 2)}). beta-TCVAE prior on equal terms: "
         f"{rng(tcr, 'prior', 'legal_rederived')[0]:.4f}-{rng(tcr, 'prior', 'legal_rederived')[1]:.4f} "
         f"(outward {outward(*rng(tcr, 'prior', 'legal_rederived'), 2)}).")
t.append("Beta = 1, ranges of the six configuration medians (prior): between-block constraints among well-formed samples "
         f"{outward(*vr['cross_given_wf_cell_median_range'], 2)}; the same for independent blocks with the model's "
         f"marginals {outward(*vr['implied_cross_cell_median_range'], 2)}; derived facts consistent given legal positions "
         f"{outward(*vr['derived_given_cross_cell_median_range'], 2)}; well formed {outward(*vr['wf_cell_median_range'], 2)}. "
         f"Aggregate posterior: well formed {outward(*vr['agg_wf_cell_median_range'], 2)}; between-block constraints among "
         f"well-formed samples at least {vr['agg_cross_given_wf_min_run']:.4f} in every run. Largest |well formed - product "
         f"of per-block fractions| over the beta = 1 runs {vr['wf_minus_product_max_abs_run']:.4f}; over all "
         f"{len(B7['per_run'])} runs {max(abs(r['prior']['wf_minus_product']) for r in B7['per_run']):.4f}.")
agg_m = {k: med_rng("aggpost", k) for k in ("cross_given_wf", "implied_cross_given_wf_independent_blocks",
                                             "derived_given_cross")}
t.append("Aggregate posterior, ranges of the six configuration medians at beta = 1 (Table 6): between-block constraints "
         "among well-formed samples " + "-".join(f"{v:.4f}" for v in agg_m["cross_given_wf"])
         + "; independent blocks with the model's marginals "
         + "-".join(f"{v:.4f}" for v in agg_m["implied_cross_given_wf_independent_blocks"])
         + f" (outward {outward(*agg_m['implied_cross_given_wf_independent_blocks'], 2)}); clear and hand_empty "
         "consistent given legal positions " + "-".join(f"{v:.4f}" for v in agg_m["derived_given_cross"])
         + f" (outward {outward(*agg_m['derived_given_cross'], 2)}).")
tables.append("## Table: legality on equal terms (pw07 B, post hoc; clear and hand_empty recomputed from the 30 "
              "position facts)\n\n" + "\n".join(t))

# ---------------------------------------------------------------- v1.7: supporting numbers that left the PDF
# Report v1.7 was cut from 15 to 10 pages. Every number that v1.6 printed and v1.7 no longer prints is read here from
# its JSON key and asserted against the value v1.6 printed (old Table 6, Sections 3-5, Table 4 positive control).
# Numbers that exist only in prose records are quoted with their source path and marked as not recomputed.
PW01_RES = load(EXP / "pw01_state_space_ground_truth/results/analysis_001/results.json")["metrics"]
pw05s = load(PW05)["metrics"]["summaries"]
pw05_lin = load(PW05)["metrics"]["q4_b_lin"]
MC4096 = EXP / "pw05_unit_anatomy/negelbo_4096/cell_negelbo_nats.json"  # 4,096-draw audit re-derivation
mc4096 = load(MC4096)
LN2 = float(np.log(2))
v17 = []

# Data (v1.6 Sections 1, 3.1, 4.2)
seq = PW01_RES["task"]["sequence_lengths"]
assert seq["train"]["n_sequences"] == 16004 == 12803 + 3201 and seq["test"]["n_sequences"] == 4001, seq
assert (PW01_RES["n_arm_empty"], PW01_RES["n_holding"], PW01_RES["n_legal"]) == (501, 365, 866)
assert (PW01_RES["linear"]["rank_all"], PW01_RES["linear"]["affine_dim_all"]) == (26, 25)
hx = pw06["cells"]["E36-L32"]["H_X_nats"]
assert f"{hx:.2f}" == "6.47" and f"{hx / LN2:.2f}" == "9.34", hx
v17.append("### Data (v1.6 Sections 1, 3.1 and 4.2)\n\n"
           f"- Sequences: training file {seq['train']['n_sequences']} (split by sequence into 12,803 training and 3,201 "
           f"development sequences; the split sizes are quoted from `docs/EXPERIMENT_LOG.md`, pw02 entry, not "
           f"recomputed here), test file {seq['test']['n_sequences']} (pw01 `metrics.task.sequence_lengths`).\n"
           f"- Legal states with the hand empty / holding a block: {PW01_RES['n_arm_empty']} / {PW01_RES['n_holding']} "
           f"of {PW01_RES['n_legal']} (pw01 `metrics.n_arm_empty`, `metrics.n_holding`).\n"
           f"- Rank of the 866 x 36 legal-state matrix: {PW01_RES['linear']['rank_all']} (affine dimension "
           f"{PW01_RES['linear']['affine_dim_all']} plus the offset; pw01 `metrics.linear.rank_all`).\n"
           f"- H(X) under the test-split frequencies: {hx:.4f} nats = {hx / LN2:.4f} bits (pw06 `cells.*.H_X_nats`).")

# Section 5.1: rate per run, prod5 entropy, counts under the inactivity threshold of Asperti and Trentin (2020)
rate_rng = {}
for lab in ("code10", "prod5"):
    rs = [r for _, cid in KL_CELLS if cid.startswith(lab) for r in a7_runs[cid]]
    rate_rng[lab] = (min(r["rate_nats"] for r in rs), max(r["rate_nats"] for r in rs))
assert outward(*rate_rng["code10"], 2) == "8.42–8.75" and outward(*rate_rng["prod5"], 2) == "8.01–8.52", rate_rng
h_p5 = pw05s["q1_H_X_bits"]["prod5-gauss_vae-A2-b1-lr0.0003"]["median"] * LN2
assert f"{h_p5:.2f}" == "8.96" and abs(h_p5 - np.log(7776)) < 1e-9, h_p5
BW1_CELLS = [f"bw-gauss_vae-{a}-b1-lr{lr}" for a in archs for lr in ("0.0003", "0.003")]  # network, lower lr first
at_med = [pw05s["q7_au_asperti_trentin"][c]["median"] for c in BW1_CELLS]
burda_med = [pw05s["n_active"][c]["median"] for c in BW1_CELLS]
assert at_med == [15, 15, 15, 9, 14, 13] and burda_med == [16, 15, 15, 10, 15, 15], (at_med, burda_med)
ctrl = [c for c in pw05s["q7_au_asperti_trentin"] if not c.startswith("bw-")]
assert len(ctrl) == 6 and all(pw05s["q7_au_asperti_trentin"][c] == pw05s["n_active"][c] for c in ctrl)
at_shift = max(abs(pw05s["q7_au_asperti_trentin"][c][k] - pw05s["n_active"][c][k]) for c in BW1_CELLS
               for k in ("median", "min", "max"))
assert at_shift <= 2, at_shift
v17.append("### Section 5.1\n\n"
           f"- Rate per run at beta = 1 (pw07 `A_unit_kl.per_run.rate_nats`, outward): code10 "
           f"{outward(*rate_rng['code10'], 2)} ({rate_rng['code10'][0]:.4f}-{rate_rng['code10'][1]:.4f}), prod5 "
           f"{outward(*rate_rng['prod5'], 2)} ({rate_rng['prod5'][0]:.4f}-{rate_rng['prod5'][1]:.4f}); Blocksworld in "
           "the KL table above. prod5 entropy "
           f"{h_p5:.4f} nats = ln 7776 (pw05 `summaries.q1_H_X_bits` x ln 2).\n"
           "- Median counts at beta = 1 under the inactivity threshold of Asperti and Trentin (2020) (post hoc; a unit "
           "is inactive when its mean posterior variance is at least 0.8; pw05 `summaries.q7_au_asperti_trentin`), "
           f"network order, lower lr first: {', '.join(f'{x:g}' for x in at_med)}; with Burda's threshold "
           f"(`summaries.n_active`): {', '.join(f'{x:g}' for x in burda_med)}. Controls identical under both. Largest "
           f"shift of a Blocksworld median or range end: {at_shift:g} units.")

# Section 5.2: 2x256 at the two learning rates, zero rate, critical beta
A2H, A2L = "bw-gauss_vae-A2-b1-lr0.003", "bw-gauss_vae-A2-b1-lr0.0003"
d128 = (pw05s["q2_D_nats"][A2H]["median"], pw05s["q2_D_nats"][A2L]["median"])
assert (f"{d128[0]:.1f}", f"{d128[1]:.1f}") == ("1.2", "1.6"), d128
ne = {c: mc4096["cell_negelbo_nats"][c] for c in (A2H, A2L)}
assert outward(ne[A2H]["min"], ne[A2H]["max"], 1) == "8.9–9.2" and \
    outward(ne[A2L]["min"], ne[A2L]["max"], 1) == "9.4–9.6", ne
assert (f"{ne[A2H]['median']:.1f}", f"{ne[A2L]['median']:.1f}") == ("9.1", "9.4"), ne
d0 = {c: pw05s["q5_D0_nats"][c]["median"] for c in BW1_CELLS}
assert len({round(x, 9) for x in d0.values()}) == 1 and f"{d0[A2L]:.2f}" == "15.66", d0
b4 = max(v["max"] for v in pw05s["q5_beta4_rate_bits"].values()) * LN2
assert b4 < 1e-4, b4
bstar = {}
for lab in ("bw", "code10", "prod5"):
    cs = [c for c in pw05s["q5_beta_star"] if c.startswith(lab + "-")]
    bstar[lab] = (min(pw05s["q5_beta_star"][c]["min"] for c in cs), max(pw05s["q5_beta_star"][c]["max"] for c in cs))
assert outward(*bstar["bw"], 2) == "1.59–1.86" and outward(*bstar["prod5"], 1) == "1.2–1.3", bstar
bstar_c10_med = [pw05s["q5_beta_star"][c]["median"] for c in pw05s["q5_beta_star"] if c.startswith("code10-")]
assert all(f"{x:.1f}" == "2.7" for x in bstar_c10_med), bstar_c10_med  # v1.6: "about 2.7" (cell medians)
v17.append("### Section 5.2\n\n"
           f"- 2x256, higher vs lower lr: median distortion {d128[0]:.4f} vs {d128[1]:.4f} nats (pw05 "
           "`summaries.q2_D_nats`, 128 draws); negative ELBO median (seed range) "
           f"{ne[A2H]['median']:.3f} ({ne[A2H]['min']:.3f}-{ne[A2H]['max']:.3f}) vs {ne[A2L]['median']:.3f} "
           f"({ne[A2L]['min']:.3f}-{ne[A2L]['max']:.3f}), outward {outward(ne[A2H]['min'], ne[A2H]['max'], 1)} vs "
           f"{outward(ne[A2L]['min'], ne[A2L]['max'], 1)} (4,096-draw audit re-derivation, "
           "`experiments/pw05_unit_anatomy/negelbo_4096/cell_negelbo_nats.json`; the "
           "128-draw pw05 ranges round outward to 9.3-9.6 for the lower lr). Rates 7.92 vs 7.84: KL table above.\n"
           f"- Zero rate: factorised-decoder distortion on Blocksworld {d0[A2L]:.4f} nats (pw05 `summaries.q5_D0_nats`); "
           f"largest rate of a beta = 4 run {b4:.2e} nats, below 1e-4 (pw05 `summaries.q5_beta4_rate_bits` x ln 2).\n"
           "- Critical-beta heuristic (post hoc; distortion saved by the code against a decoder that ignores it, divided "
           "by the rate; re-optimisation at the new beta ignored; pw05 `summaries.q5_beta_star`, per run): Blocksworld "
           f"{bstar['bw'][0]:.4f}-{bstar['bw'][1]:.4f} (outward {outward(*bstar['bw'], 2)}), code10 "
           f"{bstar['code10'][0]:.4f}-{bstar['code10'][1]:.4f} (cell medians {min(bstar_c10_med):.2f}-{max(bstar_c10_med):.2f}, "
           f"about 2.7), prod5 {bstar['prod5'][0]:.4f}-"
           f"{bstar['prod5'][1]:.4f} (outward {outward(*bstar['prod5'], 1)}).")

# Section 5.3: index codes
idx_rates = [r["rate_nats"] for lab in ("E10a-L32", "E10b-L32") for r in a7_p6[lab]]
assert outward(min(idx_rates), max(idx_rates), 1) == "1.9–2.3", idx_rates
mf6 = pw06["model_free"]
z_idx = {lab: mf6["H_nats"] + mf6["tc_e10_nats"][k] for lab, k in (("E10a", "0"), ("E10b", "1"))}  # seeds 0 / 1
assert all(f"{z:.2f}" == "6.88" for z in z_idx.values()), z_idx
v17.append("### Section 5.3\n\n"
           f"- Index codes (E10a, E10b): rate per run {min(idx_rates):.4f}-{max(idx_rates):.4f} nats (outward "
           f"{outward(min(idx_rates), max(idx_rates), 1)}; pw07 `A_unit_kl.per_run_pw06.rate_nats`); zero-rate value "
           f"(sum of the per-bit Bernoulli entropies = pw06 `model_free.H_nats` + `tc_e10_nats`) E10a "
           f"{z_idx['E10a']:.4f}, E10b {z_idx['E10b']:.4f} nats. Median negative ELBO and H(X): follow-up table above.")

# Section 5.4: activations
def au_cell(cid):
    rs = sorted((r for r in au_rows if r["cell_id"] == cid), key=lambda r: r["seed"])
    return rs, [r.get("au") for r in rs]


rs025, a025 = au_cell("b_raw_w128-gauss_vae-A2-b0.25-glearned-lr0.0003")
rs4, a4 = au_cell("b_raw_w128-gauss_vae-A2-b4-glearned-lr0.0003")
assert stat(a025) == (62, 61, 63) and not any(r["converged"] for r in rs025), a025
assert stat(a4) == (14, 13, 16), a4
h34 = pw03["secondary"]["h34_activation_substrate_alignment"]
assert f"{h34['vae_mono_b']:.2f}" == "0.14" and round(h34["sae_cov30_b"] * 30) == 17 and round(h34["vae_cov30_b"] * 30) == 3
lin_learned = [r for r in au_rows if r["cell_id"].startswith("b_lin") and "-glearned-" in r["cell_id"]]
viol = [r["asperti_not_converged"] for r in lin_learned]
assert len(viol) == 15 and (min(viol), max(viol)) == (2, 31), viol
pred1 = [pw05_lin[s]["n_above_gamma1"] for s in srcs]
predf = [pw05_lin[s]["n_above_gamma_floor"] for s in srcs]
obs_f = [x for s in srcs for x in pw05_lin[s]["au_fixed"]]
obs_l = [x for s in srcs for x in pw05_lin[s]["au_learned"]]
assert (min(pred1), max(pred1), set(predf)) == (18, 25, {25}), (pred1, predf)
assert (min(obs_f), max(obs_f), min(obs_l), max(obs_l)) == (13, 19, 13, 22), (obs_f, obs_l)
assert all(pw05_lin[s]["au_fixed"] == h24[s]["lin_fixed"]["values"] and
           pw05_lin[s]["au_learned"] == h24[s]["lin_learned"]["values"] for s in srcs)
comp_cells = ["b_raw_w128-gauss_vae-A2-b4-glearned-lr0.0003", "b_raw_w128-tc_vae-A2-b6-glearned-lr0.0003",
              "b_raw_w128-cat_vae-A2-b1-glearned-lr0.0003"]
assert all(r["competence"] for c in comp_cells for r in au_cell(c)[0]) and \
    all(r["converged"] for r in au_cell(comp_cells[2])[0])
rs256f, _ = au_cell("b_raw_w256-gauss_vae-A2-b1-gfixed-lr0.0003")
assert len(rs256f) == 3 and not any(r["competence"] for r in rs256f)
v17.append("### Section 5.4\n\n"
           f"- T128a full activations, learned gamma: {stat(a025)[0]:g} ({min(a025)}-{max(a025)}) active units at "
           f"beta = 0.25 (no run converges), {stat(a4)[0]:g} ({min(a4)}-{max(a4)}) at beta = 4 (pw02a `au_table`).\n"
           f"- T128a full activations: k = 5 SAE coverage {h34['sae_cov30_b']:.4f} (17/30), learned-gamma beta-VAE "
           f"coverage {h34['vae_cov30_b']:.4f} (3/30) and detector fraction {h34['vae_mono_b']:.4f} (pw03a "
           "`secondary.h34_activation_substrate_alignment`). With learned gamma, the beta-VAE at beta = 4, the "
           "beta-TCVAE at beta_TC = 6 and the categorical VAE pass the reconstruction criterion on T128a in all three "
           "seeds (the categorical VAE also converges); on full T256 activations the fixed-gamma beta-VAE fails it in "
           "all three seeds (pw02a `au_table` `competence`, `converged`).\n"
           f"- State-projected activations, learned gamma: units breaking the variance law per run {min(viol)}-{max(viol)} "
           "(pw02a `au_table` `asperti_not_converged`, 15 runs).\n"
           "- Linear-VAE prediction (post hoc; covariance eigenvalues above the collapse threshold beta*gamma; pw05 "
           f"`q4_b_lin`): {min(pred1)}-{max(pred1)} units at gamma = 1 against {min(obs_f)}-{max(obs_f)} observed, "
           f"and the full rank of 25 at the floor against {min(obs_l)}-{max(obs_l)} observed.")

# Section 5.5: positive control, beta = 0.25 coverage, unit-block mutual information
oracle = pw03["metrics"]["per_model"]["control_fact_oracle"]
assert (oracle["cov30"], oracle["n_active"]) == (1.0, 64) and f"{oracle['mono_frac']:.2f}" == "0.56" and \
    f"{oracle['dci']['disentanglement']:.2f}" == "1.00", oracle
c025 = sorted((v["cell_id"], v["seed"]) for v in pm.values() if (v.get("cell_id") or "").startswith("bw-gauss_vae")
              and "-b0.25-" in v["cell_id"] and round(v["cov30"] * 30) >= 25)
assert c025 == [("bw-gauss_vae-A1-b0.25-lr0.003", 4), ("bw-gauss_vae-A2-b0.25-lr0.0003", 1)], c025
nmi = {k: (pw05s["q3"][f"{k}-gauss_vae-A2-b1-lr0.0003"]["max_nmi_median"],
           pw05s["q3"][f"{k}-gauss_vae-A2-b1-lr0.0003"]["second_nmi_median"]) for k in ("prod5", "bw")}
assert [f"{x:.2f}" for x in nmi["prod5"] + nmi["bw"]] == ["0.49", "0.04", "0.33", "0.29"], nmi
v17.append("### Section 5.5\n\n"
           f"- Positive control (the 36 facts plus Gaussian noise of sd 0.1, plus 28 pure-noise coordinates; "
           f"{oracle['n_active']} units, 36 carrying one fact; pw03a `analyze_alignment.py` `controls()`): coverage {oracle['cov30']:.2f} (30/30), detector fraction "
           f"{oracle['mono_frac']:.4f}, DCI {oracle['dci']['disentanglement']:.4f} (pw03a "
           "`metrics.per_model.control_fact_oracle`).\n"
           "- Raw states at beta = 0.25, runs reaching coverage 25/30: "
           + "; ".join(f"{c} seed {s}" for c, s in c025) + " (pw03a `metrics.per_model.*.cov30`).\n"
           "- Unit-block normalised mutual information at 2x256 (post hoc; median over units, best and second-best "
           f"block; pw05 `summaries.q3.*.max_nmi_median`, `second_nmi_median`): prod5 {nmi['prod5'][0]:.4f}, "
           f"{nmi['prod5'][1]:.4f}; Blocksworld {nmi['bw'][0]:.4f}, {nmi['bw'][1]:.4f}.\n"
           "- Figure 5 panel (b) of v1.6 (prod5 seed 1 unit anatomy) is no longer drawn; its seed choice and two-fact "
           "totals stay asserted in make_figures.py.")

# Section 5.6: beta-TCVAE legality (the values are in the legality tables above; asserted here as v1.6 printed them)
tcd = [lp[k] for k in lp if lp[k]["cell_id"] == "bw-tc_vae-A2-b2-lr0.0003"]
tc_dec = (min(v["prior_mode"]["rate"] for v in tcd), max(v["prior_mode"]["rate"] for v in tcd))
tc_agg = (min(v["aggpost_mode"]["rate"] for v in tcd), max(v["aggpost_mode"]["rate"] for v in tcd))
tc_eq = rng(tcr, "prior", "legal_rederived")
assert (outward(*tc_dec, 3), outward(*tc_agg, 3), outward(*tc_eq, 2)) == \
    ("0.454–0.464", "0.738–0.754", "0.62–0.63"), (tc_dec, tc_agg, tc_eq)
assert tc_dec[1] < B2_LINE < tc_eq[0] and tc_agg[1] < 0.95, (tc_dec, tc_eq, tc_agg)
v17.append("### Section 5.6\n\n"
           f"- beta-TCVAE (beta_TC = 2, 3 seeds): prior legality as decoded {outward(*tc_dec, 3)}, on equal terms "
           f"{outward(*tc_eq, 2)}, aggregate posterior as decoded {outward(*tc_agg, 3)}: below the baseline as decoded, "
           "above it on equal terms, below the legality criterion in all three seeds (pw04a descriptive pass; pw07 B).")

# Moved from the prose in the v1.7 final edit (30 October 2026): secondary counts of Sections 5.5 and 5.6
q3_types = load(EXP / "pw05_unit_anatomy/presentation/q3_pairs.json")["pair_types_per_cell"]
ref_t = q3_types["bw-gauss_vae-A2-b1-lr0.0003"]
on_hold = {k: v for k, v in ref_t.items() if k.startswith("on(X,Y)|holding")}
assert (sum(ref_t.values()), ref_t["on(X,Y)|on(Y,X)"], ref_t["holding|holding"], sum(on_hold.values()),
        ref_t["on(X,Y)|holding(Z)"]) == (55, 44, 6, 5, 2), ref_t
p5_t = {c: v for c, v in q3_types.items() if c.startswith("prod5-")}
p5_cross = {c: v.get("on(X,Y)|holding(Z)", 0) for c, v in p5_t.items()}
assert sum(sum(v.values()) for v in p5_t.values()) == 74 and sum(p5_cross.values()) == 3 == \
    p5_cross["prod5-gauss_vae-A4-b1-lr0.0003"], p5_t
assert all(k in ("on(X,Y)|on(X,Z)", "on(X,Y)|on_table(X)", "on(X,Y)|holding(X)", "holding|on_table",
                 "on(X,Y)|holding(Z)") for v in p5_t.values() for k in v), p5_t
assert cnt["beta025"]["prior_below_rederived"] == 11 and cnt["beta025"]["n_runs"] == 30, cnt["beta025"]
v17.append("### Moved from the prose in the v1.7 final edit\n\n"
           "- Reference configuration, two-fact units over five seeds by pair type (pw05 presentation "
           "`q3_pairs.json` `pair_types_per_cell`): on(X,Y)|on(Y,X) "
           f"{ref_t['on(X,Y)|on(Y,X)']}; two holding facts {ref_t['holding|holding']}; an on fact with a holding fact "
           f"{sum(on_hold.values())} (" + ", ".join(f"{k} {v}" for k, v in sorted(on_hold.items())) + "); total "
           f"{sum(ref_t.values())}. The {ref_t['on(X,Y)|holding(Z)']} units pairing on(X,Y) with a third block's "
           "holding are the two whose facts can co-occur; all other pairs are mutually exclusive.\n"
           f"- prod5, two-fact units over the three networks: {sum(sum(v.values()) for v in p5_t.values())}; the "
           f"{sum(p5_cross.values())} that do not pair positions of one block are at 2x1024 and pair on(X,Y) with a "
           "third block's holding (same file).\n"
           "- beta = 0.25, prior samples on equal terms below the per-block baseline: "
           f"{cnt['beta025']['prior_below_rederived']} of {cnt['beta025']['n_runs']} runs (pw07 "
           "`B_matched_legality.counts.beta025.prior_below_rederived`).")

v17.append("### Quoted from prose records (not recomputed by this script)\n\n"
           "- `experiments/pw05_unit_anatomy/README.md`, Corrections (2026-10-27), Q2: \"Two independent 4,096-draw "
           "estimates give −ELBO 9.125 (8.984–9.157) for A2 lr 3e-3 against 9.436 (9.406–9.510) for A2 "
           "lr 3e-4, and D 1.17 against 1.58.\"\n"
           "- `docs/EXPERIMENT_LOG.md`, pw02 entry: \"Split by sequence (train 12,803 / val 3,201 from the thesis train "
           "file; thesis test file sealed).\"\n"
           "- The status of each planned analysis is in Appendix A of the report (planned predictions and their "
           "outcomes) and in each `experiments/pwNN_*/README.md`.")
tables.append("## Supporting numbers that left the PDF in report v1.7 (values printed in v1.6, read from JSON and "
              "asserted)\n\n" + "\n\n".join(v17))

(OUT / "tables.md").write_text("\n\n".join(tables) + "\n")
print("Figure 5 representative seeds: Blocksworld two-fact counts", bw_n2, "median", bw_med, "-> seed", bw_run["seed"],
      "| prod5", p5_n2, "median", p5_med, "-> seed", p5_run["seed"])
print("Figure 4 plotted values (label, seed, count, exact, bitwise, recon_ok):")
for row in fig3_rows:
    print("  ", row)
print("Figure 4 exact ranges", fig3_ranges)
print("Figure 3 (run, stored AU, count > 0.01, max |sum - 1|, min active Var mu, max inactive Var mu):")
for row in fig4_rows:
    print("  ", row)
print("Figure 3 plotted seeds", [(n, st[0]) for n, _, st in fig4_plot])
print("Figure 6 (cell, seed, prior as decoded, prior on equal terms, aggregate posterior):")
for row in fig5_rows:
    print("  ", row)
print("Figure 5 labels", labels_bw)
fig6_examples()
print("wrote", sorted(p.name for p in OUT.iterdir()))
