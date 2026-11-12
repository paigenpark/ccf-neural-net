"""CCF-Net architecture figure.

Run:  .venv/bin/python diagram/ccfnet_architecture.py
Writes figures/ccfnet_architecture.pdf (vector, fonts embedded), .svg and .png
(600 dpi), from any working directory.

Everything that is specific to the model is in the CONFIG block, so a change
to the selected configuration only needs an edit there.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

# ----------------------------------------------------------------- CONFIG
EMBED_DIM = 5
N_HIDDEN_BLOCKS = 4
HIDDEN_UNITS = 110
N_MEMBERS = 5
EPOCHS_PER_STAGE = 50
YEAR_ANCHOR = "1950–2015"
STAGE1_LOSS = ""  # e.g. "mean squared error"; left blank because not stated
# ------------------------------------------------------------------------

CONCAT_DIM = 1 + 2 * EMBED_DIM

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "stixsans",
    "font.size": 7.5,
    "pdf.fonttype": 42,   # embed TrueType fonts (journal-friendly)
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

INK = "#1a1a1a"
MUTED = "#606060"
S1_FILL, S1_EDGE = "#dcdcdc", "#4a4a4a"   # trained in stage 1
S2_FILL, S2_EDGE = "#a6a6a6", "#1a1a1a"   # trained in stage 2
OP_FILL, OP_EDGE = "#ffffff", "#4a4a4a"   # no trainable parameters
FRAME_FILL, FRAME_EDGE = "#f5f5f5", "#8a8a8a"

W, H = 7.38, 3.42
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

LINE = 0.138  # line pitch in inches for 7.5 pt text


def box(x, y, w, h, lines, fill, edge, z=3, lw=0.9, rot=0, fs=7.5):
    """Rounded box with centred text. `lines` is a list of (text, weight)."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.045",
        fc=fill, ec=edge, lw=lw, zorder=z))
    cx, cy = x + w / 2, y + h / 2
    n = len(lines)
    for i, (txt, weight) in enumerate(lines):
        off = ((n - 1) / 2 - i) * LINE
        if rot:
            ax.text(cx - off, cy, txt, ha="center", va="center", rotation=rot,
                    fontsize=fs, fontweight=weight, color=INK, zorder=z + 1)
        else:
            ax.text(cx, cy + off, txt, ha="center", va="center",
                    fontsize=fs, fontweight=weight, color=INK, zorder=z + 1)


def arrow(x0, y0, x1, y1, z=6):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=z,
                arrowprops=dict(arrowstyle="-|>", lw=0.85, color=INK,
                                shrinkA=0, shrinkB=0, mutation_scale=7))


def line(xs, ys, z=6):
    ax.plot(xs, ys, color=INK, lw=0.85, solid_capstyle="round",
            solid_joinstyle="round", zorder=z)


def dim(x, y, txt, ha="center"):
    ax.text(x, y, txt, ha=ha, va="bottom", fontsize=6.3, color=MUTED,
            fontstyle="italic", zorder=7)


# ------------------------------------------------- ensemble-member frame
FX0, FX1, FY0, FY1 = 0.10, 6.08, 0.93, 3.16
for k in (2, 1):
    ax.add_patch(FancyBboxPatch(
        (FX0 + 0.05 * k, FY0 + 0.05 * k), FX1 - FX0, FY1 - FY0,
        boxstyle="round,pad=0,rounding_size=0.07",
        fc=FRAME_FILL, ec=FRAME_EDGE, lw=0.7, zorder=0.2 - 0.05 * k))
ax.add_patch(FancyBboxPatch(
    (FX0, FY0), FX1 - FX0, FY1 - FY0,
    boxstyle="round,pad=0,rounding_size=0.07",
    fc=FRAME_FILL, ec=FRAME_EDGE, lw=0.7, zorder=0.3))
ax.text(FX0 + 0.12, FY1 - 0.15,
        rf"Ensemble member $m$  ($m = 1,\ldots,{N_MEMBERS}$; "
        "separate training runs)",
        ha="left", va="center", fontsize=7.5, fontweight="bold", color=INK)

# -------------------------------------------------------------- inputs
ROWS = {"year": 2.52, "age": 1.92, "country": 1.32}
MID = ROWS["age"]
IN_X = 0.68
EX, EW, EH = 0.84, 1.28, 0.42

ax.text(IN_X - 0.08, ROWS["year"], "Year", ha="right", va="center")
ax.text(IN_X - 0.08, ROWS["age"], "Age", ha="right", va="center")
ax.text(IN_X - 0.08, ROWS["country"], "Country", ha="right", va="center")
ax.text(0.22, 2.86, "One ASFR cell", ha="left", va="center",
        fontsize=6.5, color=MUTED)

box(EX, ROWS["year"] - EH / 2, EW, EH,
    [("Min–max scaling", "normal"),
     (f"anchored to {YEAR_ANCHOR}", "normal")], OP_FILL, OP_EDGE)
for key in ("age", "country"):
    box(EX, ROWS[key] - EH / 2, EW, EH,
        [("Embedding", "normal"), (f"dimension {EMBED_DIM}", "normal")],
        S1_FILL, S1_EDGE)

for key in ROWS:
    arrow(IN_X, ROWS[key], EX, ROWS[key])

# -------------------------------------------------------- concatenate
CX, CW = 2.36, 0.27
CY0, CY1 = ROWS["country"] - 0.26, ROWS["year"] + 0.26
box(CX, CY0, CW, CY1 - CY0, [("Concatenate", "normal")],
    OP_FILL, OP_EDGE, rot=90)
for key, d in (("year", 1), ("age", EMBED_DIM), ("country", EMBED_DIM)):
    arrow(EX + EW, ROWS[key], CX, ROWS[key])
    dim((EX + EW + CX) / 2 - 0.01, ROWS[key] + 0.025, str(d))

# ------------------------------------------------------- hidden blocks
HX, HW, HH = 2.91, 1.22, 0.92
HY = MID - HH / 2
for k in range(N_HIDDEN_BLOCKS - 1, 0, -1):
    ax.add_patch(FancyBboxPatch(
        (HX + 0.04 * k, HY + 0.045 * k), HW, HH,
        boxstyle="round,pad=0,rounding_size=0.045",
        fc=S1_FILL, ec=S1_EDGE, lw=0.7, zorder=2 - 0.1 * k))
box(HX, HY, HW, HH,
    [("Hidden block", "bold"),
     (f"Dense, {HIDDEN_UNITS} units", "normal"),
     ("ReLU", "normal"),
     ("Layer normalization", "normal")], S1_FILL, S1_EDGE)
ax.text(HX + HW / 2, HY - 0.12, f"× {N_HIDDEN_BLOCKS} in sequence",
        ha="center", va="center", fontsize=7.0, color=INK)

arrow(CX + CW, MID, HX, MID)
dim((CX + CW + HX) / 2 - 0.01, MID + 0.025, str(CONCAT_DIM))

# --------------------------------------------------------------- heads
HDX, HDW, HDH = 4.70, 1.00, 0.60
Y_MEAN, Y_VAR = 2.50, 1.34
FORK = 4.52
box(HDX, Y_MEAN - HDH / 2, HDW, HDH,
    [("Mean head", "bold"), ("Dense, 1 unit", "normal"),
     ("sigmoid", "normal")], S1_FILL, S1_EDGE)
box(HDX, Y_VAR - HDH / 2, HDW, HDH,
    [("Variance head", "bold"), ("Dense, 1 unit", "normal"),
     ("softplus", "normal")], S2_FILL, S2_EDGE)

line([HX + HW, FORK], [MID, MID])
line([FORK, FORK], [Y_VAR, Y_MEAN])
arrow(FORK, Y_MEAN, HDX, Y_MEAN)
arrow(FORK, Y_VAR, HDX, Y_VAR)
dim((HX + HW + 0.04 * (N_HIDDEN_BLOCKS - 1) + FORK) / 2, MID + 0.025,
    str(HIDDEN_UNITS))

# --------------------------------------------------- ensemble averaging
AX_, AW = 6.34, 0.27
box(AX_, CY0, AW, CY1 - CY0,
    [(f"Combined over {N_MEMBERS} members", "normal")],
    OP_FILL, OP_EDGE, rot=90)
arrow(HDX + HDW, Y_MEAN, AX_, Y_MEAN)
arrow(HDX + HDW, Y_VAR, AX_, Y_VAR)
ax.text(HDX + HDW + 0.07, Y_MEAN + 0.045, r"$\hat{\mu}_m$",
        ha="left", va="bottom", fontsize=7.5, color=INK, zorder=7)
ax.text(HDX + HDW + 0.07, Y_VAR + 0.045, r"$\hat{\sigma}^{2}_m$",
        ha="left", va="bottom", fontsize=7.5, color=INK, zorder=7)

OX = AX_ + AW + 0.20
arrow(AX_ + AW, Y_MEAN, OX, Y_MEAN)
arrow(AX_ + AW, Y_VAR, OX, Y_VAR)
ax.text(OX + 0.06, Y_MEAN, r"$\bar{\mu}$", ha="left", va="center",
        fontsize=9.5, color=INK)
ax.text(OX + 0.06, Y_VAR, r"$\bar{\sigma}^{2}$", ha="left", va="center",
        fontsize=9.5, color=INK)
ax.text(OX + 0.06, Y_MEAN - 0.17, "rate", ha="left", va="center",
        fontsize=6.5, color=MUTED)
ax.text(OX + 0.06, Y_VAR - 0.17, "variance", ha="left", va="center",
        fontsize=6.5, color=MUTED)

# -------------------------------------------------------------- legend
LX, SW, SH = 0.12, 0.17, 0.115
s1 = (f"Stage 1 ({EPOCHS_PER_STAGE} epochs): embeddings, hidden blocks "
      "and mean head trained together")
if STAGE1_LOSS:
    s1 += f" ({STAGE1_LOSS})"
s2 = (f"Stage 2 ({EPOCHS_PER_STAGE} epochs): all stage 1 weights frozen; "
      "variance head trained with Gaussian negative log-likelihood")
rows = [(0.66, S1_FILL, S1_EDGE, s1),
        (0.43, S2_FILL, S2_EDGE, s2),
        (0.20, OP_FILL, OP_EDGE,
         "No trainable parameters.")]
for y, fc, ec, txt in rows:
    ax.add_patch(Rectangle((LX, y - SH / 2), SW, SH, fc=fc, ec=ec, lw=0.8))
    ax.text(LX + SW + 0.08, y, txt, ha="left", va="center", fontsize=7.0,
            color=INK)

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 600})):
    fig.savefig(os.path.join(FIG_DIR, f"ccfnet_architecture.{ext}"), facecolor="white", **kw)
