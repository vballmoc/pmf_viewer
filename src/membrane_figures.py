"""Drawing and plot helpers for membrane_app.py, so all tabs share one look.

Colors follow the subject: K⁺ is lilac (potassium flame test), H⁺ amber,
pH teal, Δψ dark ink.
"""
from io import StringIO

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np

K_COLOR = "#8E5BB5"
K_LIGHT = "#E9DDF3"
H_COLOR = "#C77D12"
H_LIGHT = "#F6E7CF"
PH_COLOR = "#2A7F7F"
INK = "#22303C"
MUTED = "#66727D"
GRID = "#E3E7EB"
ALERT = "#B3412E"

CSS = f"""
<style>
  h1 {{ color: {INK}; font-weight: 700; letter-spacing: -0.01em; }}
  h2, h3, h4 {{ color: {INK}; }}
  .lede {{ color: {MUTED}; font-size: 1.05rem; max-width: 48rem; margin-top: -0.6rem; }}
  .question {{ font-size: 1.55rem; font-weight: 700; color: {INK}; line-height: 1.25;
              margin: 0.2rem 0 0.8rem 0; max-width: 40rem; }}
  .result {{ font-size: 1.02rem; line-height: 1.55; margin: 0 0 0.8rem 0; max-width: 36rem; }}
  .result b {{ font-size: 1.3rem; color: {INK}; }}
  .k {{ color: {K_COLOR}; font-weight: 600; }}
  .h {{ color: {H_COLOR}; font-weight: 600; }}
  .ph {{ color: {PH_COLOR}; font-weight: 600; }}
  .group {{ font-weight: 700; color: {INK}; margin: 0.9rem 0 0.1rem 0; }}
</style>
"""


# ------------------------------------------------------------
# Number formatting
# ------------------------------------------------------------
def fmt_mV(v):
    """Signed value with a real minus sign; tiny values shown as 0."""
    if not np.isfinite(v):
        return "–"
    if abs(v) < 0.5:
        return "0 mV"
    return f"{v:+.0f} mV".replace("-", "−")


def fmt_signed(v, digits=2):
    return f"{v:+.{digits}f}".replace("-", "−")


def fmt_count(x):
    return f"{x:,.0f}".replace(",", "'") if x >= 1 else f"{x:.2g}"


def fmt_time(s):
    if s < 0.1:
        return f"{s * 1000:.0f} ms"
    if s < 120:
        return f"{s:.1f} s"
    if s < 7200:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.1f} h"


# ------------------------------------------------------------
# Vesicle drawing
# ------------------------------------------------------------
def draw_vesicle(ax, x0, title, lines, psi_mV, arrows, fill=K_LIGHT, fill_level=0.5):
    """One vesicle.

    lines: list of (text, color) written inside, top to bottom.
    arrows: list of (flux, color, label); flux > 0 points into the vesicle, 0 = no arrow.
    """
    ax.add_patch(Circle((x0, 0), 1.0, facecolor=fill, edgecolor="none",
                        alpha=0.2 + 0.8 * float(np.clip(fill_level, 0, 1))))
    ax.add_patch(Circle((x0, 0), 1.0, facecolor="none", edgecolor=INK, linewidth=3.2))
    ax.text(x0, 1.42, title, ha="center", va="bottom", fontsize=12, color=INK, fontweight="bold")

    ys = np.linspace(0.32, -0.32, len(lines)) if len(lines) > 1 else [0.0]
    for (text, color), y in zip(lines, ys):
        ax.text(x0, y, text, ha="center", va="center", fontsize=11, color=color,
                fontweight="bold" if color != INK else "normal")

    if np.isfinite(psi_mV) and abs(psi_mV) >= 0.5:
        sign = "−" if psi_mV < 0 else "+"
        for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            ax.text(x0 + 0.84 * np.cos(a), 0.84 * np.sin(a), sign, ha="center", va="center",
                    fontsize=9, color=MUTED)

    slots = [0.45, -0.45, 0.0][:len(arrows)]
    for (flux, color, label), y in zip(arrows, slots):
        if not np.isfinite(flux) or abs(flux) < 1e-9:
            continue
        x_in, x_out = x0 + 0.62, x0 + 1.42
        start, end = ((x_out, y), (x_in, y)) if flux > 0 else ((x_in, y), (x_out, y))
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16,
                                     color=color, linewidth=2.2))
        ax.text(x_out + 0.08, y, label, ha="left", va="center", fontsize=11, color=color,
                fontweight="bold")


def vesicle_pair(left, right, footer, footer_color=MUTED):
    """Two vesicles side by side (e.g. start and end). left/right: dicts for draw_vesicle."""
    fig, ax = plt.subplots(figsize=(7.4, 3.1))
    draw_vesicle(ax, 0, **left)
    draw_vesicle(ax, 4.4, **right)
    ax.text(2.2, -1.38, footer, ha="center", fontsize=10.5, color=footer_color)
    ax.set_xlim(-1.6, 7.3)
    ax.set_ylim(-1.6, 1.8)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    return fig


# ------------------------------------------------------------
# Plots
# ------------------------------------------------------------
def new_axes(ylabel, x_max, figsize=(6.4, 3.4), xlabel="Time / s"):
    fig, ax = plt.subplots(figsize=figsize)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9.5)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=INK, fontsize=10.5)
    ax.set_ylabel(ylabel, color=INK, fontsize=10.5)
    ax.set_xlim(0, x_max)
    fig.tight_layout()
    return fig, ax


def label_line(ax, y, text, color, x_frac=0.99, va="bottom"):
    """Small text label sitting on a horizontal reference line."""
    ax.text(x_frac, y, text + " ", transform=ax.get_yaxis_transform(), ha="right", va=va,
            fontsize=9, color=color)


def finish(fig, ax, legend=False):
    if legend:
        ax.legend(frameon=False, fontsize=9.5)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------
# CSV export with the run's parameters as header lines
# ------------------------------------------------------------
def csv_with_header(title, params, df):
    lines = [f"# {title}",
             "# Lines starting with # are parameters. Read in Python with pd.read_csv(file, comment='#')"]
    lines += [f"# {key} = {value}" for key, value in params.items()]
    out = StringIO()
    out.write("\n".join(lines) + "\n")
    df.to_csv(out, index=False)
    return out.getvalue()
