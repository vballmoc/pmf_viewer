"""K⁺ gradient as a battery: valinomycin clamps Δψ to the K⁺ Nernst potential.

Redesigned version of valino1_app.py with the same model:
- Valinomycin is fast: at every moment Δψ equals the K⁺ Nernst potential.
- Outside [K⁺] is fixed (large reservoir); only the K⁺ inside the vesicle changes.
- An imposed positive-charge flux (the "load") is compensated 1:1 by K⁺ moving the other way.

Sign convention: Δψ = ψ(inside) − ψ(outside); fluxes are positive INTO the vesicle.
"""
from io import StringIO

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np
import pandas as pd
import streamlit as st

from version_info import show_version

st.set_page_config(page_title="K⁺ gradient as a battery", layout="wide")
show_version(__file__)  # version line in the sidebar (see version_info.py)

R = 8.314462618       # J mol⁻¹ K⁻¹
F = 96485.33212       # C mol⁻¹
NA = 6.02214076e23    # mol⁻¹

# Colors: potassium burns lilac in the flame test, so K⁺ is lilac everywhere.
K_COLOR = "#8E5BB5"
K_LIGHT = "#E9DDF3"
CHARGE_COLOR = "#C77D12"
INK = "#22303C"
MUTED = "#66727D"
GRID = "#E3E7EB"

st.markdown(f"""
<style>
  h1 {{ color: {INK}; font-weight: 700; letter-spacing: -0.01em; }}
  h2, h3 {{ color: {INK}; }}
  .lede {{ color: {MUTED}; font-size: 1.1rem; max-width: 46rem; margin-top: -0.6rem; }}
  .result {{ font-size: 1.05rem; line-height: 1.55; margin: 0 0 0.9rem 0; max-width: 34rem; }}
  .result b {{ font-size: 1.45rem; color: {INK}; }}
  .k {{ color: {K_COLOR}; font-weight: 600; }}
  .charge {{ color: {CHARGE_COLOR}; font-weight: 600; }}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Inputs
# ------------------------------------------------------------
LOAD_DIRECTIONS = {
    "Discharge the gradient": "dissipate",
    "Push positive charge into the vesicle": "in",
    "Push positive charge out of the vesicle": "out",
}
LOAD_MODELS = {
    "Slows down as Δψ falls": "potential",
    "Constant rate": "constant",
}

with st.sidebar:
    st.header("Vesicle")
    outer_diameter_nm = st.number_input(
        "Outer diameter / nm", min_value=5.0, max_value=5000.0, value=100.0, step=10.0,
        help="Typical liposomes are 50–200 nm. The volume inside is calculated "
             "from the diameter minus two bilayer thicknesses.")
    bilayer_thickness_nm = st.number_input(
        "Bilayer thickness / nm", min_value=0.0, max_value=20.0, value=4.5, step=0.1)
    temperature_C = st.number_input(
        "Temperature / °C", min_value=0.0, max_value=80.0, value=25.0, step=1.0)

    st.header("K⁺ gradient")
    K_in_initial_mM = st.number_input(
        "[K⁺] inside at the start / mM", min_value=0.001, max_value=2000.0, value=50.0, step=5.0,
        help="This internal K⁺ pool is the battery. It changes during the run.")
    K_out_mM = st.number_input(
        "[K⁺] outside / mM", min_value=0.001, max_value=2000.0, value=5.0, step=1.0,
        help="The outside is treated as a large reservoir, so this stays constant.")

    st.header("Load")
    load_direction_label = st.radio(
        "What the load does", list(LOAD_DIRECTIONS),
        help="'Discharge' moves positive charge in the direction that weakens Δψ "
             "(into the vesicle if the inside is negative). The other two options "
             "push charge in one fixed direction, whatever Δψ is.")
    load_model_label = st.radio(
        "How fast", list(LOAD_MODELS),
        help="'Slows down as Δψ falls': the rate is proportional to |Δψ|, like a leak. "
             "'Constant rate': the same number of charges per second throughout.")
    load_rate = st.number_input(
        "Load at the start / charges per second", min_value=0.0, max_value=1_000_000.0,
        value=1500.0, step=100.0)
    load_start_s = st.number_input(
        "Load switched on after / s", min_value=0.0, max_value=100000.0, value=0.0, step=1.0)

    st.header("Time and threshold")
    duration_s = st.number_input(
        "Simulated time / s", min_value=0.1, max_value=100000.0, value=60.0, step=10.0)
    threshold_mV = st.number_input(
        "Useful Δψ at least / mV", min_value=0.0, max_value=500.0, value=50.0, step=5.0,
        help="Below this value, Δψ is considered too weak to be useful, e.g. to drive "
             "a transporter. Used to calculate how long the gradient lasts.")

load_direction = LOAD_DIRECTIONS[load_direction_label]
load_model = LOAD_MODELS[load_model_label]


# ------------------------------------------------------------
# Model (same algorithm as valino1_app.py)
# ------------------------------------------------------------
@st.cache_data(max_entries=50)
def simulate(outer_diameter_nm, bilayer_thickness_nm, temperature_C, K_in_initial_mM,
             K_out_mM, load_direction, load_model, load_rate, load_start_s, duration_s):
    T = temperature_C + 273.15
    inner_diameter_nm = max(outer_diameter_nm - 2 * bilayer_thickness_nm, 0.001)
    volume_L = (4 / 3) * np.pi * (inner_diameter_nm / 2 * 1e-9) ** 3 * 1000

    def nernst_mV(K_in_mM):
        return (R * T / F) * np.log(max(K_out_mM, 1e-30) / max(K_in_mM, 1e-30)) * 1000

    def conc_mM(count):
        return count / NA / volume_L * 1000

    K_count = K_in_initial_mM * 1e-3 * volume_L * NA
    K_count_start = K_count
    K_count_equilibrium = K_out_mM * 1e-3 * volume_L * NA
    psi_start = nernst_mV(K_in_initial_mM)

    n = max(1000, min(30000, int(duration_s * 20)))
    time = np.linspace(0, duration_s, n)
    dt = time[1] - time[0]
    K_in_mM = np.zeros(n)
    psi_mV = np.zeros(n)
    K_ions = np.zeros(n)
    charge_flux_in = np.zeros(n)
    K_flux_in = np.zeros(n)

    for i in range(n):
        K_in_mM[i] = conc_mM(K_count)
        psi_mV[i] = nernst_mV(K_in_mM[i])
        K_ions[i] = K_count
        if i == n - 1:
            break

        if time[i] < load_start_s:
            J = 0.0
        elif load_model == "constant":
            J = load_rate
        else:
            J = load_rate * abs(psi_mV[i]) / abs(psi_start) if abs(psi_start) > 1e-12 else 0.0

        if load_direction == "dissipate":
            J_charge = -np.sign(psi_mV[i]) * J
        elif load_direction == "in":
            J_charge = J
        else:
            J_charge = -J

        J_K = -J_charge                       # valinomycin: one K⁺ the other way
        if J_K < 0:
            J_K = max(J_K, -K_count / dt)     # cannot remove more K⁺ than is inside
        if load_direction == "dissipate":     # discharge stops at [K⁺]in = [K⁺]out
            if psi_start < 0:
                J_K = max(J_K, -max((K_count - K_count_equilibrium) / dt, 0.0))
            elif psi_start > 0:
                J_K = min(J_K, max((K_count_equilibrium - K_count) / dt, 0.0))
            else:
                J_K = 0.0

        K_count = max(K_count + J_K * dt, 1e-30)
        charge_flux_in[i] = J_charge
        K_flux_in[i] = J_K

    charge_flux_in[-1] = charge_flux_in[-2]
    K_flux_in[-1] = K_flux_in[-2]

    # Internal K⁺ used up: the Nernst clamp no longer holds, so Δψ is not defined here
    used_up = K_ions <= 1e-9 * K_count_start
    t_used_up = time[np.argmax(used_up)] if used_up.any() else None
    psi_mV[used_up] = np.nan

    return dict(time=time, psi_mV=psi_mV, K_in_mM=K_in_mM, K_ions=K_ions,
                charge_flux_in=charge_flux_in, K_flux_in=K_flux_in,
                inner_diameter_nm=inner_diameter_nm, volume_L=volume_L,
                K_count_start=K_count_start, K_count_equilibrium=K_count_equilibrium,
                psi_start=psi_start, t_used_up=t_used_up)


sim = simulate(outer_diameter_nm, bilayer_thickness_nm, temperature_C, K_in_initial_mM,
               K_out_mM, load_direction, load_model, load_rate, load_start_s, duration_s)
time, psi_mV, K_in_mM, K_ions = sim["time"], sim["psi_mV"], sim["K_in_mM"], sim["K_ions"]
psi_start = sim["psi_start"]
psi_end = psi_mV[-1]

# How long Δψ stays at or above the useful threshold after the load starts
after_load = time >= load_start_s
useful = np.zeros_like(time, dtype=bool)
finite = np.isfinite(psi_mV)
useful[finite] = np.abs(psi_mV[finite]) >= threshold_mV
useful_after_load = useful & after_load
still_useful_at_end = bool(useful[-1])
if not useful_after_load.any():
    lifetime_s = None
else:
    lifetime_s = time[np.flatnonzero(useful_after_load)[-1]] - load_start_s


def fmt_mV(v):
    """Signed value with a real minus sign; tiny values shown as 0."""
    if abs(v) < 0.5:
        return "0 mV"
    return f"{v:+.0f} mV".replace("-", "−")


def fmt_count(x):
    return f"{x:,.0f}".replace(",", "'") if x >= 1 else f"{x:.2g}"


def fmt_time(s):
    if s < 120:
        return f"{s:.1f} s"
    if s < 7200:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.1f} h"


# ------------------------------------------------------------
# Schematic: the vesicle at the start and at the end
# ------------------------------------------------------------
def draw_vesicle(ax, x0, title, K_in, psi, K_flux, charge_flux, k_fill):
    ax.add_patch(Circle((x0, 0), 1.0, facecolor=K_LIGHT, edgecolor="none", alpha=0.25 + 0.75 * k_fill))
    ax.add_patch(Circle((x0, 0), 1.0, facecolor="none", edgecolor=INK, linewidth=3.2))
    ax.text(x0, 1.42, title, ha="center", va="bottom", fontsize=12, color=INK, fontweight="bold")
    K_text = f"{K_in:.3g} mM" if np.isfinite(psi) else "≈ 0 mM"
    ax.text(x0, 0.2, f"[K⁺] {K_text}", ha="center", va="center", fontsize=11, color=K_COLOR,
            fontweight="bold")
    if np.isfinite(psi):
        ax.text(x0, -0.22, f"Δψ {fmt_mV(psi)}", ha="center", va="center", fontsize=11, color=INK)
        # charge layer on the inner side of the membrane
        sign = "−" if psi < -0.5 else "+" if psi > 0.5 else ""
        if sign:
            for a in np.linspace(0, 2 * np.pi, 10, endpoint=False):
                ax.text(x0 + 0.8 * np.cos(a), 0.8 * np.sin(a), sign, ha="center", va="center",
                        fontsize=10, color=MUTED)
    else:
        ax.text(x0, -0.22, "K⁺ used up", ha="center", va="center", fontsize=11, color="#B3412E")

    # arrows through the membrane at the right side: K⁺ (upper) and imposed charge (lower)
    for y, flux, color, label in [(0.38, K_flux, K_COLOR, "K⁺"), (-0.38, charge_flux, CHARGE_COLOR, "+ charge")]:
        if abs(flux) < 1e-9:
            continue
        x_in, x_out = x0 + 0.62, x0 + 1.42
        start, end = ((x_out, y), (x_in, y)) if flux > 0 else ((x_in, y), (x_out, y))
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16,
                                     color=color, linewidth=2.2))
        ax.text(x_out + 0.08, y, label, ha="left", va="center", fontsize=11, color=color, fontweight="bold")


def schematic():
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    k_max = max(K_in_mM.max(), K_out_mM)
    i_load = int(np.argmax(time >= load_start_s))
    # use the first moment with load to show the direction of the flows
    draw_vesicle(ax, 0, "Start", K_in_mM[0], psi_start,
                 sim["K_flux_in"][i_load], sim["charge_flux_in"][i_load], K_in_mM[0] / k_max)
    draw_vesicle(ax, 4.2, f"After {fmt_time(time[-1])}", K_in_mM[-1], psi_end,
                 sim["K_flux_in"][-2], sim["charge_flux_in"][-2], K_in_mM[-1] / k_max)
    ax.text(2.1, -1.35, f"outside: [K⁺] {K_out_mM:.3g} mM, constant", ha="center", fontsize=10.5,
            color=K_COLOR)
    ax.set_xlim(-1.6, 7.0)
    ax.set_ylim(-1.6, 1.8)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    return fig


# ------------------------------------------------------------
# Header and key results
# ------------------------------------------------------------
st.title("How long can a K⁺ gradient hold a membrane potential?")
st.markdown(
    "<p class='lede'>With valinomycin, a vesicle with a K⁺ gradient behaves like a small battery: "
    "Δψ equals the K⁺ Nernst potential. A load that moves charge across the membrane "
    "is paid for with K⁺ ions, until the gradient is spent.</p>",
    unsafe_allow_html=True)

left, right = st.columns([1.25, 1], gap="large")
with left:
    st.pyplot(schematic())

with right:
    st.markdown(
        f"<p class='result'>Δψ starts at <b>{fmt_mV(psi_start)}</b>, set by "
        f"<span class='k'>{K_in_initial_mM:.3g} mM K⁺ inside</span> and "
        f"<span class='k'>{K_out_mM:.3g} mM outside</span>.</p>",
        unsafe_allow_html=True)

    if lifetime_s is None:
        text = (f"Δψ is never at least <b>{threshold_mV:.0f} mV</b> while the load runs, "
                "so this gradient is too weak to be useful from the beginning.")
    elif still_useful_at_end:
        text = (f"Δψ is still at least {threshold_mV:.0f} mV at the end: it lasts "
                f"<b>longer than {fmt_time(time[-1] - load_start_s)}</b>. "
                "Increase the simulated time to see when it drops.")
    else:
        text = (f"Δψ stays at least {threshold_mV:.0f} mV for <b>{fmt_time(lifetime_s)}</b> "
                "after the load is switched on.")
    st.markdown(f"<p class='result'>{text}</p>", unsafe_allow_html=True)

    st.markdown(
        f"<p class='result'>The vesicle contains <b>{fmt_count(sim['K_count_start'])}</b> "
        f"<span class='k'>K⁺ ions</span> in {sim['inner_diameter_nm']:.0f} nm of lumen. "
        f"{fmt_count(abs(sim['K_count_start'] - sim['K_count_equilibrium']))} of them must move "
        "until the inside matches the outside.</p>",
        unsafe_allow_html=True)

if sim["t_used_up"] is not None:
    st.warning(
        f"The K⁺ inside is used up after {fmt_time(sim['t_used_up'])}. From then on, valinomycin "
        "cannot compensate the load any more, so Δψ is no longer the K⁺ Nernst potential and is "
        "not shown. In a real vesicle, Δψ would then be set by the charge on the membrane "
        "capacitance, which this model does not include.")


# ------------------------------------------------------------
# Plots
# ------------------------------------------------------------
def new_axes(ylabel, figsize=(9, 3.6)):
    fig, ax = plt.subplots(figsize=figsize)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Time / s", color=INK, fontsize=11)
    ax.set_ylabel(ylabel, color=INK, fontsize=11)
    ax.set_xlim(0, time[-1])
    return fig, ax


def mark_load_start(ax):
    if load_start_s > 0:
        ax.axvline(load_start_s, color=CHARGE_COLOR, linestyle="--", linewidth=1)
        ax.text(load_start_s, 1.0, " load on", transform=ax.get_xaxis_transform(),
                color=CHARGE_COLOR, fontsize=9.5, va="top")


tab_psi, tab_k, tab_flux, tab_data, tab_model = st.tabs(
    ["Membrane potential", "K⁺ inside", "Ion flows", "Data export", "How the model works"])

with tab_psi:
    fig, ax = new_axes("Δψ / mV")
    ax.axhspan(-threshold_mV, threshold_mV, color=GRID, alpha=0.6, linewidth=0)
    ax.text(time[-1], threshold_mV, f"too weak (< {threshold_mV:.0f} mV)  ", ha="right", va="bottom",
            fontsize=9.5, color=MUTED)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.plot(time, psi_mV, color=INK, linewidth=2.2)
    if sim["t_used_up"] is not None:
        ax.axvline(sim["t_used_up"], color="#B3412E", linestyle=":", linewidth=1.2)
        ax.text(sim["t_used_up"], 0.5, " K⁺ used up", transform=ax.get_xaxis_transform(),
                color="#B3412E", fontsize=9.5, va="center")
    mark_load_start(ax)
    st.pyplot(fig)
    st.caption("Δψ = ψ(inside) − ψ(outside). Because valinomycin is fast, Δψ always equals the "
               "Nernst potential of the K⁺ that is left inside. In the grey band, Δψ is below "
               "the useful threshold set in the sidebar.")

with tab_k:
    fig, ax = new_axes("[K⁺] / mM")
    ax.plot(time, K_in_mM, color=K_COLOR, linewidth=2.2, label="inside")
    ax.axhline(K_out_mM, color=K_COLOR, linestyle="--", linewidth=1.2, label="outside (constant)")
    ax.set_ylim(bottom=0)
    mark_load_start(ax)
    ax.legend(frameon=False, fontsize=10)
    st.pyplot(fig)
    st.caption(f"The inside holds {fmt_count(sim['K_count_start'])} K⁺ ions at the start and "
               f"{fmt_count(K_ions[-1])} at the end. Δψ is zero when the inside "
               "concentration reaches the outside one.")

with tab_flux:
    fig, ax = new_axes("Flow into the vesicle / ions s⁻¹")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.plot(time, sim["charge_flux_in"], color=CHARGE_COLOR, linewidth=2.2,
            label="positive charge moved by the load")
    ax.plot(time, sim["K_flux_in"], color=K_COLOR, linewidth=2.2, label="K⁺ moved by valinomycin")
    mark_load_start(ax)
    ax.legend(frameon=False, fontsize=10)
    st.pyplot(fig)
    st.caption("Positive values mean ions move into the vesicle. The two curves mirror each "
               "other: every positive charge the load moves is compensated by one K⁺ moving "
               "the other way, so the net charge across the membrane stays constant.")

# ------------------------------------------------------------
# Data export
# ------------------------------------------------------------
with tab_data:
    df = pd.DataFrame({
        "time_s": time,
        "delta_psi_mV": psi_mV,
        "K_inside_mM": K_in_mM,
        "K_inside_ions": K_ions,
        "charge_flow_into_vesicle_per_s": sim["charge_flux_in"],
        "K_flow_into_vesicle_per_s": sim["K_flux_in"],
    })
    header = [
        "# Valinomycin K+ Nernst clamp simulation",
        "# Lines starting with # are parameters. Read in Python with pd.read_csv(file, comment='#')",
        f"# outer_diameter_nm = {outer_diameter_nm}",
        f"# bilayer_thickness_nm = {bilayer_thickness_nm}",
        f"# inner_diameter_nm = {sim['inner_diameter_nm']:.3f}",
        f"# inner_volume_L = {sim['volume_L']:.6e}",
        f"# temperature_C = {temperature_C}",
        f"# K_inside_start_mM = {K_in_initial_mM}",
        f"# K_outside_mM = {K_out_mM}",
        f"# load = {load_direction_label}; {load_model_label}",
        f"# load_at_start_charges_per_s = {load_rate}",
        f"# load_switched_on_s = {load_start_s}",
        f"# useful_threshold_mV = {threshold_mV}",
        "# delta_psi_mV is empty where the internal K+ is used up",
    ]
    csv = StringIO()
    csv.write("\n".join(header) + "\n")
    df.to_csv(csv, index=False)

    st.markdown(f"The table has **{len(df):,} time points** and contains every curve shown "
                "above. The file starts with the parameters of this run, so you always know "
                "how the data were made.")
    st.download_button(
        "Download data as CSV", data=csv.getvalue(),
        file_name=f"valinomycin_Kin{K_in_initial_mM:g}_Kout{K_out_mM:g}_d{outer_diameter_nm:g}nm.csv",
        mime="text/csv", type="primary")
    st.caption("Excel shows the parameter lines as the first rows, followed by the data. "
               "Each plot can also be saved as an image: right-click it and choose 'Save image as'.")
    st.dataframe(df.head(200), hide_index=True, use_container_width=True)

# ------------------------------------------------------------
# Explanation
# ------------------------------------------------------------
with tab_model:
    st.markdown("#### What is calculated")
    st.markdown("At every moment, Δψ is the Nernst potential of K⁺:")
    st.latex(r"\Delta\psi = \frac{RT}{F}\,\ln\frac{[\mathrm{K^+}]_{out}}{[\mathrm{K^+}]_{in}}")
    st.markdown(
        "The load moves positive charge across the membrane. Valinomycin compensates each "
        "charge immediately by moving one K⁺ in the opposite direction:")
    st.latex(r"J_{\mathrm{K^+}} = -\,J_{\mathrm{charge}}"
             r"\qquad \frac{dN_{\mathrm{K^+,in}}}{dt} = J_{\mathrm{K^+}}")
    st.markdown(
        "The internal K⁺ concentration changes, and with it Δψ. The two load types differ "
        "in how fast they run:\n"
        "- **Constant rate:** the same number of charges per second.\n"
        "- **Slows down as Δψ falls:** the rate is proportional to |Δψ|, so the discharge "
        "becomes slower and slower, like a capacitor discharging through a resistor.")
    st.markdown("#### Directions")
    st.markdown(
        "- Δψ = ψ(inside) − ψ(outside).\n"
        "- If [K⁺] is higher inside than outside, Δψ is negative (inside negative).\n"
        "- **Discharge** moves positive charge so that Δψ becomes weaker, and stops when "
        "[K⁺]inside = [K⁺]outside.\n"
        "- The two **push** options move charge in one fixed direction. Pushing K⁺ out "
        "can use up the internal pool completely.")
    st.markdown("#### Simplifications")
    st.markdown(
        "- Valinomycin kinetics are ignored: it is assumed to be much faster than the load.\n"
        "- The outside is a large reservoir with constant [K⁺].\n"
        "- Osmotic effects, counter-ions and volume changes are not included.\n"
        "- This is a teaching model for the finite energy content of a K⁺ gradient, not a "
        "quantitative prediction.")
