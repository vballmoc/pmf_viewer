import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ------------------------------------------------------------
# Educational model:
# Proton pumping into a spherical vesicle.
#
# Main idea:
#   A few transported charges already create a large membrane
#   potential because the membrane capacitance is tiny.
#   The same number of protons changes bulk pH only weakly,
#   especially when buffer is present.
#
# This is deliberately simplified for teaching.
# ------------------------------------------------------------

F = 96485.33212        # C/mol
R = 8.314462618       # J/(mol K)

st.set_page_config(page_title="Ion transport: Δψ vs ΔpH", layout="wide")

st.title("Ion transport across a membrane: Δψ changes much faster than pH")

st.markdown("""
This simple model shows why proton pumping across a membrane can rapidly generate
a membrane potential, while the bulk pH changes only little, especially in a buffered solution.
""")

# -----------------------------
# Sidebar parameters
# -----------------------------
st.sidebar.header("Vesicle and solution")

radius_nm = st.sidebar.slider("Vesicle radius / nm", 25, 1000, 100, step=25)
capacitance_uF_cm2 = st.sidebar.slider("Membrane capacitance / µF cm⁻²", 0.2, 2.0, 1.0, step=0.1)
pH_initial = st.sidebar.slider("Initial pH inside", 6.0, 8.5, 7.0, step=0.1)
buffer_mM = st.sidebar.slider("Buffer concentration / mM", 0.0, 200.0, 50.0, step=5.0)
buffer_pKa = st.sidebar.slider("Buffer pKa", 5.5, 8.5, 7.5, step=0.1)

st.sidebar.header("Proton pump")

pump_rate = st.sidebar.slider("Pumped protons per second", 1, 100000, 1000, step=100)
duration_s = st.sidebar.slider("Simulation time / s", 0.1, 60.0, 10.0, step=0.1)
direction = st.sidebar.radio(
    "Direction",
    ["pump H⁺ into vesicle", "pump H⁺ out of vesicle"],
    index=0,
)

st.sidebar.header("Display")

max_points = 1000
n_points = min(max_points, max(100, int(duration_s * 100)))
time = np.linspace(0, duration_s, n_points)

# -----------------------------
# Geometry and capacitance
# -----------------------------
radius_m = radius_nm * 1e-9
area_m2 = 4 * np.pi * radius_m**2
volume_L = (4/3) * np.pi * radius_m**3 * 1000  # m3 to L

# 1 µF/cm2 = 0.01 F/m2
capacitance_F_m2 = capacitance_uF_cm2 * 0.01
C_total = capacitance_F_m2 * area_m2

# -----------------------------
# Proton transport
# -----------------------------
sign = +1 if direction == "pump H⁺ into vesicle" else -1

n_protons = pump_rate * time
mol_H_pumped = sign * n_protons / 6.02214076e23

# Charge separation creates voltage:
# Q = z F n; V = Q / C
charge_C = sign * (n_protons / 6.02214076e23) * F
delta_psi_V = charge_C / C_total
delta_psi_mV = delta_psi_V * 1000

# -----------------------------
# pH model with buffer
# -----------------------------
# Buffer capacity beta near pH:
# beta = 2.303 * C_buffer * Ka*[H] / (Ka + [H])^2
# plus water contribution, which is tiny around neutral pH.
#
# dpH = -dn_H / (beta * volume)
# This is a local approximation around the initial pH.
H0 = 10**(-pH_initial)
Ka = 10**(-buffer_pKa)
C_buffer_M = buffer_mM / 1000

beta_buffer = 2.303 * C_buffer_M * Ka * H0 / (Ka + H0)**2
beta_water = 2.303 * (H0 + 1e-14 / H0)
beta_total = beta_buffer + beta_water

delta_pH = -mol_H_pumped / (beta_total * volume_L)
pH_inside = pH_initial + delta_pH

# Proton motive force contribution:
# Δp = Δψ - 59 mV * ΔpH at 25 °C, sign convention simplified
pmf_mV = delta_psi_mV - 59.16 * (pH_inside - pH_initial)

# -----------------------------
# Summary metrics
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Vesicle volume", f"{volume_L*1e18:.2f} aL")
with col2:
    st.metric("Membrane capacitance", f"{C_total:.2e} F")
with col3:
    st.metric("Final Δψ", f"{delta_psi_mV[-1]:.1f} mV")
with col4:
    st.metric("Final ΔpH", f"{pH_inside[-1] - pH_initial:.4f}")

# -----------------------------
# Warning when voltage becomes unrealistic
# -----------------------------
if np.max(np.abs(delta_psi_mV)) > 250:
    st.warning(
        "The calculated membrane potential exceeds ±250 mV. "
        "Real membranes would usually respond by counter-ion movement, leakage, "
        "pump slowdown, or membrane failure. This is useful here as a teaching point."
    )

# -----------------------------
# Plots
# -----------------------------
left, right = st.columns(2)

with left:
    fig, ax = plt.subplots()
    ax.plot(time, delta_psi_mV)
    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Membrane potential Δψ / mV")
    ax.set_title("Electrical effect of proton pumping")
    st.pyplot(fig)

with right:
    fig, ax = plt.subplots()
    ax.plot(time, pH_inside)
    ax.axhline(pH_initial, linewidth=0.8)
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Internal pH")
    ax.set_title("Bulk pH change inside the vesicle")
    st.pyplot(fig)

fig, ax = plt.subplots()
ax.plot(time, pmf_mV)
ax.axhline(0, linewidth=0.8)
ax.set_xlabel("Time / s")
ax.set_ylabel("Approx. proton motive force / mV")
ax.set_title("Approximate combined driving force")
st.pyplot(fig)

# -----------------------------
# Teaching notes
# -----------------------------
st.subheader("Teaching interpretation")

st.markdown(f"""
At the selected settings, the model pumps **{n_protons[-1]:.0f} protons** over
**{duration_s:.1f} s**.

Because the total membrane capacitance of a {radius_nm} nm vesicle is only
**{C_total:.2e} F**, this small amount of charge already produces a sizeable
voltage.

The pH change is much smaller because the transported protons are diluted into
the vesicle volume and buffered by **{buffer_mM:.1f} mM** buffer.
""")

st.subheader("Important simplifications")

st.markdown("""
- No counter-ion movement is included.
- No proton leakage is included.
- The pump rate is kept constant, even at high Δψ.
- Buffering is treated as a local linear approximation around the starting pH.
- The model is meant for intuition, not quantitative prediction of a real experiment.
""")
