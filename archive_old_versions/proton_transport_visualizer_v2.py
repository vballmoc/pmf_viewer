import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ------------------------------------------------------------
# Educational model:
# Proton pumping into/out of a spherical vesicle.
#
# Version 2:
# - Pump slows down as the opposing membrane potential increases.
# - Pump has a maximum driving energy ΔG_pump.
# - Membrane leaks when Δψ becomes large.
# - Proton flux slider restricted to 1–5000 H+/s.
#
# This remains a didactic model, not a full electrochemical simulation.
# ------------------------------------------------------------

F = 96485.33212          # C/mol
R = 8.314462618         # J/(mol K)
NA = 6.02214076e23      # mol^-1
T = 298.15              # K

st.set_page_config(page_title="Ion transport: Δψ vs ΔpH", layout="wide")

st.title("Ion transport across a membrane: Δψ changes much faster than pH")

st.markdown("""
This model shows why proton pumping across a membrane rapidly generates a membrane
potential, while the bulk pH changes much less, especially in a buffered solution.

Version 2 includes two important effects:

1. the pump slows down as the opposing membrane potential increases;
2. the membrane develops a leak when the potential becomes large.
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

pump_rate_max = st.sidebar.slider("Maximum pump rate / H⁺ s⁻¹", 1, 5000, 1000, step=10)
deltaG_pump_kJ_mol = st.sidebar.slider("Pump driving energy / kJ mol⁻¹ H⁺", 2.0, 30.0, 15.0, step=0.5)
pump_steepness_mV = st.sidebar.slider("Pump slowdown width / mV", 1.0, 50.0, 10.0, step=1.0)

duration_s = st.sidebar.slider("Simulation time / s", 0.1, 120.0, 20.0, step=0.1)
direction = st.sidebar.radio(
    "Direction",
    ["pump H⁺ into vesicle", "pump H⁺ out of vesicle"],
    index=0,
)

st.sidebar.header("Membrane leak")

leak_threshold_mV = st.sidebar.slider("Leak onset |Δψ| / mV", 50, 300, 150, step=10)
leak_strength = st.sidebar.slider("Leak strength", 0.0, 2.0, 0.3, step=0.05)

st.sidebar.header("Display")
show_fluxes = st.sidebar.checkbox("Show pump and leak fluxes", value=True)

# -----------------------------
# Geometry and capacitance
# -----------------------------
radius_m = radius_nm * 1e-9
area_m2 = 4 * np.pi * radius_m**2
volume_L = (4/3) * np.pi * radius_m**3 * 1000  # m3 to L

# 1 µF/cm2 = 0.01 F/m2
capacitance_F_m2 = capacitance_uF_cm2 * 0.01
C_total = capacitance_F_m2 * area_m2

# Pump stopping potential from ΔG = F Δψ
# J/mol / C/mol = V
deltaG_pump_J_mol = deltaG_pump_kJ_mol * 1000
psi_stop_mV = deltaG_pump_J_mol / F * 1000

# -----------------------------
# Buffer capacity around starting pH
# -----------------------------
H0 = 10**(-pH_initial)
Ka = 10**(-buffer_pKa)
C_buffer_M = buffer_mM / 1000

beta_buffer = 2.303 * C_buffer_M * Ka * H0 / (Ka + H0)**2
beta_water = 2.303 * (H0 + 1e-14 / H0)
beta_total = beta_buffer + beta_water

# -----------------------------
# Time integration
# -----------------------------
n_points = max(500, min(3000, int(duration_s * 100)))
time = np.linspace(0, duration_s, n_points)
dt = time[1] - time[0]

sign = +1 if direction == "pump H⁺ into vesicle" else -1

delta_psi_mV = np.zeros_like(time)
pH_inside = np.zeros_like(time)
pH_inside[0] = pH_initial

net_pumped_protons = np.zeros_like(time)
pump_flux = np.zeros_like(time)
leak_flux = np.zeros_like(time)

# State variable: net charge-separated protons.
# Positive means positive charge accumulated inside.
q_protons = 0.0
mol_H_inside_change = 0.0

for i in range(1, n_points):
    psi = delta_psi_mV[i-1]

    # Opposing potential in the direction of pumping.
    # If pumping H+ inward, positive-inside Δψ opposes further pumping.
    # If pumping H+ outward, negative-inside Δψ opposes further pumping.
    opposing_psi_mV = sign * psi

    # Pump slows as opposing potential approaches the thermodynamic limit.
    # Logistic form: near full flux at low opposing Δψ, half at ψ_stop,
    # almost zero above ψ_stop.
    pump_fraction = 1.0 / (1.0 + np.exp((opposing_psi_mV - psi_stop_mV) / pump_steepness_mV))
    pump_flux_i = sign * pump_rate_max * pump_fraction

    # Leak is small below a threshold and rises above it.
    # It always dissipates the membrane potential.
    abs_psi = abs(psi)
    if abs_psi <= leak_threshold_mV or leak_strength == 0:
        leak_flux_i = 0.0
    else:
        excess = abs_psi - leak_threshold_mV

        # Flux in H+/s. This scaling is deliberately empirical for teaching.
        leak_flux_i = -np.sign(psi) * leak_strength * excess**2 / 10.0

    # Net proton movement as charge separation
    total_flux_i = pump_flux_i + leak_flux_i
    q_protons += total_flux_i * dt

    # Same proton movement changes the internal proton amount.
    # Positive flux adds H+ inside; negative removes H+ from inside.
    mol_H_inside_change += total_flux_i * dt / NA

    # Convert charge separation to voltage.
    charge_C = q_protons / NA * F
    delta_psi_mV[i] = (charge_C / C_total) * 1000

    # Convert transported protons to bulk pH using local buffer capacity.
    delta_pH = -mol_H_inside_change / (beta_total * volume_L)
    pH_inside[i] = pH_initial + delta_pH

    net_pumped_protons[i] = q_protons
    pump_flux[i] = pump_flux_i
    leak_flux[i] = leak_flux_i

# Approximate proton motive force contribution.
pmf_mV = delta_psi_mV - 59.16 * (pH_inside - pH_initial)

# -----------------------------
# Summary metrics
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Vesicle volume", f"{volume_L*1e18:.2f} aL")
with col2:
    st.metric("Total capacitance", f"{C_total:.2e} F")
with col3:
    st.metric("Pump stopping potential", f"{psi_stop_mV:.0f} mV")
with col4:
    st.metric("Final Δψ", f"{delta_psi_mV[-1]:.1f} mV")

col5, col6, col7, col8 = st.columns(4)

with col5:
    st.metric("Final internal pH", f"{pH_inside[-1]:.3f}")
with col6:
    st.metric("Final ΔpH", f"{pH_inside[-1] - pH_initial:.4f}")
with col7:
    st.metric("Final pump flux", f"{abs(pump_flux[-1]):.1f} H⁺ s⁻¹")
with col8:
    st.metric("Final leak flux", f"{abs(leak_flux[-1]):.1f} H⁺ s⁻¹")

# -----------------------------
# Plots
# -----------------------------
left, right = st.columns(2)

with left:
    fig, ax = plt.subplots()
    ax.plot(time, delta_psi_mV)
    ax.axhline(0, linewidth=0.8)
    ax.axhline(sign * psi_stop_mV, linestyle="--", linewidth=0.8)
    ax.axhline(sign * leak_threshold_mV, linestyle=":", linewidth=0.8)
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Membrane potential Δψ / mV")
    ax.set_title("Electrical effect: approach to a limiting potential")
    st.pyplot(fig)

with right:
    fig, ax = plt.subplots()
    ax.plot(time, pH_inside)
    ax.axhline(pH_initial, linewidth=0.8)
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Internal pH")
    ax.set_title("Bulk pH changes much less")
    st.pyplot(fig)

fig, ax = plt.subplots()
ax.plot(time, pmf_mV)
ax.axhline(0, linewidth=0.8)
ax.set_xlabel("Time / s")
ax.set_ylabel("Approx. proton motive force / mV")
ax.set_title("Approximate combined driving force")
st.pyplot(fig)

if show_fluxes:
    fig, ax = plt.subplots()
    ax.plot(time, pump_flux, label="pump flux")
    ax.plot(time, leak_flux, label="leak flux")
    ax.plot(time, pump_flux + leak_flux, label="net flux")
    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Flux / H⁺ s⁻¹")
    ax.set_title("Fluxes: pump slows, leak rises")
    ax.legend()
    st.pyplot(fig)

# -----------------------------
# Teaching notes
# -----------------------------
st.subheader("Teaching interpretation")

st.markdown(f"""
The pump has a driving energy of **{deltaG_pump_kJ_mol:.1f} kJ mol⁻¹ per H⁺**.
This corresponds to a thermodynamic stopping potential of approximately
**{psi_stop_mV:.0f} mV** for one transported proton.

In the beginning, the pump runs close to its maximum rate of
**{pump_rate_max:.0f} H⁺ s⁻¹**. As Δψ increases, the opposing electrochemical
work increases, so the pump slows down. Once the membrane potential becomes large,
a phenomenological leak dissipates the gradient.

The pH changes much less because the same transported protons are distributed
in the vesicle volume and buffered by **{buffer_mM:.1f} mM** buffer.
""")

st.subheader("Important simplifications")

st.markdown("""
- The pump is represented by a simple thermodynamic slowdown, not by a detailed kinetic model.
- The leak is phenomenological and only meant to prevent unlimited voltage buildup.
- Counter-ion movement is not explicitly modelled.
- Buffering is treated as a local linear approximation around the starting pH.
- The model is meant for teaching intuition, not quantitative prediction.
""")
