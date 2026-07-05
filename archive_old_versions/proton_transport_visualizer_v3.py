import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ------------------------------------------------------------
# Educational model:
# Proton pumping into/out of a spherical vesicle.
#
# Version 3:
# - ΔpH is plotted instead of absolute pH.
# - pH range is 5–9.
# - Pump slows down as Δψ approaches the pump stopping potential.
# - Leak is linear with Δψ, producing a natural steady state.
# - Optional membrane/headgroup buffering can be included.
#
# This remains a didactic model, not a full electrochemical simulation.
# ------------------------------------------------------------

F = 96485.33212          # C/mol
NA = 6.02214076e23      # mol^-1

st.set_page_config(page_title="Ion transport: Δψ vs ΔpH", layout="wide")

st.title("Ion transport across a membrane: Δψ changes much faster than ΔpH")

st.markdown("""
This model shows why proton pumping across a membrane rapidly generates a membrane
potential, while the bulk pH changes much less, especially in a buffered solution.

Version 3 includes:
- pump slowdown against the opposing membrane potential;
- a linear leak that dissipates Δψ;
- optional buffering by membrane headgroups;
- ΔpH plotted directly instead of absolute pH.
""")

# -----------------------------
# Sidebar parameters
# -----------------------------
st.sidebar.header("Vesicle and solution")

radius_nm = st.sidebar.slider("Vesicle radius / nm", 25, 1000, 100, step=25)
capacitance_uF_cm2 = st.sidebar.slider("Membrane capacitance / µF cm⁻²", 0.2, 2.0, 1.0, step=0.1)

pH_initial = st.sidebar.slider("Initial pH inside", 5.0, 9.0, 7.0, step=0.1)
buffer_mM = st.sidebar.slider("Soluble buffer concentration / mM", 0.0, 200.0, 50.0, step=5.0)
buffer_pKa = st.sidebar.slider("Soluble buffer pKa", 5.0, 9.0, 7.5, step=0.1)

st.sidebar.header("Membrane buffering")

include_membrane_buffer = st.sidebar.checkbox("Include membrane/headgroup buffering", value=True)

lipid_area_nm2 = st.sidebar.slider("Area per lipid / nm²", 0.4, 1.0, 0.7, step=0.05)
buffering_lipid_fraction = st.sidebar.slider("Fraction of titratable inner leaflet lipids", 0.0, 1.0, 0.25, step=0.05)
headgroup_pKa = st.sidebar.slider("Headgroup apparent pKa", 3.0, 9.0, 6.5, step=0.1)

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

leak_conductance = st.sidebar.slider(
    "Linear leak conductance / H⁺ s⁻¹ mV⁻¹",
    0.0, 50.0, 5.0, step=0.5
)

st.sidebar.header("Display")
show_fluxes = st.sidebar.checkbox("Show pump and leak fluxes", value=True)
show_absolute_pH = st.sidebar.checkbox("Also show absolute internal pH", value=False)

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
deltaG_pump_J_mol = deltaG_pump_kJ_mol * 1000
psi_stop_mV = deltaG_pump_J_mol / F * 1000

# -----------------------------
# Buffer capacity around starting pH
# -----------------------------
H0 = 10**(-pH_initial)
Kw = 1e-14

# Soluble buffer capacity beta in mol/L/pH
Ka_buffer = 10**(-buffer_pKa)
C_buffer_M = buffer_mM / 1000
beta_soluble = 2.303 * C_buffer_M * Ka_buffer * H0 / (Ka_buffer + H0)**2

# Water contribution, usually tiny around neutral pH
beta_water = 2.303 * (H0 + Kw / H0)

# Membrane/headgroup buffering:
# We approximate titratable inner-leaflet lipid headgroups as an additional
# finite buffer pool in the inner leaflet. Its buffering capacity is converted
# to an effective mol/L/pH by dividing by the vesicle volume.
area_nm2 = area_m2 * 1e18
n_lipids_inner = area_nm2 / lipid_area_nm2

Ka_headgroup = 10**(-headgroup_pKa)
n_titratable = n_lipids_inner * buffering_lipid_fraction

# Buffer capacity of N sites:
# d(bound protons)/dpH = ln(10) * N * Ka*H / (Ka+H)^2
# Convert molecules/pH to mol/L/pH.
buffer_sites_per_pH = 2.303 * n_titratable * Ka_headgroup * H0 / (Ka_headgroup + H0)**2
beta_membrane = buffer_sites_per_pH / NA / volume_L

if not include_membrane_buffer:
    beta_membrane = 0.0

beta_total = beta_soluble + beta_water + beta_membrane

# -----------------------------
# Time integration
# -----------------------------
n_points = max(500, min(3000, int(duration_s * 100)))
time = np.linspace(0, duration_s, n_points)
dt = time[1] - time[0]

sign = +1 if direction == "pump H⁺ into vesicle" else -1

delta_psi_mV = np.zeros_like(time)
delta_pH = np.zeros_like(time)
pH_inside = np.zeros_like(time)
pH_inside[0] = pH_initial

net_charge_protons = np.zeros_like(time)
pump_flux = np.zeros_like(time)
leak_flux = np.zeros_like(time)

# State variables
q_protons = 0.0               # charge-separated protons; positive means inside positive
mol_H_inside_change = 0.0     # mol H+ added to inside

for i in range(1, n_points):
    psi = delta_psi_mV[i-1]

    # Opposing potential in the direction of pumping.
    opposing_psi_mV = sign * psi

    # Pump slows as opposing potential approaches the thermodynamic limit.
    pump_fraction = 1.0 / (1.0 + np.exp((opposing_psi_mV - psi_stop_mV) / pump_steepness_mV))
    pump_flux_i = sign * pump_rate_max * pump_fraction

    # Linear leak: always dissipates Δψ.
    # If inside is positive, leak flux is negative.
    # If inside is negative, leak flux is positive.
    leak_flux_i = -leak_conductance * psi

    # Net proton movement as charge separation
    total_flux_i = pump_flux_i + leak_flux_i
    q_protons += total_flux_i * dt

    # Same proton movement changes the internal proton amount.
    mol_H_inside_change += total_flux_i * dt / NA

    # Convert charge separation to voltage.
    charge_C = q_protons / NA * F
    delta_psi_mV[i] = (charge_C / C_total) * 1000

    # Convert transported protons to bulk ΔpH using buffer capacity.
    delta_pH[i] = -mol_H_inside_change / (beta_total * volume_L)
    pH_inside[i] = pH_initial + delta_pH[i]

    net_charge_protons[i] = q_protons
    pump_flux[i] = pump_flux_i
    leak_flux[i] = leak_flux_i

# Approximate proton motive force contribution.
pmf_mV = delta_psi_mV - 59.16 * delta_pH

# -----------------------------
# Summary metrics
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Start pH inside", f"{pH_initial:.1f}")
with col2:
    st.metric("Final internal pH", f"{pH_inside[-1]:.3f}")
with col3:
    st.metric("Final ΔpH", f"{delta_pH[-1]:+.4f}")
with col4:
    st.metric("Final Δψ", f"{delta_psi_mV[-1]:+.1f} mV")

col5, col6, col7, col8 = st.columns(4)

with col5:
    st.metric("Vesicle volume", f"{volume_L*1e18:.2f} aL")
with col6:
    st.metric("Total capacitance", f"{C_total:.2e} F")
with col7:
    st.metric("Pump stopping potential", f"{psi_stop_mV:.0f} mV")
with col8:
    st.metric("Final net flux", f"{pump_flux[-1] + leak_flux[-1]:+.1f} H⁺ s⁻¹")

# -----------------------------
# Buffer capacity display
# -----------------------------
st.subheader("Buffer capacity used in the model")

b1, b2, b3, b4 = st.columns(4)

with b1:
    st.metric("Soluble buffer β", f"{beta_soluble:.2e} mol L⁻¹ pH⁻¹")
with b2:
    st.metric("Water β", f"{beta_water:.2e} mol L⁻¹ pH⁻¹")
with b3:
    st.metric("Membrane β", f"{beta_membrane:.2e} mol L⁻¹ pH⁻¹")
with b4:
    st.metric("Total β", f"{beta_total:.2e} mol L⁻¹ pH⁻¹")

st.caption(
    "Membrane buffering is treated as an effective inner-leaflet buffer pool. "
    "This is a simplification, but it shows why lipid headgroups can matter strongly in small vesicles."
)

# -----------------------------
# Plots
# -----------------------------
left, right = st.columns(2)

with left:
    fig, ax = plt.subplots()
    ax.plot(time, delta_psi_mV)
    ax.axhline(0, linewidth=0.8)
    ax.axhline(sign * psi_stop_mV, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Membrane potential Δψ / mV")
    ax.set_title("Electrical effect: fast approach to steady state")
    st.pyplot(fig)

with right:
    fig, ax = plt.subplots()
    ax.plot(time, delta_pH)
    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("Time / s")
    ax.set_ylabel("ΔpH = pH(in) − pH(start)")
    ax.set_title("Bulk pH changes much less")
    st.pyplot(fig)

fig, ax = plt.subplots()
ax.plot(time, pmf_mV)
ax.axhline(0, linewidth=0.8)
ax.set_xlabel("Time / s")
ax.set_ylabel("Approx. proton motive force / mV")
ax.set_title("Approximate combined driving force")
st.pyplot(fig)

if show_absolute_pH:
    fig, ax = plt.subplots()
    ax.plot(time, pH_inside)
    ax.axhline(pH_initial, linewidth=0.8)
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Internal pH")
    ax.set_title("Absolute internal pH")
    st.pyplot(fig)

if show_fluxes:
    fig, ax = plt.subplots()
    ax.plot(time, pump_flux, label="pump flux")
    ax.plot(time, leak_flux, label="leak flux")
    ax.plot(time, pump_flux + leak_flux, label="net flux")
    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Flux / H⁺ s⁻¹")
    ax.set_title("Fluxes: pump slows, linear leak rises")
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

The linear leak has the form:

`J_leak = −g_leak · Δψ`

Therefore, the leak always runs against the charge separation. At steady state,
the remaining pump flux and the leak flux balance each other.

For the pH calculation, the model uses a total buffer capacity:

`β_total = β_soluble buffer + β_water + β_membrane`

The membrane term is calculated from the number of titratable inner-leaflet
headgroups and then converted into an effective molar buffer capacity within
the vesicle volume.
""")

st.subheader("Important simplifications")

st.markdown("""
- The pump is represented by a simple thermodynamic slowdown, not by a detailed kinetic model.
- The leak is linear in Δψ and phenomenological.
- Counter-ion movement is not explicitly modelled.
- Buffering is treated as a local linear approximation around the starting pH.
- Membrane/headgroup buffering is approximated as an effective inner-volume buffer.
- The model is meant for teaching intuition, not quantitative prediction.
""")
