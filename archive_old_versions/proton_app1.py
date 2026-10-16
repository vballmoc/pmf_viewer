import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

# At the TOP of your script (after imports):
if 'reset_counter' not in st.session_state:
    st.session_state.reset_counter = 0

# Replace your reset button:
if st.sidebar.button("🔄 Reset to Defaults"):
    st.session_state.reset_counter += 1  # Force widget re-creation
    st.rerun()
    
# Custom styling
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .param-box {
        background-color: #e6f3ff;
        padding: 8px;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
    }
    .group-header {
        color: #1f77b4;
        font-weight: bold;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Constants
F = 96485.33212  # C/mol
NA = 6.02214076e23  # mol^-1

st.set_page_config(page_title="Ion transport: Δψ vs ΔpH", layout="wide")

st.title("Ion transport across a membrane: Δψ changes much faster than ΔpH")

# Reset button
if st.sidebar.button("🔄 Reset to Defaults"):
    st.rerun()

# Key Concepts expander
with st.expander("🔑 **Key Concepts**"):
    st.markdown("""
    - **Δψ (Membrane Potential)**: Electrical gradient across the membrane (mV).
    - **ΔpH**: pH difference between inside and outside.
    - **Buffer Capacity (β)**: Resistance to pH changes (higher β = more stable pH).
    - **Pump Stopping Potential**: Maximum Δψ the pump can generate (thermodynamic limit).
    - **Soft-Threshold Leak**: Leakiness increases smoothly above a membrane potential threshold.
    """)

st.markdown("""
This model shows why proton pumping across a membrane rapidly generates a membrane
potential, while the bulk pH changes much less, especially in a buffered solution.

Version 4 includes:
- Pump slowdown against the opposing membrane potential
- A soft-threshold leak that dissipates Δψ
- Optional buffering by membrane headgroups
- ΔpH plotted directly instead of absolute pH.
""")

# Sidebar parameters with tooltips
st.sidebar.header("Vesicle and solution")
radius_nm = st.sidebar.slider(
    "Vesicle radius / nm",
    25, 1000, 100, step=25,
    help="Typical liposomes: 50–200 nm. Smaller vesicles reach steady-state faster."
)
capacitance_uF_cm2 = st.sidebar.slider(
    "Membrane capacitance / µF cm⁻²",
    0.2, 2.0, 1.0, step=0.1,
    help="Higher capacitance = more charge storage per voltage."
)
pH_initial = st.sidebar.slider(
    "Initial pH inside",
    5.0, 9.0, 7.0, step=0.1,
    help="Starting pH inside the vesicle."
)
buffer_mM = st.sidebar.slider(
    "Soluble buffer concentration / mM",
    0.0, 200.0, 50.0, step=5.0,
    help="Higher buffer = more resistance to pH changes."
)
buffer_pKa = st.sidebar.slider(
    "Soluble buffer pKa",
    5.0, 9.0, 7.5, step=0.1,
    help="pKa of the soluble buffer. Buffer works best near its pKa."
)

st.sidebar.header("Membrane buffering")
include_membrane_buffer = st.sidebar.checkbox(
    "Include membrane/headgroup buffering",
    value=True,
    help="Account for proton binding to lipid headgroups."
)
lipid_area_nm2 = st.sidebar.slider(
    "Area per lipid / nm²",
    0.4, 1.0, 0.7, step=0.05,
    help="Smaller area = more lipids per vesicle."
)
buffering_lipid_fraction = st.sidebar.slider(
    "Fraction of titratable inner leaflet lipids",
    0.0, 1.0, 0.25, step=0.05,
    help="Fraction of lipids that can bind/release protons."
)
headgroup_pKa = st.sidebar.slider(
    "Headgroup apparent pKa",
    3.0, 9.0, 6.5, step=0.1,
    help="pKa of lipid headgroups. Typically 3–7 for phospholipids."
)

st.sidebar.header("Proton pump")
pump_rate_max = st.sidebar.slider(
    "Maximum pump rate / H⁺ s⁻¹",
    1, 5000, 1000, step=10,
    help="Maximum protons pumped per second."
)
deltaG_pump_kJ_mol = st.sidebar.slider(
    "Pump driving energy / kJ mol⁻¹ H⁺",
    2.0, 30.0, 15.0, step=0.5,
    help="Energy driving the pump. Higher = stronger pumping."
)
pump_steepness_mV = st.sidebar.slider(
    "Pump slowdown width / mV",
    1.0, 50.0, 10.0, step=1.0,
    help="How sharply the pump slows near its stopping potential."
)
duration_s = st.sidebar.slider(
    "Simulation time / s",
    0.1, 120.0, 20.0, step=0.1,
    help="Total duration of the simulation."
)
direction = st.sidebar.radio(
    "Direction",
    ["pump H⁺ into vesicle", "pump H⁺ out of vesicle"],
    index=0,
    help="Direction of proton pumping."
)

st.sidebar.header("Membrane leak")
leak_conductance = st.sidebar.slider(
    "Leak conductance above threshold / H⁺ s⁻¹ mV⁻¹",
    0.0, 50.0, 5.0, step=0.5,
    help="How strongly the leak dissipates Δψ above the threshold."
)
leak_threshold_mV = st.sidebar.slider(
    "Soft leak threshold / mV",
    0.0, 200.0, 40.0, step=5.0,
    help="Δψ at which leakiness starts to increase."
)
leak_softness_mV = st.sidebar.slider(
    "Softness of leak onset / mV",
    1.0, 50.0, 10.0, step=1.0,
    help="Width of the transition from no leak to full leak."
)

st.sidebar.header("Display")
show_fluxes = st.sidebar.checkbox("Show pump and leak fluxes", value=True)
show_absolute_pH = st.sidebar.checkbox("Also show absolute internal pH", value=False)
show_leak_curve = st.sidebar.checkbox("Show leak curve", value=True)

# Helper function: soft-threshold leak
def soft_threshold_leak(psi_mV, conductance, threshold_mV, softness_mV):
    abs_psi = np.abs(psi_mV)
    x = (abs_psi - threshold_mV) / softness_mV
    excess = softness_mV * np.logaddexp(0, x)
    return -np.sign(psi_mV) * conductance * excess

# Geometry and capacitance
radius_m = radius_nm * 1e-9
area_m2 = 4 * np.pi * radius_m**2
volume_L = (4/3) * np.pi * radius_m**3 * 1000  # m3 to L
capacitance_F_m2 = capacitance_uF_cm2 * 0.01
C_total = capacitance_F_m2 * area_m2
deltaG_pump_J_mol = deltaG_pump_kJ_mol * 1000
psi_stop_mV = deltaG_pump_J_mol / F * 1000

# Buffer capacity
H0 = 10**(-pH_initial)
Kw = 1e-14
Ka_buffer = 10**(-buffer_pKa)
C_buffer_M = buffer_mM / 1000
beta_soluble = 2.303 * C_buffer_M * Ka_buffer * H0 / (Ka_buffer + H0)**2
beta_water = 2.303 * (H0 + Kw / H0)
area_nm2 = area_m2 * 1e18
n_lipids_inner = area_nm2 / lipid_area_nm2
Ka_headgroup = 10**(-headgroup_pKa)
n_titratable = n_lipids_inner * buffering_lipid_fraction
buffer_sites_per_pH = 2.303 * n_titratable * Ka_headgroup * H0 / (Ka_headgroup + H0)**2
beta_membrane = buffer_sites_per_pH / NA / volume_L
if not include_membrane_buffer:
    beta_membrane = 0.0
beta_total = beta_soluble + beta_water + beta_membrane
if beta_total <= 0:
    beta_total = 1e-30

# Time integration (vectorized)
n_points = max(500, min(3000, int(duration_s * 100)))
time = np.linspace(0, duration_s, n_points)
dt = time[1] - time[0]
sign = +1 if direction == "pump H⁺ into vesicle" else -1

# Initialize arrays
delta_psi_mV = np.zeros_like(time)
delta_pH = np.zeros_like(time)
pH_inside = np.zeros_like(time)
pH_inside[0] = pH_initial
net_charge_protons = np.zeros_like(time)
pump_flux = np.zeros_like(time)
leak_flux = np.zeros_like(time)

# Vectorized simulation
for i in range(1, n_points):
    psi = delta_psi_mV[i-1]
    opposing_psi_mV = sign * psi
    pump_fraction = 1.0 / (1.0 + np.exp((opposing_psi_mV - psi_stop_mV) / pump_steepness_mV))
    pump_flux_i = sign * pump_rate_max * pump_fraction
    leak_flux_i = soft_threshold_leak(psi, leak_conductance, leak_threshold_mV, leak_softness_mV)
    total_flux_i = pump_flux_i + leak_flux_i
    q_protons = net_charge_protons[i-1] + total_flux_i * dt
    mol_H_inside_change = q_protons / NA
    charge_C = q_protons / NA * F
    delta_psi_mV[i] = (charge_C / C_total) * 1000
    delta_pH[i] = -mol_H_inside_change / (beta_total * volume_L)
    pH_inside[i] = pH_initial + delta_pH[i]
    net_charge_protons[i] = q_protons
    pump_flux[i] = pump_flux_i
    leak_flux[i] = leak_flux_i

pmf_mV = delta_psi_mV - 59.16 * delta_pH

# Summary metrics
col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("Start pH inside", f"{pH_initial:.1f}")
with col2: st.metric("Final internal pH", f"{pH_inside[-1]:.3f}")
with col3: st.metric("Final ΔpH", f"{delta_pH[-1]:+.4f}")
with col4: st.metric("Final Δψ", f"{delta_psi_mV[-1]:+.1f} mV")

col5, col6, col7, col8 = st.columns(4)
with col5: st.metric("Vesicle volume", f"{volume_L*1e18:.2f} aL")
with col6: st.metric("Total capacitance", f"{C_total:.2e} F")
with col7: st.metric("Pump stopping potential", f"{psi_stop_mV:.0f} mV")
with col8: st.metric("Final net flux", f"{pump_flux[-1] + leak_flux[-1]:+.1f} H⁺ s⁻¹")

# Buffer capacity display
st.subheader("Buffer capacity used in the model")
b1, b2, b3, b4 = st.columns(4)
with b1: st.metric("Soluble buffer β", f"{beta_soluble:.2e} mol L⁻¹ pH⁻¹")
with b2: st.metric("Water β", f"{beta_water:.2e} mol L⁻¹ pH⁻¹")
with b3: st.metric("Membrane β", f"{beta_membrane:.2e} mol L⁻¹ pH⁻¹")
with b4: st.metric("Total β", f"{beta_total:.2e} mol L⁻¹ pH⁻¹")

# -----------------------------
# PARAMETER SUMMARY (Boxed)
# -----------------------------
st.subheader("📋 Simulation Parameters")

params = {
    "🔵 **Vesicle**": [
        ("Radius", f"{radius_nm} nm"),
        ("Membrane capacitance", f"{capacitance_uF_cm2} µF/cm²"),
    ],
    "🟢 **Solution**": [
        ("Initial pH", f"{pH_initial}"),
        ("Buffer concentration", f"{buffer_mM} mM"),
        ("Buffer pKa", f"{buffer_pKa}"),
    ],
    "🟣 **Membrane Buffering**": [
        ("Membrane buffering", "✅ Yes" if include_membrane_buffer else "❌ No"),
        ("Lipid area", f"{lipid_area_nm2} nm²"),
        ("Titratable lipids", f"{buffering_lipid_fraction*100:.0f}%"),
        ("Headgroup pKa", f"{headgroup_pKa}"),
    ],
    "🔴 **Pump**": [
        ("Max pump rate", f"{pump_rate_max} H⁺/s"),
        ("Pump energy", f"{deltaG_pump_kJ_mol} kJ/mol"),
        ("Pump steepness", f"{pump_steepness_mV} mV"),
        ("Direction", direction),
    ],
    "🟠 **Leak**": [
        ("Leak conductance", f"{leak_conductance} H⁺/s/mV"),
        ("Leak threshold", f"{leak_threshold_mV} mV"),
        ("Leak softness", f"{leak_softness_mV} mV"),
    ],
    "⏱️ **Simulation**": [
        ("Duration", f"{duration_s} s"),
    ],
}

col1, col2 = st.columns(2)
for i, (group, items) in enumerate(params.items()):
    with col1 if i % 2 == 0 else col2:
        st.markdown(f"<div class='group-header'>{group}</div>", unsafe_allow_html=True)
        for label, value in items:
            st.markdown(f"<div class='param-box'>{label}: {value}</div>", unsafe_allow_html=True)

# -----------------------------
# Plots (improved styling)
# -----------------------------
left, right = st.columns(2)

with left:
    fig, ax = plt.subplots()
    ax.plot(time, delta_psi_mV)
    ax.axhline(0, linewidth=0.8)
    ax.axhline(sign * psi_stop_mV, linestyle="--", linewidth=0.8, label="pump stopping potential")
    ax.axhline(sign * leak_threshold_mV, linestyle=":", linewidth=0.8, label="soft leak threshold")
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Membrane potential Δψ / mV")
    ax.set_title("Electrical effect: fast approach to steady state")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f0f2f6')
    st.pyplot(fig)

with right:
    fig, ax = plt.subplots()
    ax.plot(time, delta_pH)
    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("Time / s")
    ax.set_ylabel("ΔpH = pH(in) − pH(start)")
    ax.set_title("Bulk pH changes much less")
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f0f2f6')
    st.pyplot(fig)

fig, ax = plt.subplots()
ax.plot(time, pmf_mV)
ax.axhline(0, linewidth=0.8)
ax.set_xlabel("Time / s")
ax.set_ylabel("Approx. proton motive force / mV")
ax.set_title("Approximate combined driving force")
ax.grid(True, alpha=0.3)
ax.set_facecolor('#f0f2f6')
st.pyplot(fig)

if show_absolute_pH:
    fig, ax = plt.subplots()
    ax.plot(time, pH_inside)
    ax.axhline(pH_initial, linewidth=0.8)
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Internal pH")
    ax.set_title("Absolute internal pH")
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f0f2f6')
    st.pyplot(fig)

if show_fluxes:
    fig, ax = plt.subplots()
    ax.plot(time, pump_flux, label="pump flux")
    ax.plot(time, leak_flux, label="leak flux")
    ax.plot(time, pump_flux + leak_flux, label="net flux")
    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Flux / H⁺ s⁻¹")
    ax.set_title("Fluxes: pump slows, soft-threshold leak rises")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f0f2f6')
    st.pyplot(fig)

if show_leak_curve:
    psi_range = np.linspace(-250, 250, 1000)
    leak_range = soft_threshold_leak(
        psi_range,
        leak_conductance,
        leak_threshold_mV,
        leak_softness_mV
    )
    fig, ax = plt.subplots()
    ax.plot(psi_range, leak_range)
    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)
    ax.axvline(leak_threshold_mV, linestyle=":", linewidth=0.8)
    ax.axvline(-leak_threshold_mV, linestyle=":", linewidth=0.8)
    ax.set_xlabel("Membrane potential Δψ / mV")
    ax.set_ylabel("Leak flux / H⁺ s⁻¹")
    ax.set_title("Soft-threshold leak function")
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f0f2f6')
    st.pyplot(fig)

# Teaching notes
st.subheader("Teaching interpretation")
st.markdown(f"""
The pump has a driving energy of **{deltaG_pump_kJ_mol:.1f} kJ mol⁻¹ per H⁺**.
This corresponds to a thermodynamic stopping potential of approximately
**{psi_stop_mV:.0f} mV** for one transported proton.

The leak is now described by a soft threshold. Below about
**{leak_threshold_mV:.0f} mV**, the membrane is almost tight. Around the threshold,
leakiness starts to increase smoothly. The transition width is controlled by the
softness parameter, currently **{leak_softness_mV:.0f} mV**.

For the pH calculation, the model uses a total buffer capacity:

`β_total = β_soluble buffer + β_water + β_membrane`

The membrane term is calculated from the number of titratable inner-leaflet
headgroups and then converted into an effective molar buffer capacity within
the vesicle volume.
""")

st.subheader("Important simplifications")
st.markdown("""
- The pump is represented by a simple thermodynamic slowdown, not by a detailed kinetic model.
- The leak is phenomenological and represented by a smooth threshold function.
- Counter-ion movement is not explicitly modelled.
- Buffering is treated as a local linear approximation around the starting pH.
- Membrane/headgroup buffering is approximated as an effective inner-volume buffer.
- The model is meant for teaching intuition, not quantitative prediction.
""")