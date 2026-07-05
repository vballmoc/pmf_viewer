import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from io import StringIO

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
        margin-bottom: 5px;
    }
    .result-box {
        background-color: #e6ffe6;
        padding: 8px;
        border-radius: 5px;
        border-left: 4px solid #2e7d32;
        margin-bottom: 5px;
    }
    .group-header {
        color: #1f77b4;
        font-weight: bold;
        margin-top: 15px;
    }
    .result-header {
        color: #2e7d32;
        font-weight: bold;
        margin-top: 15px;
    }
    .streamlit-expanderHeader {
        font-size: 1.1em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Constants
F = 96485.33212  # C/mol
NA = 6.02214076e23  # mol^-1

st.set_page_config(page_title="Ion transport: Δψ vs ΔpH", layout="wide")

st.title("Ion transport across a membrane: Δψ changes faster than ΔpH")

# Reset button
if st.sidebar.button("🔄 Reset to Defaults"):
    st.rerun()

# Key Concepts expander
with st.expander("🔑 **Key Concepts**", expanded=False):
    st.markdown("""
    - **Δψ (Membrane Potential)**: Electrical gradient across the membrane (mV).
    - **ΔpH**: pH difference between inside and outside.
    - **Buffer Capacity (β)**: Resistance to pH changes (higher β = more stable pH).
    - **Pump Stopping Potential**: Maximum Δψ the pump can generate (thermodynamic limit).
    - **Soft-Threshold Leak**: Leakiness increases smoothly above a membrane potential threshold.
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

# Time integration
# Add to SIDEBAR (under "Simulation" section):
n_points = st.sidebar.slider(
    "Number of time points",
    100, 3000, 500, step=100,
    help="Higher = smoother curves but slower. Default: 1000."
)

# Then replace the old line with:
# Time integration
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
mol_H_inside_change = 0.0  # Track cumulative moles of H+ added

# Simulation loop with pH-dependent β
for i in range(1, n_points):
    psi = delta_psi_mV[i-1]
    opposing_psi_mV = sign * psi

    # Pump and leak fluxes
    pump_fraction = 1.0 / (1.0 + np.exp((opposing_psi_mV - psi_stop_mV) / pump_steepness_mV))
    pump_flux_i = sign * pump_rate_max * pump_fraction
    leak_flux_i = soft_threshold_leak(psi, leak_conductance, leak_threshold_mV, leak_softness_mV)
    total_flux_i = pump_flux_i + leak_flux_i

    # Incremental changes
    delta_q_protons = total_flux_i * dt
    delta_mol_H = delta_q_protons / NA
    mol_H_inside_change += delta_mol_H
    q_protons = net_charge_protons[i-1] + delta_q_protons

    # Recalculate β at CURRENT pH (not initial!)
    current_H = 10**(-pH_inside[i-1])
    current_beta_soluble = 2.303 * C_buffer_M * Ka_buffer * current_H / (Ka_buffer + current_H)**2
    current_beta_water = 2.303 * (current_H + Kw / current_H)
    current_beta_total = current_beta_soluble + current_beta_water + beta_membrane

    # Update state variables
    charge_C = q_protons / NA * F
    delta_psi_mV[i] = (charge_C / C_total) * 1000

    # Incremental ΔpH (correct for nonlinear β)
    delta_pH[i] = delta_pH[i-1] - delta_mol_H / (current_beta_total * volume_L)
    pH_inside[i] = pH_initial + delta_pH[i]

    net_charge_protons[i] = q_protons
    pump_flux[i] = pump_flux_i
    leak_flux[i] = leak_flux_i

pmf_mV = delta_psi_mV - 59.16 * delta_pH

# -----------------------------
# PARAMETER SUMMARY (Toggleable)
# -----------------------------
with st.expander("📋 Simulation Parameters", expanded=False):
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
# CALCULATED RESULTS (Toggleable)
# -----------------------------
with st.expander("📊 Calculated Results", expanded=True):
    results = {
        "🔵 **Initial/Final States**": [
            ("Start pH inside", f"{pH_initial:.1f}"),
            ("Final internal pH", f"{pH_inside[-1]:.3f}"),
            ("Final ΔpH", f"{delta_pH[-1]:+.4f}"),
            ("Final Δψ", f"{delta_psi_mV[-1]:+.1f} mV"),
        ],
        "🟢 **Vesicle Properties**": [
            ("Vesicle volume", f"{volume_L*1e18:.2f} aL"),
            ("Total capacitance", f"{C_total:.2e} F"),
        ],
        "🔴 **Pump & Leak**": [
            ("Pump stopping potential", f"{psi_stop_mV:.0f} mV"),
            ("Final net flux", f"{pump_flux[-1] + leak_flux[-1]:+.1f} H⁺ s⁻¹"),
        ],
        "🟢 **Buffer Capacity**": [
            ("Soluble buffer β", f"{beta_soluble:.2e} mol L⁻¹ pH⁻¹"),
            ("Water β", f"{beta_water:.2e} mol L⁻¹ pH⁻¹"),
            ("Membrane β", f"{beta_membrane:.2e} mol L⁻¹ pH⁻¹"),
            ("Total β", f"{beta_total:.2e} mol L⁻¹ pH⁻¹"),
        ],
    }
    col1, col2 = st.columns(2)
    for i, (group, items) in enumerate(results.items()):
        with col1 if i % 2 == 0 else col2:
            st.markdown(f"<div class='result-header'>{group}</div>", unsafe_allow_html=True)
            for label, value in items:
                st.markdown(f"<div class='result-box'>{label}: {value}</div>", unsafe_allow_html=True)

# -----------------------------
# CSV EXPORT (Sidebar)
# -----------------------------
st.sidebar.header("💾 Export Data")
st.sidebar.markdown("Download simulation data as CSV to compare results across different parameter sets.")

# Create DataFrame with all simulation data
data = {
    "time_s": time,
    "delta_psi_mV": delta_psi_mV,
    "delta_pH": delta_pH,
    "pH_inside": pH_inside,
    "pmf_mV": pmf_mV,
    "pump_flux_H+s-1": pump_flux,
    "leak_flux_H+s-1": leak_flux,
    "net_flux_H+s-1": pump_flux + leak_flux,
}

# Create metadata as a list of strings (no DataFrame needed)
metadata_lines = [
    "# Proton Transport Simulation Data",
    f"# Vesicle radius: {radius_nm} nm",
    f"# Initial pH: {pH_initial}",
    f"# Buffer: {buffer_mM} mM, pKa {buffer_pKa}",
    f"# Membrane buffering: {'Yes' if include_membrane_buffer else 'No'}",
    f"# Pump: {pump_rate_max} H⁺/s, {deltaG_pump_kJ_mol} kJ/mol",
    f"# Leak: {leak_conductance} H⁺/s/mV, threshold {leak_threshold_mV} mV",
    f"# Calculated: volume={volume_L*1e18:.2f} aL, C_total={C_total:.2e} F, β_total={beta_total:.2e} mol/L/pH",
    "#",
    "# Simulation Data (time-series):",
]

# Combine metadata and data
output = StringIO()
output.write("\n".join(metadata_lines) + "\n")
df = pd.DataFrame(data)
df.to_csv(output, index=False)

# Download button
st.sidebar.download_button(
    label="📥 Download Full CSV",
    data=output.getvalue(),
    file_name=f"proton_transport_r{radius_nm}nm_pH{pH_initial}.csv",
    mime="text/csv",
    help="Includes metadata (parameters) and all time-series data"
)

# -----------------------------
# Plots
# -----------------------------
# -----------------------------
# Plots (Consistent 2-column layout)
# -----------------------------
# Row 1: Δψ and ΔpH
left, right = st.columns(2)
with left:
    fig, ax = plt.subplots(figsize=(6, 4))
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
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(time, delta_pH)
    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("Time / s")
    ax.set_ylabel("ΔpH = pH(in) − pH(start)")
    ax.set_title("Bulk pH changes much less")
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f0f2f6')
    st.pyplot(fig)

# Row 2: PMF and Absolute pH
left, right = st.columns(2)
with left:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(time, pmf_mV)
    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Approx. proton motive force / mV")
    ax.set_title("Approximate combined driving force")
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f0f2f6')
    st.pyplot(fig)

with right:
    if show_absolute_pH:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(time, pH_inside)
        ax.axhline(pH_initial, linewidth=0.8)
        ax.set_xlabel("Time / s")
        ax.set_ylabel("Internal pH")
        ax.set_title("Absolute internal pH")
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f0f2f6')
        st.pyplot(fig)

# Row 3: Fluxes and Leak Curve
left, right = st.columns(2)
with left:
    if show_fluxes:
        fig, ax = plt.subplots(figsize=(6, 4))
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

with right:
    if show_leak_curve:
        psi_range = np.linspace(-250, 250, 1000)
        leak_range = soft_threshold_leak(
            psi_range,
            leak_conductance,
            leak_threshold_mV,
            leak_softness_mV
        )
        fig, ax = plt.subplots(figsize=(6, 4))
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