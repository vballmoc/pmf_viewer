import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ------------------------------------------------------------
# Educational model:
# Valinomycin as an instantaneous K+ Nernst clamp.
#
# Version 3:
# - Outside K+ concentration is held fixed, as in a large external reservoir.
# - Vesicle size is entered as diameter, not radius.
# - Valinomycin is assumed to be much faster than the load.
# - Therefore Δψ is always the instantaneous K+ Nernst potential.
# - A proton/charge load is compensated by K+ movement, consuming the finite
#   internal K+ gradient.
# - A potential-dependent load can be used to make the flux decline as Δψ collapses.
#
# This is a teaching model, not a full electrodiffusion simulation.
# ------------------------------------------------------------

R = 8.314462618          # J mol^-1 K^-1
F = 96485.33212          # C mol^-1
NA = 6.02214076e23       # mol^-1

st.set_page_config(page_title="Valinomycin Nernst clamp", layout="wide")

st.title("Valinomycin as a K⁺ Nernst clamp")

st.markdown("""
This model assumes that valinomycin is **not rate-limiting**.  
At every time point, K⁺ redistributes fast enough that the membrane potential is
equal to the instantaneous K⁺ Nernst potential.

The outside K⁺ concentration is treated as a large reservoir and kept fixed.
Only the finite K⁺ content inside the vesicle is consumed.
""")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Vesicle")

diameter_nm = st.sidebar.slider("Vesicle diameter / nm", 25, 1000, 100, step=25)
temperature_C = st.sidebar.slider("Temperature / °C", 5.0, 45.0, 25.0, step=1.0)

st.sidebar.header("K⁺ concentrations")

K_in_initial_mM = st.sidebar.slider("[K⁺] inside initially / mM", 0.1, 500.0, 50.0, step=0.1)
K_out_fixed_mM = st.sidebar.slider("[K⁺] outside fixed / mM", 0.1, 500.0, 5.0, step=0.1)

st.sidebar.header("Load consuming Δψ")

load_type = st.sidebar.radio(
    "Load model",
    ["constant H⁺/charge flux", "potential-dependent H⁺/charge flux"],
    index=1
)

constant_H_flux = st.sidebar.slider(
    "Constant load / charges s⁻¹",
    0.0, 10000.0, 1500.0, step=10.0,
    help="Used only for constant load. This remains constant until the K+ gradient is spent."
)

max_potential_dependent_flux = st.sidebar.slider(
    "Load at initial |Δψ| / charges s⁻¹",
    0.0, 10000.0, 1500.0, step=10.0,
    help="Used only for potential-dependent load. The load declines as |Δψ| declines."
)

load_start_s = st.sidebar.slider("Load starts after / s", 0.0, 600.0, 0.0, step=1.0)
duration_s = st.sidebar.slider("Simulation time / s", 1.0, 7200.0, 60.0, step=1.0)

st.sidebar.header("Stopping and display")

min_abs_potential_mV = st.sidebar.slider(
    "Relevant Δψ threshold / mV",
    0.0, 200.0, 50.0, step=5.0
)

show_amounts = st.sidebar.checkbox("Show K⁺ ion numbers", value=True)

# -----------------------------
# Geometry
# -----------------------------
T = temperature_C + 273.15

radius_m = (diameter_nm / 2) * 1e-9
volume_in_L = (4/3) * np.pi * radius_m**3 * 1000

# Initial K+ molecule count inside
K_in_count = K_in_initial_mM * 1e-3 * volume_in_L * NA

# -----------------------------
# Helper functions
# -----------------------------
def concentration_mM(count, volume_L):
    return count / NA / volume_L * 1000


def nernst_K_mV(K_out_mM, K_in_mM):
    """K+ Nernst potential, inside relative to outside."""
    K_out_mM = max(K_out_mM, 1e-30)
    K_in_mM = max(K_in_mM, 1e-30)
    return (R * T / F) * np.log(K_out_mM / K_in_mM) * 1000


initial_psi = nernst_K_mV(K_out_fixed_mM, K_in_initial_mM)

# Equilibrium endpoint with fixed outside concentration:
# the K+ gradient is exhausted when K_in = K_out_fixed.
K_in_equilibrium_count = K_out_fixed_mM * 1e-3 * volume_in_L * NA

# Direction of K movement needed to collapse the gradient
# If E_K is negative, K+ leaves the vesicle.
# If E_K is positive, K+ enters the vesicle.
gradient_direction = np.sign(initial_psi)

# -----------------------------
# Time integration
# -----------------------------
n_points = max(1000, min(30000, int(duration_s * 20)))
time = np.linspace(0, duration_s, n_points)
dt = time[1] - time[0]

K_in_mM = np.zeros_like(time)
psi_mV = np.zeros_like(time)
H_load_flux = np.zeros_like(time)
K_compensating_flux = np.zeros_like(time)
K_in_counts = np.zeros_like(time)

for i in range(n_points):
    K_in_mM[i] = concentration_mM(K_in_count, volume_in_L)
    psi_mV[i] = nernst_K_mV(K_out_fixed_mM, K_in_mM[i])
    K_in_counts[i] = K_in_count

    if i == n_points - 1:
        break

    if time[i] < load_start_s:
        J_H = 0.0
    else:
        if load_type == "constant H⁺/charge flux":
            J_H = constant_H_flux if abs(psi_mV[i]) > 1e-12 else 0.0
        else:
            # Declining load: scaled to equal the selected value at the initial |Δψ|.
            # As |Δψ| collapses, the load approaches zero.
            if abs(initial_psi) > 1e-12:
                J_H = max_potential_dependent_flux * abs(psi_mV[i]) / abs(initial_psi)
            else:
                J_H = 0.0

    # Compensating K+ flux.
    # Convention:
    #   J_K > 0 means K+ enters vesicle.
    #
    # If ψ is negative, the load brings positive charge in and K+ must leave:
    #   J_K negative.
    # If ψ is positive, K+ must enter:
    #   J_K positive.
    J_K = np.sign(psi_mV[i]) * J_H

    # Do not move beyond K_in = K_out_fixed.
    if initial_psi < 0:
        # K+ leaves until K_in reaches K_out.
        max_leave_to_equilibrium = max((K_in_count - K_in_equilibrium_count) / dt, 0.0)
        J_K = max(J_K, -max_leave_to_equilibrium)
    elif initial_psi > 0:
        # K+ enters until K_in reaches K_out.
        max_enter_to_equilibrium = max((K_in_equilibrium_count - K_in_count) / dt, 0.0)
        J_K = min(J_K, max_enter_to_equilibrium)
    else:
        J_K = 0.0

    K_in_count += J_K * dt
    K_in_count = max(K_in_count, 1e-30)

    H_load_flux[i] = J_H
    K_compensating_flux[i] = J_K

H_load_flux[-1] = H_load_flux[-2]
K_compensating_flux[-1] = K_compensating_flux[-2]

# -----------------------------
# Lifetime estimates
# -----------------------------
after_load = time >= load_start_s
above = np.abs(psi_mV) >= min_abs_potential_mV
indices = np.where(after_load & above)[0]

if len(indices) == 0:
    lifetime_text = "not above threshold"
else:
    lifetime = time[indices[-1]] - load_start_s
    lifetime_text = f"{lifetime:.1f} s"

usable_K_ions = abs(K_in_counts[0] - K_in_equilibrium_count)

if load_type == "constant H⁺/charge flux" and constant_H_flux > 0:
    rough_full_collapse_time = usable_K_ions / constant_H_flux
    rough_text = f"{rough_full_collapse_time:.1f} s"
elif load_type == "potential-dependent H⁺/charge flux" and max_potential_dependent_flux > 0:
    rough_text = "declines with Δψ"
else:
    rough_text = "no load"

# -----------------------------
# Metrics
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Initial Δψ = E_K", f"{psi_mV[0]:+.1f} mV")
with col2:
    st.metric("Final Δψ = E_K", f"{psi_mV[-1]:+.1f} mV")
with col3:
    st.metric("Lifetime above threshold", lifetime_text)
with col4:
    st.metric("Vesicle volume", f"{volume_in_L*1e18:.3f} aL")

col5, col6, col7, col8 = st.columns(4)

with col5:
    st.metric("Initial K⁺ inside", f"{K_in_counts[0]:.0f} ions")
with col6:
    st.metric("Usable K⁺ gradient", f"{usable_K_ions:.0f} ions")
with col7:
    st.metric("Initial [K⁺]in", f"{K_in_mM[0]:.2f} mM")
with col8:
    st.metric("Final [K⁺]in", f"{K_in_mM[-1]:.2f} mM")

st.info(
    f"For a {diameter_nm:.0f} nm diameter vesicle at {K_in_initial_mM:.1f} mM K⁺, "
    f"the model contains about {K_in_counts[0]:.0f} K⁺ ions inside. "
    f"With fixed outside [K⁺] = {K_out_fixed_mM:.1f} mM, only about "
    f"{usable_K_ions:.0f} ions are available to spend before K_in reaches K_out. "
    f"Rough full-collapse estimate: {rough_text}."
)

# -----------------------------
# Plots
# -----------------------------
left, right = st.columns(2)

with left:
    fig, ax = plt.subplots()
    ax.plot(time, psi_mV)
    ax.axhline(0, linewidth=0.8)
    ax.axhline(min_abs_potential_mV, linestyle=":", linewidth=0.8)
    ax.axhline(-min_abs_potential_mV, linestyle=":", linewidth=0.8)
    if constant_H_flux > 0 or max_potential_dependent_flux > 0:
        ax.axvline(load_start_s, linestyle="--", linewidth=0.8, label="load starts")
        ax.legend()
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Δψ = E_K / mV")
    ax.set_title("Membrane potential clamped to instantaneous K⁺ Nernst potential")
    st.pyplot(fig)

with right:
    fig, ax = plt.subplots()
    ax.plot(time, K_in_mM, label="[K⁺] inside")
    ax.axhline(K_out_fixed_mM, linestyle="--", linewidth=0.8, label="[K⁺] outside fixed")
    if constant_H_flux > 0 or max_potential_dependent_flux > 0:
        ax.axvline(load_start_s, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Time / s")
    ax.set_ylabel("[K⁺] / mM")
    ax.set_title("Internal K⁺ approaches the fixed external concentration")
    ax.legend()
    st.pyplot(fig)

fig, ax = plt.subplots()
ax.plot(time, H_load_flux, label="H⁺/charge load")
ax.plot(time, K_compensating_flux, label="compensating K⁺ flux")
ax.axhline(0, linewidth=0.8)
if constant_H_flux > 0 or max_potential_dependent_flux > 0:
    ax.axvline(load_start_s, linestyle="--", linewidth=0.8)
ax.set_xlabel("Time / s")
ax.set_ylabel("Flux / ions s⁻¹")
ax.set_title("Each dissipating charge is compensated by K⁺ movement")
ax.legend()
st.pyplot(fig)

if show_amounts:
    fig, ax = plt.subplots()
    ax.plot(time, K_in_counts, label="K⁺ ions inside")
    ax.axhline(K_in_equilibrium_count, linestyle="--", linewidth=0.8, label="K⁺ ions at K_in = K_out")
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Number of K⁺ ions")
    ax.set_title("Finite K⁺ pool inside the vesicle")
    ax.legend()
    st.pyplot(fig)

# -----------------------------
# Teaching notes
# -----------------------------
st.subheader("Teaching interpretation")

st.markdown(f"""
At the start:

`Δψ = E_K = {psi_mV[0]:+.1f} mV`

Because valinomycin is assumed to be fast, every dissipating charge is rapidly
compensated by K⁺ movement. The outside concentration is fixed, so the only
finite resource is the internal K⁺ excess or deficit.

For this vesicle, the initial internal K⁺ pool is approximately
**{K_in_counts[0]:.0f} ions**, and the usable K⁺ gradient before
`[K⁺]in = [K⁺]out` is approximately **{usable_K_ions:.0f} ions**.

With a constant load, the compensating K⁺ flux is also constant until the
gradient is nearly exhausted. With a potential-dependent load, the flux declines
as the Nernst potential collapses.
""")

st.subheader("Important simplifications")

st.markdown("""
- Valinomycin kinetics are ignored; Δψ is assumed to equal the K⁺ Nernst potential at all times.
- Outside K⁺ concentration is fixed.
- One dissipating positive charge is compensated by one K⁺ movement.
- Osmotic effects, electroneutrality constraints, counterions, and volume changes are not modelled.
- This is a teaching model for the finite energy content of a K⁺ gradient.
""")
