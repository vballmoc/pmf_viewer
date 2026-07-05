import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ------------------------------------------------------------
# Educational model:
# Valinomycin as an instantaneous K+ Nernst clamp.
#
# Assumption:
#   Valinomycin is much faster than the proton/charge load.
#   Therefore Δψ is always equal to the instantaneous K+ Nernst potential.
#
# Consequence:
#   A proton leak or ATP synthase-like charge flux dissipates Δψ.
#   Valinomycin immediately compensates by moving K+.
#   This consumes the finite K+ gradient.
#
# This is a thermodynamic teaching model, not a kinetic valinomycin model.
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

A proton leak or ATP synthase-like load can therefore be supported only as long
as the finite K⁺ gradient remains.
""")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Vesicle and external volume")

radius_nm = st.sidebar.slider("Vesicle radius / nm", 25, 1000, 100, step=25)
outside_volume_ratio = st.sidebar.slider(
    "Outside volume / inside volume",
    1, 100000, 1000, step=10,
    help="Large values mimic a large external reservoir."
)
temperature_C = st.sidebar.slider("Temperature / °C", 5.0, 45.0, 25.0, step=1.0)

st.sidebar.header("Initial K⁺ concentrations")

K_in_initial_mM = st.sidebar.slider("[K⁺] inside initially / mM", 0.1, 500.0, 150.0, step=0.1)
K_out_initial_mM = st.sidebar.slider("[K⁺] outside initially / mM", 0.1, 500.0, 5.0, step=0.1)

st.sidebar.header("Load consuming Δψ")

load_type = st.sidebar.radio(
    "Load model",
    ["constant H⁺ flux", "potential-dependent H⁺ flux"],
    index=0
)

constant_H_flux = st.sidebar.slider(
    "Constant H⁺ flux / s⁻¹",
    0.0, 10000.0, 500.0, step=10.0,
    help="Positive-charge flux consuming the membrane potential."
)

potential_load_conductance = st.sidebar.slider(
    "Potential-dependent load / H⁺ s⁻¹ per 100 mV",
    0.0, 10000.0, 1000.0, step=10.0,
    help="Only used for the potential-dependent load model."
)

load_start_s = st.sidebar.slider("Load starts after / s", 0.0, 600.0, 30.0, step=1.0)
duration_s = st.sidebar.slider("Simulation time / s", 1.0, 7200.0, 1200.0, step=1.0)

st.sidebar.header("Stopping criterion")

min_abs_potential_mV = st.sidebar.slider(
    "Relevant Δψ threshold / mV",
    0.0, 200.0, 50.0, step=5.0,
    help="Used only for the displayed lifetime estimate."
)

st.sidebar.header("Display")

show_amounts = st.sidebar.checkbox("Show K⁺ ion numbers", value=False)
show_energy = st.sidebar.checkbox("Show approximate gradient energy", value=True)

# -----------------------------
# Geometry
# -----------------------------
T = temperature_C + 273.15

radius_m = radius_nm * 1e-9
volume_in_L = (4/3) * np.pi * radius_m**3 * 1000
volume_out_L = volume_in_L * outside_volume_ratio

# Initial K+ molecule counts
K_in_count = K_in_initial_mM * 1e-3 * volume_in_L * NA
K_out_count = K_out_initial_mM * 1e-3 * volume_out_L * NA
K_total_count = K_in_count + K_out_count

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


def equilibrium_counts_for_ratio(total_count, volume_in, volume_out):
    """
    Final equilibrium when [K+]in = [K+]out.
    """
    total_volume = volume_in + volume_out
    c_final_mol_L = total_count / NA / total_volume
    K_in_eq = c_final_mol_L * volume_in * NA
    K_out_eq = c_final_mol_L * volume_out * NA
    return K_in_eq, K_out_eq


def gradient_free_energy_relative_to_equilibrium(Kin, Kout, Kin_eq, Kout_eq):
    """
    Approximate mixing free energy relative to final equal-concentration state:
        G = RT * sum n_i ln(c_i / c_eq)
    Returns J.
    """
    c_in = max(Kin / NA / volume_in_L, 1e-30)
    c_out = max(Kout / NA / volume_out_L, 1e-30)
    c_eq = (Kin + Kout) / NA / (volume_in_L + volume_out_L)

    n_in = Kin / NA
    n_out = Kout / NA

    return R * T * (n_in * np.log(c_in / c_eq) + n_out * np.log(c_out / c_eq))


K_in_eq, K_out_eq = equilibrium_counts_for_ratio(K_total_count, volume_in_L, volume_out_L)

# -----------------------------
# Time integration
# -----------------------------
n_points = max(1000, min(30000, int(duration_s * 10)))
time = np.linspace(0, duration_s, n_points)
dt = time[1] - time[0]

K_in_mM = np.zeros_like(time)
K_out_mM = np.zeros_like(time)
psi_mV = np.zeros_like(time)
H_load_flux = np.zeros_like(time)
K_compensating_flux = np.zeros_like(time)
K_in_counts = np.zeros_like(time)
K_out_counts = np.zeros_like(time)
energy_J = np.zeros_like(time)

lifetime_above_threshold = None

for i in range(n_points):
    K_in_mM[i] = concentration_mM(K_in_count, volume_in_L)
    K_out_mM[i] = concentration_mM(K_out_count, volume_out_L)
    psi_mV[i] = nernst_K_mV(K_out_mM[i], K_in_mM[i])

    K_in_counts[i] = K_in_count
    K_out_counts[i] = K_out_count
    energy_J[i] = gradient_free_energy_relative_to_equilibrium(
        K_in_count, K_out_count, K_in_eq, K_out_eq
    )

    if i == n_points - 1:
        break

    if time[i] < load_start_s:
        J_H = 0.0
    else:
        if load_type == "constant H⁺ flux":
            # Charge leak consumes Δψ by moving positive charge downhill.
            # If inside is negative, H+ enters vesicle, compensated by K+ leaving.
            # If inside is positive, H+ leaves vesicle, compensated by K+ entering.
            J_H = constant_H_flux * np.sign(abs(psi_mV[i]))
        else:
            # Flux scales with magnitude of Δψ.
            J_H = potential_load_conductance * abs(psi_mV[i]) / 100.0

    # Determine direction of compensating K+ movement.
    #
    # Convention:
    #   J_K > 0 means K+ enters vesicle.
    #
    # If K_in > K_out, E_K is negative. Valinomycin makes inside negative.
    # A proton load discharges that by bringing positive charge in.
    # To restore E_K, K+ must leave: J_K < 0.
    #
    # Thus compensating K+ flux has the sign of E_K:
    #   E_K negative -> K+ leaves -> negative J_K
    #   E_K positive -> K+ enters -> positive J_K
    J_K = np.sign(psi_mV[i]) * J_H

    # Do not move beyond concentration equilibrium.
    # The relevant endpoint is [K+]in = [K+]out, not zero K+.
    if J_K < 0:
        # K+ leaves inside.
        max_leave_to_eq = max((K_in_count - K_in_eq) / dt, 0.0)
        J_K = max(J_K, -max_leave_to_eq)
    elif J_K > 0:
        # K+ enters inside.
        max_enter_to_eq = max((K_in_eq - K_in_count) / dt, 0.0)
        J_K = min(J_K, max_enter_to_eq)

    K_in_count += J_K * dt
    K_out_count -= J_K * dt

    H_load_flux[i] = J_H
    K_compensating_flux[i] = J_K

# Fill last values
H_load_flux[-1] = H_load_flux[-2]
K_compensating_flux[-1] = K_compensating_flux[-2]

# Lifetime estimate after load starts
after_load = time >= load_start_s
above = np.abs(psi_mV) >= min_abs_potential_mV
indices = np.where(after_load & above)[0]

if len(indices) == 0:
    lifetime_text = "not above threshold after load starts"
else:
    last_above = indices[-1]
    lifetime = time[last_above] - load_start_s
    lifetime_text = f"{lifetime:.1f} s"

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
    st.metric("Vesicle volume", f"{volume_in_L*1e18:.2f} aL")

col5, col6, col7, col8 = st.columns(4)

with col5:
    st.metric("Initial [K⁺]in", f"{K_in_mM[0]:.2f} mM")
with col6:
    st.metric("Final [K⁺]in", f"{K_in_mM[-1]:.2f} mM")
with col7:
    st.metric("Initial [K⁺]out", f"{K_out_mM[0]:.2f} mM")
with col8:
    st.metric("Final [K⁺]out", f"{K_out_mM[-1]:.2f} mM")

st.info(
    "Assumption: valinomycin is infinitely fast relative to the load. "
    "Therefore Δψ is always recalculated from the current K⁺ concentrations using the Nernst equation."
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
    if constant_H_flux > 0 or potential_load_conductance > 0:
        ax.axvline(load_start_s, linestyle="--", linewidth=0.8, label="load starts")
        ax.legend()
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Δψ = E_K / mV")
    ax.set_title("Membrane potential clamped to K⁺ Nernst potential")
    st.pyplot(fig)

with right:
    fig, ax = plt.subplots()
    ax.plot(time, K_in_mM, label="[K⁺] inside")
    ax.plot(time, K_out_mM, label="[K⁺] outside")
    if constant_H_flux > 0 or potential_load_conductance > 0:
        ax.axvline(load_start_s, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Time / s")
    ax.set_ylabel("[K⁺] / mM")
    ax.set_title("The K⁺ gradient is consumed")
    ax.legend()
    st.pyplot(fig)

fig, ax = plt.subplots()
ax.plot(time, H_load_flux, label="H⁺/charge load")
ax.plot(time, K_compensating_flux, label="compensating K⁺ flux")
ax.axhline(0, linewidth=0.8)
if constant_H_flux > 0 or potential_load_conductance > 0:
    ax.axvline(load_start_s, linestyle="--", linewidth=0.8)
ax.set_xlabel("Time / s")
ax.set_ylabel("Flux / ions s⁻¹")
ax.set_title("Each dissipating charge is compensated by K⁺ movement")
ax.legend()
st.pyplot(fig)

if show_energy:
    fig, ax = plt.subplots()
    ax.plot(time, energy_J)
    ax.axhline(0, linewidth=0.8)
    if constant_H_flux > 0 or potential_load_conductance > 0:
        ax.axvline(load_start_s, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Approx. K⁺ gradient free energy / J")
    ax.set_title("Free energy stored in the finite K⁺ gradient")
    st.pyplot(fig)

if show_amounts:
    fig, ax = plt.subplots()
    ax.plot(time, K_in_counts, label="K⁺ ions inside")
    ax.plot(time, K_out_counts, label="K⁺ ions outside")
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Number of K⁺ ions")
    ax.set_title("K⁺ ion numbers")
    ax.legend()
    st.pyplot(fig)

# -----------------------------
# Teaching notes
# -----------------------------
st.subheader("Teaching interpretation")

st.markdown(f"""
At the start, the K⁺ gradient gives:

`Δψ = E_K = {psi_mV[0]:+.1f} mV`

Because valinomycin is assumed to be fast, every dissipating charge is rapidly
compensated by K⁺ movement. This keeps Δψ at the Nernst value corresponding to
the **current** K⁺ concentrations.

The important consequence is that the membrane potential does not simply
"run down" by capacitor discharge. Instead, the K⁺ gradient is spent. As the
inside and outside K⁺ concentrations approach each other, the Nernst potential
collapses.
""")

st.subheader("Important simplifications")

st.markdown("""
- Valinomycin kinetics are ignored; the membrane is assumed to be always at K⁺ Nernst equilibrium.
- The proton/charge load is simplified as an integer charge flux.
- One dissipating positive charge is compensated by one K⁺ moving in the direction required to restore E_K.
- Osmotic effects, electroneutrality constraints, counterions, and volume changes are not modelled.
- This is a teaching model for the finite energy content of a K⁺ gradient.
""")
