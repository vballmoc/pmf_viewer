import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ------------------------------------------------------------
# Educational model:
# Valinomycin as an instantaneous K+ Nernst clamp.
#
# Version 5:
# - Outside K+ concentration is fixed, as in a large external reservoir.
# - Vesicle size is entered as outer diameter; aqueous lumen deducts bilayer thickness.
# - Exact number inputs are used for key scientific parameters.
# - Valinomycin is assumed to be much faster than the proton/charge load.
# - Therefore Δψ is always the instantaneous K+ Nernst potential.
# - The direction of the proton/charge load can be selected explicitly.
#
# Convention:
#   Δψ = ψ_inside - ψ_outside
#   H_flux_inside > 0 means positive charge enters the vesicle
#   K_flux_inside > 0 means K+ enters the vesicle
# ------------------------------------------------------------

R = 8.314462618
F = 96485.33212
NA = 6.02214076e23

st.set_page_config(page_title="Valinomycin Nernst clamp", layout="wide")

st.title("Valinomycin as a K⁺ Nernst clamp")

st.markdown("""
This model assumes that valinomycin is **not rate-limiting**.  
At every time point, K⁺ redistributes fast enough that the membrane potential is
equal to the instantaneous K⁺ Nernst potential.

The outside K⁺ concentration is treated as a large reservoir and kept fixed.
Only the finite K⁺ content inside the vesicle changes.
""")

st.sidebar.header("Vesicle")

outer_diameter_nm = st.sidebar.number_input(
    "Outer vesicle diameter / nm",
    min_value=5.0, max_value=5000.0, value=100.0, step=1.0,
)

bilayer_thickness_nm = st.sidebar.number_input(
    "Bilayer thickness / nm",
    min_value=0.0, max_value=20.0, value=4.5, step=0.1,
)

temperature_C = st.sidebar.number_input(
    "Temperature / °C",
    min_value=0.0, max_value=80.0, value=25.0, step=1.0,
)

st.sidebar.header("K⁺ concentrations")

K_in_initial_mM = st.sidebar.number_input(
    "[K⁺] inside initially / mM",
    min_value=0.001, max_value=2000.0, value=50.0, step=1.0,
)

K_out_fixed_mM = st.sidebar.number_input(
    "[K⁺] outside fixed / mM",
    min_value=0.001, max_value=2000.0, value=5.0, step=1.0,
)

st.sidebar.header("Load consuming or reinforcing Δψ")

load_direction = st.sidebar.radio(
    "Direction of positive-charge flux",
    [
        "dissipate current Δψ",
        "positive charge into vesicle",
        "positive charge out of vesicle",
    ],
    index=0,
    help=(
        "For negative-inside Δψ, a dissipating proton flux enters the vesicle. "
        "For positive-inside Δψ, a dissipating proton flux leaves the vesicle."
    ),
)

load_type = st.sidebar.radio(
    "Load model",
    ["constant flux", "potential-dependent flux"],
    index=1,
)

constant_charge_flux = st.sidebar.number_input(
    "Constant load / charges s⁻¹",
    min_value=0.0, max_value=1_000_000.0, value=1500.0, step=100.0,
)

max_potential_dependent_flux = st.sidebar.number_input(
    "Load at initial |Δψ| / charges s⁻¹",
    min_value=0.0, max_value=1_000_000.0, value=1500.0, step=100.0,
)

load_start_s = st.sidebar.number_input(
    "Load starts after / s",
    min_value=0.0, max_value=100000.0, value=0.0, step=1.0,
)

duration_s = st.sidebar.number_input(
    "Simulation time / s",
    min_value=0.1, max_value=100000.0, value=60.0, step=1.0,
)

st.sidebar.header("Stopping and display")

min_abs_potential_mV = st.sidebar.number_input(
    "Relevant Δψ threshold / mV",
    min_value=0.0, max_value=500.0, value=50.0, step=5.0,
)

show_amounts = st.sidebar.checkbox("Show K⁺ ion numbers", value=True)
show_direction_explanation = st.sidebar.checkbox("Show sign convention explanation", value=True)

# Geometry
T = temperature_C + 273.15
inner_diameter_nm = max(outer_diameter_nm - 2 * bilayer_thickness_nm, 0.001)
radius_m = (inner_diameter_nm / 2) * 1e-9
volume_in_L = (4/3) * np.pi * radius_m**3 * 1000

K_in_count = K_in_initial_mM * 1e-3 * volume_in_L * NA

def concentration_mM(count, volume_L):
    return count / NA / volume_L * 1000

def nernst_K_mV(K_out_mM, K_in_mM):
    K_out_mM = max(K_out_mM, 1e-30)
    K_in_mM = max(K_in_mM, 1e-30)
    return (R * T / F) * np.log(K_out_mM / K_in_mM) * 1000

initial_psi = nernst_K_mV(K_out_fixed_mM, K_in_initial_mM)
K_in_equilibrium_count = K_out_fixed_mM * 1e-3 * volume_in_L * NA

# Time integration
n_points = max(1000, min(30000, int(duration_s * 20)))
time = np.linspace(0, duration_s, n_points)
dt = time[1] - time[0]

K_in_mM = np.zeros_like(time)
psi_mV = np.zeros_like(time)
charge_flux_inside = np.zeros_like(time)
K_compensating_flux = np.zeros_like(time)
K_in_counts = np.zeros_like(time)

for i in range(n_points):
    K_in_mM[i] = concentration_mM(K_in_count, volume_in_L)
    psi_mV[i] = nernst_K_mV(K_out_fixed_mM, K_in_mM[i])
    K_in_counts[i] = K_in_count

    if i == n_points - 1:
        break

    if time[i] < load_start_s:
        J_mag = 0.0
    else:
        if load_type == "constant flux":
            J_mag = constant_charge_flux
        else:
            if abs(initial_psi) > 1e-12:
                J_mag = max_potential_dependent_flux * abs(psi_mV[i]) / abs(initial_psi)
            else:
                J_mag = 0.0

    # Positive-charge flux into vesicle
    if load_direction == "dissipate current Δψ":
        # negative-inside: positive charge enters; positive-inside: positive charge leaves
        J_charge_inside = -np.sign(psi_mV[i]) * J_mag
    elif load_direction == "positive charge into vesicle":
        J_charge_inside = +J_mag
    else:
        J_charge_inside = -J_mag

    # Instantaneous valinomycin compensation: opposite K+ movement
    J_K = -J_charge_inside

    # Prevent negative internal K+ if K+ leaves.
    if J_K < 0:
        J_K = max(J_K, -K_in_count / dt)

    # In dissipative mode, stop at K_in = K_out because the useful gradient is spent.
    if load_direction == "dissipate current Δψ":
        if initial_psi < 0:
            max_leave_to_equilibrium = max((K_in_count - K_in_equilibrium_count) / dt, 0.0)
            J_K = max(J_K, -max_leave_to_equilibrium)
        elif initial_psi > 0:
            max_enter_to_equilibrium = max((K_in_equilibrium_count - K_in_count) / dt, 0.0)
            J_K = min(J_K, max_enter_to_equilibrium)
        else:
            J_K = 0.0

    K_in_count += J_K * dt
    K_in_count = max(K_in_count, 1e-30)

    charge_flux_inside[i] = J_charge_inside
    K_compensating_flux[i] = J_K

charge_flux_inside[-1] = charge_flux_inside[-2]
K_compensating_flux[-1] = K_compensating_flux[-2]

after_load = time >= load_start_s
above = np.abs(psi_mV) >= min_abs_potential_mV
indices = np.where(after_load & above)[0]

if len(indices) == 0:
    lifetime_text = "not above threshold"
else:
    lifetime_text = f"{time[indices[-1]] - load_start_s:.1f} s"

usable_K_ions_to_equilibrium = abs(K_in_counts[0] - K_in_equilibrium_count)

# Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Initial Δψ = E_K", f"{psi_mV[0]:+.1f} mV")
with col2:
    st.metric("Final Δψ = E_K", f"{psi_mV[-1]:+.1f} mV")
with col3:
    st.metric("Lifetime above threshold", lifetime_text)
with col4:
    st.metric("Aqueous diameter", f"{inner_diameter_nm:.1f} nm")

col5, col6, col7, col8 = st.columns(4)

with col5:
    st.metric("Initial K⁺ inside", f"{K_in_counts[0]:.0f} ions")
with col6:
    st.metric("K⁺ to K_in = K_out", f"{usable_K_ions_to_equilibrium:.0f} ions")
with col7:
    st.metric("Initial [K⁺]in", f"{K_in_mM[0]:.2f} mM")
with col8:
    st.metric("Final [K⁺]in", f"{K_in_mM[-1]:.2f} mM")

st.info(
    f"Outside [K⁺] is fixed at {K_out_fixed_mM:.2f} mM. "
    f"For a {outer_diameter_nm:.1f} nm outer-diameter vesicle with a "
    f"{inner_diameter_nm:.1f} nm aqueous lumen, the initial internal K⁺ pool is "
    f"about {K_in_counts[0]:.0f} ions."
)

if show_direction_explanation:
    st.subheader("Direction convention")

    if psi_mV[0] < 0:
        case_text = (
            "Here [K⁺]in > [K⁺]out, so E_K is negative: the inside is negative. "
            "A dissipating positive-charge flux therefore goes into the vesicle, "
            "and valinomycin compensates by moving K⁺ out."
        )
    elif psi_mV[0] > 0:
        case_text = (
            "Here [K⁺]out > [K⁺]in, so E_K is positive: the inside is positive. "
            "A dissipating positive-charge flux therefore goes out of the vesicle, "
            "and valinomycin compensates by moving K⁺ in."
        )
    else:
        case_text = "Here [K⁺]in equals [K⁺]out, so E_K is zero."

    st.markdown(f"""
{case_text}

Sign convention:

- positive `charge flux` = positive charge enters the vesicle;
- positive `K⁺ flux` = K⁺ enters the vesicle;
- instantaneous valinomycin compensation means:

`K⁺ flux = − charge flux`
""")

# Plots
left, right = st.columns(2)

with left:
    fig, ax = plt.subplots()
    ax.plot(time, psi_mV)
    ax.axhline(0, linewidth=0.8)
    ax.axhline(min_abs_potential_mV, linestyle=":", linewidth=0.8)
    ax.axhline(-min_abs_potential_mV, linestyle=":", linewidth=0.8)
    ax.axvline(load_start_s, linestyle="--", linewidth=0.8, label="load starts")
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Δψ = E_K / mV")
    ax.set_title("Membrane potential clamped to instantaneous K⁺ Nernst potential")
    ax.legend()
    st.pyplot(fig)

with right:
    fig, ax = plt.subplots()
    ax.plot(time, K_in_mM, label="[K⁺] inside")
    ax.axhline(K_out_fixed_mM, linestyle="--", linewidth=0.8, label="[K⁺] outside fixed")
    ax.axvline(load_start_s, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Time / s")
    ax.set_ylabel("[K⁺] / mM")
    ax.set_title("Internal K⁺ changes while outside is fixed")
    ax.legend()
    st.pyplot(fig)

fig, ax = plt.subplots()
ax.plot(time, charge_flux_inside, label="positive-charge flux into vesicle")
ax.plot(time, K_compensating_flux, label="compensating K⁺ flux into vesicle")
ax.axhline(0, linewidth=0.8)
ax.axvline(load_start_s, linestyle="--", linewidth=0.8)
ax.set_xlabel("Time / s")
ax.set_ylabel("Flux / ions s⁻¹")
ax.set_title("Valinomycin compensation works in either direction")
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

st.subheader("Teaching interpretation")

st.markdown(f"""
At the start:

`Δψ = E_K = {psi_mV[0]:+.1f} mV`

Because valinomycin is assumed to be fast, every imposed positive-charge movement
is rapidly compensated by an opposite K⁺ movement. This is true in both directions:

- if the inside is negative, a dissipating proton/charge flux goes inward and K⁺ moves outward;
- if the inside is positive, a dissipating proton/charge flux goes outward and K⁺ moves inward.

The Nernst potential is then recalculated from the new internal K⁺ concentration.
""")

st.subheader("Important simplifications")

st.markdown("""
- Valinomycin kinetics are ignored; Δψ is assumed to equal the K⁺ Nernst potential at all times.
- Outside K⁺ concentration is fixed.
- One imposed positive charge is compensated by one K⁺ ion in the opposite direction.
- Osmotic effects, electroneutrality constraints, counterions, and volume changes are not modelled.
- This is a teaching model for the finite energy content of a K⁺ gradient.
""")
