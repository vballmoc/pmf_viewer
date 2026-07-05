import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ------------------------------------------------------------
# Educational model:
# Valinomycin-induced membrane potential from a K+ gradient.
#
# Core idea:
#   Valinomycin makes the membrane selectively permeable to K+.
#   K+ moves until the membrane potential approaches the K+ Nernst potential.
#
# Optional extensions:
# - Simple electrical leak / ATP synthase-like discharge
# - Goldman-Hodgkin-Katz mode for multi-ion permeability
#
# This is a didactic model, not a full electrodiffusion simulation.
# ------------------------------------------------------------

R = 8.314462618          # J mol^-1 K^-1
F = 96485.33212          # C mol^-1
NA = 6.02214076e23       # mol^-1

st.set_page_config(page_title="Valinomycin and membrane potential", layout="wide")

st.title("Valinomycin: membrane potential from a K⁺ gradient")

st.markdown("""
Valinomycin is a K⁺ ionophore. It does not create energy by itself; instead, it
allows K⁺ to move across the membrane until the electrical potential balances
the K⁺ concentration gradient.

This app compares three levels of description:

1. **Nernst only**: ideal K⁺-selective membrane.
2. **Dynamic K⁺ model**: valinomycin drives Δψ toward the K⁺ Nernst potential.
3. **Goldman mode**: several ions contribute according to their relative permeabilities.
""")

# -----------------------------
# Sidebar parameters
# -----------------------------
st.sidebar.header("Geometry and capacitance")

radius_nm = st.sidebar.slider("Vesicle radius / nm", 25, 1000, 100, step=25)
capacitance_uF_cm2 = st.sidebar.slider("Membrane capacitance / µF cm⁻²", 0.2, 2.0, 1.0, step=0.1)
temperature_C = st.sidebar.slider("Temperature / °C", 5.0, 45.0, 25.0, step=1.0)

st.sidebar.header("K⁺ concentrations")

K_in_mM = st.sidebar.slider("[K⁺] inside / mM", 1.0, 500.0, 150.0, step=1.0)
K_out_mM = st.sidebar.slider("[K⁺] outside / mM", 1.0, 500.0, 5.0, step=1.0)

st.sidebar.header("Valinomycin dynamics")

k_val = st.sidebar.slider(
    "Valinomycin coupling rate / s⁻¹",
    0.001, 5.0, 0.2, step=0.001,
    help="How fast Δψ approaches the K⁺ Nernst potential."
)

duration_s = st.sidebar.slider("Simulation time / s", 1.0, 600.0, 120.0, step=1.0)

st.sidebar.header("Leak / ATP synthase-like discharge")

include_leak = st.sidebar.checkbox("Include electrical leak / ATP synthase-like discharge", value=True)

leak_rate = st.sidebar.slider(
    "Leak discharge rate / s⁻¹",
    0.0, 5.0, 0.05, step=0.001,
    help="Simple discharge term. Larger values dissipate Δψ faster."
)

leak_start_s = st.sidebar.slider(
    "Leak starts after / s",
    0.0, float(duration_s), 30.0, step=1.0,
    help="Useful for simulating addition of ATP synthase or an uncoupler after valinomycin."
)

st.sidebar.header("Goldman mode")

use_goldman = st.sidebar.checkbox("Show Goldman-Hodgkin-Katz estimate", value=False)

Na_in_mM = st.sidebar.slider("[Na⁺] inside / mM", 0.1, 200.0, 10.0, step=0.1)
Na_out_mM = st.sidebar.slider("[Na⁺] outside / mM", 0.1, 200.0, 150.0, step=0.1)
Cl_in_mM = st.sidebar.slider("[Cl⁻] inside / mM", 0.1, 200.0, 10.0, step=0.1)
Cl_out_mM = st.sidebar.slider("[Cl⁻] outside / mM", 0.1, 200.0, 150.0, step=0.1)

P_K = st.sidebar.slider("Relative permeability P_K", 0.0, 1000.0, 100.0, step=1.0)
P_Na = st.sidebar.slider("Relative permeability P_Na", 0.0, 100.0, 1.0, step=0.1)
P_Cl = st.sidebar.slider("Relative permeability P_Cl", 0.0, 100.0, 1.0, step=0.1)

# -----------------------------
# Geometry
# -----------------------------
T = temperature_C + 273.15

radius_m = radius_nm * 1e-9
area_m2 = 4 * np.pi * radius_m**2
volume_L = (4/3) * np.pi * radius_m**3 * 1000

capacitance_F_m2 = capacitance_uF_cm2 * 0.01  # 1 µF/cm² = 0.01 F/m²
C_total = capacitance_F_m2 * area_m2

# -----------------------------
# Potentials
# -----------------------------
def nernst_potential_mV(c_out_mM, c_in_mM, z=1):
    """Potential inside relative to outside for ion z."""
    return (R * T / (z * F)) * np.log(c_out_mM / c_in_mM) * 1000


def goldman_potential_mV(K_out, K_in, Na_out, Na_in, Cl_out, Cl_in, Pk, Pna, Pcl):
    """
    GHK voltage equation.
    For cations, outside concentration is in numerator.
    For anions, inside concentration is in numerator.
    Returns inside relative to outside.
    """
    numerator = Pk * K_out + Pna * Na_out + Pcl * Cl_in
    denominator = Pk * K_in + Pna * Na_in + Pcl * Cl_out

    if numerator <= 0 or denominator <= 0:
        return np.nan

    return (R * T / F) * np.log(numerator / denominator) * 1000


E_K_mV = nernst_potential_mV(K_out_mM, K_in_mM, z=1)
E_Na_mV = nernst_potential_mV(Na_out_mM, Na_in_mM, z=1)
E_Cl_mV = nernst_potential_mV(Cl_out_mM, Cl_in_mM, z=-1)
E_GHK_mV = goldman_potential_mV(
    K_out_mM, K_in_mM,
    Na_out_mM, Na_in_mM,
    Cl_out_mM, Cl_in_mM,
    P_K, P_Na, P_Cl
)

# -----------------------------
# Dynamic simulation
# -----------------------------
n_points = max(500, min(5000, int(duration_s * 20)))
time = np.linspace(0, duration_s, n_points)
dt = time[1] - time[0]

psi = np.zeros_like(time)
val_flux_equiv = np.zeros_like(time)
leak_flux_equiv = np.zeros_like(time)

# Didactic voltage model:
# dψ/dt = k_val * (E_K - ψ) - k_leak * ψ
#
# If leak is off, ψ approaches E_K.
# If leak is on, steady state is:
# ψ_ss = k_val/(k_val + k_leak) * E_K
#
# This treats fluxes as voltage-equivalent rates and avoids needing to
# explicitly simulate very small K+ movements.
for i in range(1, n_points):
    current_leak_rate = leak_rate if (include_leak and time[i-1] >= leak_start_s) else 0.0

    val_drive = k_val * (E_K_mV - psi[i-1])
    leak_drive = -current_leak_rate * psi[i-1]

    psi[i] = psi[i-1] + (val_drive + leak_drive) * dt

    val_flux_equiv[i] = val_drive
    leak_flux_equiv[i] = leak_drive

# Estimate how many net K+ ions need to cross to charge the membrane.
# Q = C V; mol = Q/F; ions = mol * NA
K_ions_for_final_psi = C_total * abs(psi[-1] / 1000) / F * NA

# Estimate change in internal K concentration from that charge transfer.
# This is usually tiny compared with bulk K+.
delta_K_mol = K_ions_for_final_psi / NA
delta_K_mM = delta_K_mol / volume_L * 1000

if E_K_mV < 0:
    k_direction = "K⁺ tends to leave the vesicle, making the inside negative."
elif E_K_mV > 0:
    k_direction = "K⁺ tends to enter the vesicle, making the inside positive."
else:
    k_direction = "There is no K⁺ gradient, so valinomycin creates no K⁺ diffusion potential."

# -----------------------------
# Metrics
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("K⁺ Nernst potential", f"{E_K_mV:+.1f} mV")
with col2:
    st.metric("Final simulated Δψ", f"{psi[-1]:+.1f} mV")
with col3:
    st.metric("Vesicle volume", f"{volume_L*1e18:.2f} aL")
with col4:
    st.metric("Total capacitance", f"{C_total:.2e} F")

col5, col6, col7, col8 = st.columns(4)

with col5:
    st.metric("K⁺ ions to charge membrane", f"{K_ions_for_final_psi:.0f}")
with col6:
    st.metric("Bulk [K⁺] change estimate", f"{delta_K_mM:.4f} mM")
with col7:
    if include_leak:
        steady = k_val / (k_val + leak_rate) * E_K_mV if (k_val + leak_rate) > 0 else np.nan
        st.metric("Expected steady Δψ with leak", f"{steady:+.1f} mV")
    else:
        st.metric("Expected steady Δψ", f"{E_K_mV:+.1f} mV")
with col8:
    if use_goldman:
        st.metric("Goldman potential", f"{E_GHK_mV:+.1f} mV")
    else:
        st.metric("Goldman mode", "off")

st.info(k_direction)

# -----------------------------
# Plots
# -----------------------------
left, right = st.columns(2)

with left:
    fig, ax = plt.subplots()
    ax.plot(time, psi, label="simulated Δψ")
    ax.axhline(E_K_mV, linestyle="--", linewidth=0.8, label="K⁺ Nernst potential")
    if include_leak:
        ax.axvline(leak_start_s, linestyle=":", linewidth=0.8, label="leak added")
        steady = k_val / (k_val + leak_rate) * E_K_mV if (k_val + leak_rate) > 0 else np.nan
        ax.axhline(steady, linestyle="-.", linewidth=0.8, label="steady state with leak")
    if use_goldman and np.isfinite(E_GHK_mV):
        ax.axhline(E_GHK_mV, linestyle=":", linewidth=1.2, label="Goldman estimate")
    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Membrane potential Δψ / mV")
    ax.set_title("Valinomycin-driven membrane potential")
    ax.legend()
    st.pyplot(fig)

with right:
    fig, ax = plt.subplots()
    ax.plot(time, val_flux_equiv, label="valinomycin drive")
    ax.plot(time, leak_flux_equiv, label="leak / ATP synthase-like discharge")
    ax.plot(time, val_flux_equiv + leak_flux_equiv, label="net voltage change")
    ax.axhline(0, linewidth=0.8)
    if include_leak:
        ax.axvline(leak_start_s, linestyle=":", linewidth=0.8)
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Voltage change rate / mV s⁻¹")
    ax.set_title("Competing processes")
    ax.legend()
    st.pyplot(fig)

# -----------------------------
# Goldman section
# -----------------------------
if use_goldman:
    st.subheader("Goldman-Hodgkin-Katz estimate")

    st.markdown(f"""
With the selected permeabilities, the Goldman estimate is **{E_GHK_mV:+.1f} mV**.

For a K⁺-selective membrane, the Goldman potential approaches the K⁺ Nernst potential.
As permeability to Na⁺ or Cl⁻ increases, the membrane potential moves away from
the K⁺ equilibrium potential.
""")

    ions = ["K⁺", "Na⁺", "Cl⁻"]
    values = [E_K_mV, E_Na_mV, E_Cl_mV]

    fig, ax = plt.subplots()
    ax.bar(ions, values)
    ax.axhline(0, linewidth=0.8)
    ax.axhline(E_GHK_mV, linestyle="--", linewidth=1.0, label="Goldman estimate")
    ax.set_ylabel("Equilibrium potential / mV")
    ax.set_title("Individual ion Nernst potentials")
    ax.legend()
    st.pyplot(fig)

# -----------------------------
# Teaching notes
# -----------------------------
st.subheader("Teaching interpretation")

st.markdown(f"""
For the selected concentrations, the K⁺ Nernst potential is **{E_K_mV:+.1f} mV**.

If **[K⁺] inside is high** and **[K⁺] outside is low**, valinomycin allows K⁺ to
leave the vesicle. The inside becomes negative until the electrical force pulls
K⁺ back strongly enough to balance the concentration gradient.

A key point is that only a small number of ions are required to charge the membrane.
Here, approximately **{K_ions_for_final_psi:.0f} net K⁺ ions** are enough to create
the final simulated potential. This corresponds to only about
**{delta_K_mM:.4f} mM** change in bulk K⁺ concentration inside the vesicle.
""")

st.subheader("Important simplifications")

st.markdown("""
- The dynamic model treats voltage relaxation phenomenologically rather than solving full electrodiffusion.
- Bulk K⁺ concentrations are held constant, except for the displayed charge estimate.
- The leak / ATP synthase term is represented as electrical discharge, not as a detailed enzyme mechanism.
- Goldman mode gives a static estimate from ion permeabilities, not a dynamic multi-ion simulation.
- The model is meant for teaching intuition.
""")
