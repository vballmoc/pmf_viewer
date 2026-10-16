"""Models for the three-tab membrane app (membrane_app.py).

No Streamlit code in here, so the models can be tested on their own.

Common conventions
- Δψ = ψ(inside) − ψ(outside), in mV.
- Fluxes are counted in ions per second, positive INTO the vesicle.
- ΔpH = pH(inside) − pH(start); the outside pH stays at the start value.
- Geometry: the membrane capacitance uses the outer surface area, all
  concentrations use the aqueous lumen (outer diameter minus two bilayers).
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import expit

R = 8.314462618       # J mol⁻¹ K⁻¹
F = 96485.33212       # C mol⁻¹
NA = 6.02214076e23    # mol⁻¹
KW = 1e-14
LN10 = np.log(10)


# ------------------------------------------------------------
# Shared pieces
# ------------------------------------------------------------
def geometry(outer_diameter_nm, bilayer_nm, capacitance_uF_cm2=1.0):
    """Lumen volume (L), outer area (m²), capacitance (F) and mV per transferred charge."""
    r_out = outer_diameter_nm / 2 * 1e-9
    r_in = max(outer_diameter_nm / 2 - bilayer_nm, 0.0005) * 1e-9
    area_m2 = 4 * np.pi * r_out ** 2
    volume_L = 4 / 3 * np.pi * r_in ** 3 * 1000
    C_F = capacitance_uF_cm2 * 0.01 * area_m2
    mV_per_charge = F / NA / C_F * 1000
    return dict(volume_L=volume_L, area_m2=area_m2, C_F=C_F, mV_per_charge=mV_per_charge,
                inner_diameter_nm=2 * r_in * 1e9)


def nernst_factor_mV(temperature_C):
    """RT/F in mV."""
    return R * (temperature_C + 273.15) / F * 1000


def soluble_beta(pH, buffer_M, pKa):
    """Buffer capacity of a soluble buffer plus water, mol L⁻¹ pH⁻¹."""
    H = 10.0 ** (-pH)
    Ka = 10.0 ** (-pKa)
    return LN10 * buffer_M * Ka * H / (Ka + H) ** 2 + LN10 * (H + KW / H)


def pump_rate(back_pressure_mV, rate_max, stop_mV, width_mV):
    """Pump rate as a function of the back-pressure (the H⁺ gradient it pumps against).

    Full speed far from equilibrium, exactly zero when the back-pressure equals the
    stopping potential (the pump's driving energy per charge), and running backwards
    beyond it. tanh(x/2) = 2·expit(x) − 1; width_mV sets how gradual the slowdown is.
    """
    return rate_max * (2 * expit((stop_mV - back_pressure_mV) / width_mV) - 1)


def soft_leak(psi_mV, conductance, threshold_mV, softness_mV):
    """Proton leak (into the vesicle) that rises smoothly above |Δψ| = threshold."""
    x = (np.abs(psi_mV) - threshold_mV) / softness_mV
    return -np.sign(psi_mV) * conductance * softness_mV * np.logaddexp(0, x)


# ------------------------------------------------------------
# 1) Proton pump: Δψ builds up fast, ΔpH slowly
# ------------------------------------------------------------
def proton_model(outer_diameter_nm=200.0, bilayer_nm=4.5, capacitance_uF_cm2=1.0,
                 pH_start=7.0, buffer_mM=50.0, buffer_pKa=7.5,
                 membrane_buffering=True, lipid_area_nm2=0.7, titratable_fraction=0.25,
                 headgroup_pKa=6.5,
                 pump_max=1000.0, pump_dG_kJ=15.0, pump_width_mV=10.0, pump_into=True,
                 pump_off_s=None,
                 leak_conductance=5.0, leak_threshold_mV=40.0, leak_softness_mV=10.0,
                 anion_conductance=0.0, anion_in_mM=100.0, anion_out_mM=100.0,
                 temperature_C=25.0, duration_s=20.0, n_points=1000):
    """Proton pump with proton leak and an optional anion (counter-ion) leak.

    States: net positive charge moved in (count), ΔpH, anions inside (count).
    Protons change both Δψ and pH; anions only change Δψ. Anions move along their
    electrochemical gradient: J_anion,in = g · (Δψ − E_anion), E_anion = RT/F · ln([A⁻]in/[A⁻]out).
    pump_off_s: time at which the pump is switched off (None = runs throughout).
    """
    geo = geometry(outer_diameter_nm, bilayer_nm, capacitance_uF_cm2)
    V, mVpc = geo["volume_L"], geo["mV_per_charge"]
    RT_F = nernst_factor_mV(temperature_C)
    Z = np.log(10) * RT_F                    # mV per pH unit
    stop_mV = pump_dG_kJ * 1000 / F * 1000
    s = 1.0 if pump_into else -1.0
    A_start = anion_in_mM * 1e-3 * V * NA

    # headgroups on the inner leaflet, treated as an extra buffer in the lumen
    H0 = 10.0 ** (-pH_start)
    Kh = 10.0 ** (-headgroup_pKa)
    r_in = geo["inner_diameter_nm"] / 2
    n_sites = 4 * np.pi * r_in ** 2 / lipid_area_nm2 * titratable_fraction
    beta_membrane = LN10 * n_sites * Kh * H0 / (Kh + H0) ** 2 / NA / V if membrane_buffering else 0.0

    def beta(pH):
        return soluble_beta(pH, buffer_mM / 1000, buffer_pKa) + beta_membrane

    def fluxes(q, dpH, N_A, pump_on):
        psi = q * mVpc
        back = s * (psi - Z * dpH)           # electrochemical H⁺ gradient against the pump
        J_pump = s * pump_rate(back, pump_max, stop_mV, pump_width_mV) * pump_on
        J_leak = soft_leak(psi, leak_conductance, leak_threshold_mV, leak_softness_mV)
        A_in_mM = np.maximum(N_A, 1e-9) / NA / V * 1000
        E_A = RT_F * np.log(A_in_mM / anion_out_mM) if anion_conductance > 0 else 0.0
        J_anion = anion_conductance * (psi - E_A)        # anions INTO the vesicle
        return psi, J_pump, J_leak, J_anion, A_in_mM

    def make_rhs(pump_on):
        def rhs(t, y):
            q, dpH, N_A = y
            _, J_pump, J_leak, J_anion, _ = fluxes(q, dpH, N_A, pump_on)
            J_H = J_pump + J_leak
            return [J_H - J_anion, -J_H / NA / (beta(pH_start + dpH) * V), J_anion]
        return rhs

    t = np.linspace(0, duration_s, n_points)
    t_off = duration_s if pump_off_s is None else min(max(pump_off_s, 0.0), duration_s)
    atol = [1e-6, 1e-10, max(A_start, 1.0) * 1e-10]
    y0 = [0.0, 0.0, A_start]
    # Solve the "pump on" and "pump off" periods separately; the switch is a sudden jump.
    # Time points before t_off belong to the first period, the rest to the second.
    pump_runs_to_end = t_off >= duration_s
    periods = [(0.0, t_off, 1.0, t if pump_runs_to_end else t[t < t_off])]
    if not pump_runs_to_end:
        periods.append((t_off, duration_s, 0.0, t[t >= t_off]))
    pieces, on_flags = [], []
    for a, b, on, t_eval in periods:
        if b <= a:                                    # pump off from the start
            continue
        sol = solve_ivp(make_rhs(on), (a, b), y0, method="Radau", t_eval=t_eval,
                        rtol=1e-7, atol=atol, dense_output=True)
        if not sol.success:
            raise RuntimeError(sol.message)
        pieces.append(sol.y)
        on_flags.append(np.full(sol.y.shape[1], on))
        y0 = sol.sol(b)
    q, dpH, N_A = np.hstack(pieces)
    pump_on = np.concatenate(on_flags)
    psi, J_pump, J_leak, J_anion, A_in_mM = fluxes(q, dpH, N_A, pump_on)
    return dict(time=t, psi_mV=psi, delta_pH=dpH, pH_in=pH_start + dpH,
                pmf_mV=psi - Z * dpH, J_pump=J_pump, J_leak=J_leak, J_anion=J_anion,
                anion_in_mM=A_in_mM, anions_in=N_A - A_start, t_pump_off=pump_off_s,
                stop_mV=stop_mV, beta_start=beta(pH_start), beta_membrane=beta_membrane,
                Z_mV=Z, **geo)


# ------------------------------------------------------------
# 2) Valinomycin: a K⁺ gradient as a battery (same algorithm as valino1_app.py)
# ------------------------------------------------------------
def valino_model(outer_diameter_nm=100.0, bilayer_nm=4.5, temperature_C=25.0,
                 K_in_mM=50.0, K_out_mM=5.0, load_direction="dissipate", load_model="potential",
                 load_rate=1500.0, load_start_s=0.0, duration_s=60.0):
    """load_direction: 'dissipate', 'in' or 'out'; load_model: 'potential' or 'constant'."""
    geo = geometry(outer_diameter_nm, bilayer_nm)
    V = geo["volume_L"]
    RT_F = nernst_factor_mV(temperature_C)

    def nernst(K_mM):
        return RT_F * np.log(max(K_out_mM, 1e-30) / max(K_mM, 1e-30))

    N = K_in_mM * 1e-3 * V * NA
    N_start = N
    N_eq = K_out_mM * 1e-3 * V * NA
    psi_start = nernst(K_in_mM)

    n = max(1000, min(30000, int(duration_s * 20)))
    t = np.linspace(0, duration_s, n)
    dt = t[1] - t[0]
    K_mM, psi, N_K = np.zeros(n), np.zeros(n), np.zeros(n)
    J_charge, J_K = np.zeros(n), np.zeros(n)

    for i in range(n):
        K_mM[i] = N / NA / V * 1000
        psi[i] = nernst(K_mM[i])
        N_K[i] = N
        if i == n - 1:
            break
        if t[i] < load_start_s:
            J = 0.0
        elif load_model == "constant":
            J = load_rate
        else:
            J = load_rate * abs(psi[i]) / abs(psi_start) if abs(psi_start) > 1e-12 else 0.0
        if load_direction == "dissipate":
            Jc = -np.sign(psi[i]) * J
        elif load_direction == "in":
            Jc = J
        else:
            Jc = -J
        Jk = -Jc
        if Jk < 0:
            Jk = max(Jk, -N / dt)
        if load_direction == "dissipate":
            if psi_start < 0:
                Jk = max(Jk, -max((N - N_eq) / dt, 0.0))
            elif psi_start > 0:
                Jk = min(Jk, max((N_eq - N) / dt, 0.0))
            else:
                Jk = 0.0
        N = max(N + Jk * dt, 1e-30)
        J_charge[i], J_K[i] = Jc, Jk
    J_charge[-1], J_K[-1] = J_charge[-2], J_K[-2]

    used_up = N_K <= 1e-9 * N_start
    t_used_up = t[np.argmax(used_up)] if used_up.any() else None
    psi[used_up] = np.nan
    return dict(time=t, psi_mV=psi, K_in_mM=K_mM, K_ions=N_K, J_charge=J_charge, J_K=J_K,
                K_ions_start=N_start, K_ions_equilibrium=N_eq, psi_start=psi_start,
                t_used_up=t_used_up, **geo)


# ------------------------------------------------------------
# 3) Combined: proton pump + valinomycin + internal buffer
# ------------------------------------------------------------
VALINOMYCIN_SPEED = 10000.0  # s⁻¹: Δψ relaxes to E_K within ~0.1 ms, much faster than the pump


def combined_model(outer_diameter_nm=100.0, bilayer_nm=4.5, capacitance_uF_cm2=1.0,
                   temperature_C=25.0, pH_start=7.0, buffer_mM=50.0, buffer_pKa=7.5,
                   K_in_mM=50.0, K_out_mM=50.0,
                   pump_max=1000.0, pump_dG_kJ=15.0, pump_width_mV=10.0,
                   valinomycin=True, duration_s=60.0, n_points=1500):
    """A pump moves H⁺ into the vesicle; valinomycin lets K⁺ move to compensate the charge.

    States: H⁺ pumped in (count), K⁺ inside (count), ΔpH.
    Δψ comes from the net charge on the membrane capacitance:
        Δψ = (H⁺ pumped in − K⁺ that left) · mV per charge
    Valinomycin moves K⁺ along its electrochemical gradient, J_K ∝ −(Δψ − E_K),
    fast enough that Δψ stays at E_K while K⁺ is available. When the internal K⁺
    runs out, E_K rises steeply and K⁺ export stops by itself, so from then on the
    pumped protons charge the membrane, as without valinomycin.
    """
    geo = geometry(outer_diameter_nm, bilayer_nm, capacitance_uF_cm2)
    V, mVpc = geo["volume_L"], geo["mV_per_charge"]
    RT_F = nernst_factor_mV(temperature_C)
    Z = np.log(10) * RT_F
    stop_mV = pump_dG_kJ * 1000 / F * 1000
    N_start = K_in_mM * 1e-3 * V * NA
    g_val = VALINOMYCIN_SPEED / mVpc if valinomycin else 0.0   # K⁺ s⁻¹ mV⁻¹

    def state(Q, N, dpH):
        psi = (Q - (N_start - N)) * mVpc
        K_mM = max(N, 0.0) / NA / V * 1000
        E_K = RT_F * np.log(K_out_mM / max(K_mM, 1e-12))
        back = psi - Z * dpH
        J_H = pump_rate(back, pump_max, stop_mV, pump_width_mV)
        J_K = -g_val * (psi - E_K)          # K⁺ into the vesicle
        return psi, K_mM, E_K, back, J_H, J_K

    def rhs(t, y):
        Q, N, dpH = y
        _, _, _, _, J_H, J_K = state(Q, N, dpH)
        return [J_H, J_K, -J_H / NA / (soluble_beta(pH_start + dpH, buffer_mM / 1000, buffer_pKa) * V)]

    t = np.linspace(0, duration_s, n_points)
    sol = solve_ivp(rhs, (0, duration_s), [0.0, N_start, 0.0], method="Radau", t_eval=t,
                    rtol=1e-7, atol=[1e-6, max(N_start, 1.0) * 1e-9, 1e-10])
    if not sol.success:
        raise RuntimeError(sol.message)
    Q, N, dpH = sol.y
    out = [state(*y) for y in sol.y.T]
    psi, K_mM, E_K, back, J_H, J_K = (np.array(v) for v in zip(*out))
    return dict(time=t, psi_mV=psi, pH_in=pH_start + dpH, delta_pH=dpH, K_in_mM=K_mM,
                K_ions=N, K_ions_start=N_start, H_pumped=Q, E_K_mV=E_K,
                back_pressure_mV=back, pH_part_mV=-Z * dpH, J_H=J_H, J_K=J_K,
                pump_fraction=J_H / pump_max if pump_max > 0 else np.zeros_like(t),
                stop_mV=stop_mV, Z_mV=Z, **geo)
