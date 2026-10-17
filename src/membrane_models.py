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
    """Proton leak (into the vesicle) that rises smoothly above |Δψ| = threshold.

    The value at Δψ = 0 is subtracted, so the leak is exactly zero without a potential
    and changes smoothly through 0 mV. (Without this, it jumped between −0.9 and +0.9
    H⁺/s at 0 mV, which made the solver stick to Δψ = 0.)
    """
    x = (np.abs(psi_mV) - threshold_mV) / softness_mV
    x0 = -threshold_mV / softness_mV
    return -np.sign(psi_mV) * conductance * softness_mV * (np.logaddexp(0, x) - np.logaddexp(0, x0))


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
# 2) Valinomycin: a K⁺ gradient as a battery, optionally with an anion leak
# ------------------------------------------------------------
def valino_model(outer_diameter_nm=100.0, bilayer_nm=4.5, temperature_C=25.0,
                 K_in_mM=50.0, K_out_mM=5.0, load_direction="dissipate", load_model="potential",
                 load_rate=1500.0, load_start_s=0.0,
                 anion_conductance=0.0, anion_in_mM=100.0, anion_out_mM=100.0,
                 duration_s=60.0):
    """Valinomycin is fast, so Δψ always equals the K⁺ Nernst potential.

    load_direction: 'dissipate', 'in' or 'out'; load_model: 'potential' or 'constant'.
    Every charge that crosses the membrane (the load, or anions through the anion leak)
    is compensated by one K⁺ moving through valinomycin:
        dN_K/dt = −J_load + J_anion      (all fluxes counted INTO the vesicle)
        J_anion = g_A · (Δψ − E_anion)
    'Discharge' stops when Δψ reaches 0. If the internal K⁺ is used up, the clamp
    no longer holds and Δψ is not defined from then on (NaN).
    """
    geo = geometry(outer_diameter_nm, bilayer_nm)
    V = geo["volume_L"]
    RT_F = nernst_factor_mV(temperature_C)

    def count(mM):
        return mM * 1e-3 * V * NA

    def conc_mM(n):
        return np.maximum(n, 1e-30) / NA / V * 1000

    N0, A0, N_eq = count(K_in_mM), count(anion_in_mM), count(K_out_mM)

    def psi_of(N):
        return RT_F * np.log(K_out_mM / conc_mM(N))

    psi0 = float(psi_of(N0))
    s0 = np.sign(psi0)

    def load_in(psi, active):
        """Positive charge moved INTO the vesicle by the load."""
        if not active:
            return 0.0 * psi
        if load_model == "constant":
            J = load_rate + 0.0 * psi
        else:
            J = load_rate * np.abs(psi) / abs(psi0) if abs(psi0) > 1e-12 else 0.0 * psi
        if load_direction == "dissipate":
            return -s0 * J
        return J if load_direction == "in" else -J

    def anion_in(psi, N_A):
        if anion_conductance <= 0:
            return 0.0 * psi
        return anion_conductance * (psi - RT_F * np.log(conc_mM(N_A) / anion_out_mM))

    def make_rhs(active):
        def rhs(t, y):
            N, N_A = y
            psi = psi_of(N)
            J_a = anion_in(psi, N_A)
            return [-load_in(psi, active) + J_a, J_a]
        return rhs

    def ev_used_up(t, y):
        return y[0] - 1e-9 * N0
    ev_used_up.terminal, ev_used_up.direction = True, -1

    def ev_gradient_spent(t, y):
        return y[0] - N_eq
    ev_gradient_spent.terminal = True

    n = max(1000, min(30000, int(duration_s * 20)))
    t = np.linspace(0, duration_s, n)
    N_out, A_out = np.full(n, np.nan), np.full(n, np.nan)
    t_now, y = 0.0, np.array([N0, A0])
    spent = load_direction == "dissipate" and abs(psi0) < 1e-12
    t_spent = 0.0 if spent else np.inf
    t_used_up = None
    filled = 0
    atol = [max(N0, 1.0) * 1e-12, max(A0, 1.0) * 1e-12]

    # Integrate piece by piece: before the load starts, while it runs, and after a
    # 'discharge' load has spent the gradient. Each switch is a sudden change.
    while t_now < duration_s:
        load_on = t_now >= load_start_s
        t_end = duration_s if load_on else min(load_start_s, duration_s)
        active = load_on and not spent and load_rate > 0
        events = [ev_used_up] + ([ev_gradient_spent] if active and load_direction == "dissipate" else [])
        sol = solve_ivp(make_rhs(active), (t_now, t_end), y, method="Radau", events=events,
                        dense_output=True, rtol=1e-9, atol=atol)
        if not sol.success:
            raise RuntimeError(sol.message)
        t_stop = sol.t[-1]
        idx = np.arange(filled, n)[t[filled:] <= t_stop + 1e-12 * max(duration_s, 1)]
        if idx.size:
            N_out[idx], A_out[idx] = sol.sol(np.minimum(t[idx], t_stop))
            filled = idx[-1] + 1
        y, t_now = sol.y[:, -1], t_stop
        if sol.status == 1:
            if sol.t_events[0].size:          # internal K⁺ used up
                t_used_up = t_stop
                break
            spent, t_spent = True, t_stop     # discharge finished: load switches off

    used_up = np.zeros(n, dtype=bool)
    if t_used_up is not None:
        used_up[filled:] = True
        N_out[filled:] = 0.0
        A_out[filled:] = y[1]

    psi = psi_of(N_out)
    active_grid = (t >= load_start_s) & (t < t_spent) & (load_rate > 0)
    J_charge = np.where(active_grid, load_in(psi, True), 0.0)
    J_A = anion_in(psi, A_out)
    J_K = -J_charge + J_A
    if t_used_up is not None:               # clamp no longer valid: Δψ undefined
        J_charge = np.where(used_up, load_in(np.full(n, psi0), True) * (t >= load_start_s), J_charge)
        J_K[used_up], J_A[used_up] = 0.0, 0.0
        psi[used_up] = np.nan
    return dict(time=t, psi_mV=psi, K_in_mM=conc_mM(N_out) * (N_out > 0), K_ions=N_out,
                J_charge=J_charge, J_K=J_K, J_anion=J_A, anion_in_mM=conc_mM(A_out),
                anions_in=A_out - A0, K_ions_start=N0, K_ions_equilibrium=N_eq,
                psi_start=psi0, t_used_up=t_used_up, **geo)


# ------------------------------------------------------------
# 3) Combined: proton pump + valinomycin + internal buffer
# ------------------------------------------------------------
VALINOMYCIN_SPEED = 10000.0  # s⁻¹: Δψ relaxes to E_K within ~0.1 ms, much faster than the pump


def combined_model(outer_diameter_nm=100.0, bilayer_nm=4.5, capacitance_uF_cm2=1.0,
                   temperature_C=25.0, pH_start=7.0, buffer_mM=50.0, buffer_pKa=7.5,
                   K_in_mM=50.0, K_out_mM=50.0,
                   pump_max=1000.0, pump_dG_kJ=15.0, pump_width_mV=10.0,
                   valinomycin=True,
                   leak_conductance=0.0, leak_threshold_mV=40.0, leak_softness_mV=10.0,
                   anion_conductance=0.0, anion_in_mM=100.0, anion_out_mM=100.0,
                   duration_s=60.0, n_points=1500):
    """A pump moves H⁺ into the vesicle; valinomycin lets K⁺ move to compensate the charge.

    Optional leaks, the same as in the proton model: a proton leak (changes Δψ and pH)
    and an anion leak (changes only Δψ).

    States: net H⁺ moved in, K⁺ inside, ΔpH, anions inside, H⁺ moved by the pump (counts).
    Δψ comes from the net charge on the membrane capacitance:
        Δψ = (H⁺ moved in − K⁺ that left − anions that entered) · mV per charge
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
    A_start = anion_in_mM * 1e-3 * V * NA
    g_val = VALINOMYCIN_SPEED / mVpc if valinomycin else 0.0   # K⁺ s⁻¹ mV⁻¹

    def state(Q, N, dpH, N_A):
        psi = (Q - (N_start - N) - (N_A - A_start)) * mVpc
        K_mM = max(N, 0.0) / NA / V * 1000
        E_K = RT_F * np.log(K_out_mM / max(K_mM, 1e-12))
        back = psi - Z * dpH
        J_pump = pump_rate(back, pump_max, stop_mV, pump_width_mV)
        J_leak = soft_leak(psi, leak_conductance, leak_threshold_mV, leak_softness_mV)
        J_K = -g_val * (psi - E_K)          # K⁺ into the vesicle
        if anion_conductance > 0:
            A_mM = max(N_A, 1e-9) / NA / V * 1000
            J_A = anion_conductance * (psi - RT_F * np.log(A_mM / anion_out_mM))
        else:
            A_mM, J_A = N_A / NA / V * 1000, 0.0
        return psi, K_mM, E_K, back, J_pump, J_leak, J_K, J_A, A_mM

    def rhs(t, y):
        Q, N, dpH, N_A, _ = y
        _, _, _, _, J_pump, J_leak, J_K, J_A, _ = state(Q, N, dpH, N_A)
        J_H = J_pump + J_leak
        beta = soluble_beta(pH_start + dpH, buffer_mM / 1000, buffer_pKa)
        return [J_H, J_K, -J_H / NA / (beta * V), J_A, J_pump]

    t = np.linspace(0, duration_s, n_points)
    sol = solve_ivp(rhs, (0, duration_s), [0.0, N_start, 0.0, A_start, 0.0], method="Radau", t_eval=t,
                    rtol=1e-7, atol=[1e-6, max(N_start, 1.0) * 1e-9, 1e-10, max(A_start, 1.0) * 1e-9, 1e-6])
    if not sol.success:
        raise RuntimeError(sol.message)
    Q, N, dpH, N_A, Q_pumped = sol.y
    out = [state(*y[:4]) for y in sol.y.T]
    psi, K_mM, E_K, back, J_pump, J_leak, J_K, J_A, A_mM = (np.array(v) for v in zip(*out))
    return dict(time=t, psi_mV=psi, pH_in=pH_start + dpH, delta_pH=dpH, K_in_mM=K_mM,
                K_ions=N, K_ions_start=N_start, H_pumped=Q_pumped, H_net_in=Q, E_K_mV=E_K,
                back_pressure_mV=back, pH_part_mV=-Z * dpH, J_H=J_pump, J_leak=J_leak, J_K=J_K,
                J_anion=J_A, anion_in_mM=A_mM, anions_in=N_A - A_start,
                pump_fraction=J_pump / pump_max if pump_max > 0 else np.zeros_like(t),
                stop_mV=stop_mV, Z_mV=Z, **geo)
