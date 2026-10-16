"""Δψ, ΔpH and K⁺ across a vesicle membrane: three teaching models in one app.

Tab 1  Proton pump: why Δψ builds up much faster than ΔpH
Tab 2  Valinomycin: a K⁺ gradient as a battery
Tab 3  Pump + valinomycin: how long K⁺ export keeps Δψ low while a pump runs

The models are in membrane_models.py, drawing helpers in membrane_figures.py.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import membrane_models as m
from membrane_figures import (ALERT, CSS, GRID, H_COLOR, H_LIGHT, INK, K_COLOR, K_LIGHT, MUTED,
                              PH_COLOR, csv_with_header, finish, fmt_count, fmt_mV, fmt_signed,
                              fmt_time, label_line, new_axes, vesicle_pair)
from version_info import show_version

st.set_page_config(page_title="Membrane potential lab", layout="wide")
show_version(__file__)  # version line in the sidebar (see version_info.py)
st.markdown(CSS, unsafe_allow_html=True)


def group(title):
    st.markdown(f"<p class='group'>{title}</p>", unsafe_allow_html=True)


def sentence(html):
    st.markdown(f"<p class='result'>{html}</p>", unsafe_allow_html=True)


def question(text):
    st.markdown(f"<p class='question'>{text}</p>", unsafe_allow_html=True)


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


@st.cache_data(max_entries=50)
def run_proton(**kw):
    return m.proton_model(**kw)


@st.cache_data(max_entries=50)
def run_valino(**kw):
    return m.valino_model(**kw)


@st.cache_data(max_entries=50)
def run_combined(**kw):
    return m.combined_model(**kw)


st.title("Δψ, ΔpH and K⁺ across a vesicle membrane")
st.markdown("<p class='lede'>Three teaching models for proteoliposome experiments. "
            "Each tab has its own settings in the column on the left.</p>", unsafe_allow_html=True)

tab_proton, tab_valino, tab_combined = st.tabs(
    ["Proton pump", "Valinomycin and K⁺", "Pump with valinomycin"])


# ==================================================================
# Tab 1: proton pump
# ==================================================================
with tab_proton:
    ctrl, main = st.columns([1, 2.8], gap="large")
    with ctrl:
        group("Vesicle")
        p_d = st.number_input("Outer diameter / nm", 20.0, 5000.0, 200.0, 10.0, key="p_d",
                              help="Typical liposomes are 50–200 nm.")
        p_C = st.number_input("Membrane capacitance / µF cm⁻²", 0.2, 2.0, 1.0, 0.1, key="p_C",
                              help="About 1 µF cm⁻² for a lipid bilayer. Higher capacitance "
                                   "means more charge is needed for the same Δψ.")
        group("Inside the vesicle")
        p_pH = st.number_input("pH at the start", 4.0, 10.0, 7.0, 0.1, key="p_pH")
        p_buf = st.number_input("Buffer / mM", 0.0, 500.0, 50.0, 5.0, key="p_buf")
        p_pKa = st.number_input("Buffer pKa", 3.0, 11.0, 7.5, 0.1, key="p_pKa",
                                help="A buffer works best within about one unit of its pKa.")
        p_mem = st.checkbox("Lipid headgroups also buffer", value=True, key="p_mem",
                            help="Titratable headgroups on the inner leaflet bind or release H⁺.")
        with st.expander("Headgroup details"):
            p_area = st.number_input("Area per lipid / nm²", 0.4, 1.0, 0.7, 0.05, key="p_area")
            p_frac = st.number_input("Titratable fraction of inner lipids", 0.0, 1.0, 0.25, 0.05, key="p_frac")
            p_hpKa = st.number_input("Headgroup pKa", 3.0, 9.0, 6.5, 0.1, key="p_hpKa")
        group("Proton pump")
        p_into = st.radio("Direction", ["H⁺ into the vesicle", "H⁺ out of the vesicle"],
                          key="p_dir") == "H⁺ into the vesicle"
        p_rate = st.number_input("Maximum rate / H⁺ per second", 1.0, 100000.0, 1000.0, 100.0, key="p_rate",
                                 help="Total for all pumps in one vesicle.")
        p_dG = st.number_input("Driving energy / kJ per mol H⁺", 2.0, 50.0, 15.0, 0.5, key="p_dG",
                               help="The pump stops when the H⁺ gradient it pumps against "
                                    "(Δψ plus the ΔpH part) equals this energy.")
        p_w = st.number_input("Slowdown width / mV", 1.0, 50.0, 10.0, 1.0, key="p_w",
                              help="How gradually the pump slows down before it stops.")
        group("Proton leak")
        p_g = st.number_input("Leak above threshold / H⁺ s⁻¹ mV⁻¹", 0.0, 1000.0, 5.0, 0.5, key="p_g",
                              help="Set to 0 for a perfectly tight membrane.")
        with st.expander("Leak details"):
            p_thr = st.number_input("Leak starts at |Δψ| / mV", 0.0, 300.0, 40.0, 5.0, key="p_thr")
            p_soft = st.number_input("Softness of the onset / mV", 1.0, 50.0, 10.0, 1.0, key="p_soft")
        group("Time")
        p_T = st.number_input("Simulated time / s", 0.1, 10000.0, 20.0, 5.0, key="p_T")

    r = run_proton(outer_diameter_nm=p_d, capacitance_uF_cm2=p_C, pH_start=p_pH, buffer_mM=p_buf,
                   buffer_pKa=p_pKa, membrane_buffering=p_mem, lipid_area_nm2=p_area,
                   titratable_fraction=p_frac, headgroup_pKa=p_hpKa, pump_max=p_rate, pump_dG_kJ=p_dG,
                   pump_width_mV=p_w, pump_into=p_into, leak_conductance=p_g, leak_threshold_mV=p_thr,
                   leak_softness_mV=p_soft, duration_s=p_T)
    t = r["time"]
    psi_end, dpH_end = r["psi_mV"][-1], r["delta_pH"][-1]
    i90 = int(np.argmax(np.abs(r["psi_mV"]) >= 0.9 * abs(psi_end))) if abs(psi_end) > 1 else 0
    charges_end = abs(psi_end) / r["mV_per_charge"]
    H_per_pH = r["beta_start"] * r["volume_L"] * m.NA       # H⁺ needed to change the pH by 1 unit

    with main:
        question("Why does a proton pump build up Δψ, but hardly any ΔpH?")
        left, right = st.columns([1.2, 1], gap="large")
        with right:
            sentence(f"Δψ reaches <b>{fmt_mV(psi_end)}</b> (90 % of it within {fmt_time(t[i90])}).")
            pH_mV = abs(r["Z_mV"] * dpH_end)
            if pH_mV < 0.2 * abs(psi_end):
                sentence(f"The <span class='ph'>pH inside</span> changes by only <b>{fmt_signed(dpH_end, 3)}</b> "
                         f"units, worth {pH_mV:.1f} mV of proton-motive force.")
                sentence(f"The reason: moving about <b>{fmt_count(charges_end)}</b> <span class='h'>H⁺</span> "
                         f"across the membrane gives {fmt_mV(psi_end)}, but it takes "
                         f"{fmt_count(H_per_pH)} H⁺ to change the pH inside by one unit. "
                         "The membrane capacitance is small, the buffer capacity is large.")
            else:
                sentence(f"Here the <span class='ph'>pH inside</span> changes a lot: <b>{fmt_signed(dpH_end, 2)}</b> "
                         f"units, worth {pH_mV:.0f} mV of proton-motive force.")
                sentence(f"The reason: the lumen is hardly buffered. It takes only {fmt_count(H_per_pH)} "
                         f"<span class='h'>H⁺</span> to change its pH by one unit, so the pH changes "
                         "before the membrane is charged.")
        with left:
            i_end = len(t) - 1
            show(vesicle_pair(
                dict(title="Start", lines=[(f"pH {p_pH:.2f}", PH_COLOR), ("Δψ 0 mV", INK)], psi_mV=0.0,
                     arrows=[(r["J_pump"][0], H_COLOR, "H⁺ pump")], fill=H_LIGHT, fill_level=0.3),
                dict(title=f"After {fmt_time(t[-1])}",
                     lines=[(f"pH {r['pH_in'][i_end]:.2f}", PH_COLOR), (f"Δψ {fmt_mV(psi_end)}", INK)],
                     psi_mV=psi_end,
                     arrows=[(r["J_pump"][i_end], H_COLOR, "H⁺ pump"), (r["J_leak"][i_end], MUTED, "H⁺ leak")],
                     fill=H_LIGHT, fill_level=0.3),
                f"outside: pH {p_pH:.2f}, constant"))

        sign = 1 if p_into else -1
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = new_axes("Δψ / mV", t[-1])
            ax.axhline(0, color=MUTED, linewidth=0.8)
            ax.axhline(sign * r["stop_mV"], color=H_COLOR, linestyle="--", linewidth=1)
            label_line(ax, sign * r["stop_mV"], "pump stops", H_COLOR, va="bottom" if p_into else "top")
            if p_g > 0:
                ax.axhline(sign * p_thr, color=MUTED, linestyle=":", linewidth=1)
                label_line(ax, sign * p_thr, "leak starts", MUTED, va="top" if p_into else "bottom")
            ax.plot(t, r["psi_mV"], color=INK, linewidth=2.2)
            show(finish(fig, ax))
            st.caption("Membrane potential. It approaches its final value within a fraction of a second.")
        with c2:
            fig, ax = new_axes("ΔpH = pH(inside) − pH(start)", t[-1])
            ax.axhline(0, color=MUTED, linewidth=0.8)
            ax.plot(t, r["delta_pH"], color=PH_COLOR, linewidth=2.2)
            show(finish(fig, ax))
            st.caption("Change of the internal pH. Note the much smaller scale.")
        c3, c4 = st.columns(2)
        with c3:
            fig, ax = new_axes("mV", t[-1])
            ax.axhline(0, color=MUTED, linewidth=0.8)
            ax.plot(t, r["psi_mV"] - r["Z_mV"] * r["delta_pH"], color=INK, linewidth=2.2,
                    label="total H⁺ gradient")
            ax.plot(t, r["psi_mV"], color=INK, linewidth=1.2, linestyle="--", label="from Δψ")
            ax.plot(t, -r["Z_mV"] * r["delta_pH"], color=PH_COLOR, linewidth=1.8, label="from ΔpH")
            show(finish(fig, ax, legend=True))
            st.caption("Proton-motive force: Δψ − 59 mV·ΔpH at 25 °C. Almost all of it is Δψ.")
        with c4:
            fig, ax = new_axes("Flow into the vesicle / H⁺ s⁻¹", t[-1])
            ax.axhline(0, color=MUTED, linewidth=0.8)
            ax.plot(t, r["J_pump"], color=H_COLOR, linewidth=2.2, label="pump")
            ax.plot(t, r["J_leak"], color=MUTED, linewidth=1.8, label="leak")
            ax.plot(t, r["J_pump"] + r["J_leak"], color=INK, linewidth=1.2, linestyle="--", label="net")
            show(finish(fig, ax, legend=True))
            st.caption("The pump slows down as the gradient grows; the leak sets in above its threshold.")

        df = pd.DataFrame({"time_s": t, "delta_psi_mV": r["psi_mV"], "pH_inside": r["pH_in"],
                           "delta_pH": r["delta_pH"], "pmf_mV": r["pmf_mV"],
                           "pump_H_per_s": r["J_pump"], "leak_H_per_s": r["J_leak"]})
        params = dict(outer_diameter_nm=p_d, capacitance_uF_per_cm2=p_C, pH_start=p_pH, buffer_mM=p_buf,
                      buffer_pKa=p_pKa, headgroup_buffering=p_mem, area_per_lipid_nm2=p_area,
                      titratable_fraction=p_frac, headgroup_pKa=p_hpKa,
                      pump_direction="into vesicle" if p_into else "out of vesicle", pump_max_H_per_s=p_rate,
                      pump_driving_energy_kJ_per_mol=p_dG, pump_slowdown_width_mV=p_w,
                      leak_H_per_s_per_mV=p_g, leak_threshold_mV=p_thr, leak_softness_mV=p_soft,
                      inner_volume_L=f"{r['volume_L']:.6e}", capacitance_F=f"{r['C_F']:.6e}",
                      pump_stopping_potential_mV=f"{r['stop_mV']:.2f}")
        st.download_button("Download data as CSV", csv_with_header("Proton pump simulation", params, df),
                           file_name=f"proton_pump_d{p_d:g}nm_pH{p_pH:g}.csv", mime="text/csv",
                           key="p_csv", type="primary")

        with st.expander("How the model works"):
            st.markdown(
                "Every proton the pump moves adds one elementary charge to the membrane capacitor "
                "and one H⁺ to the lumen, where most of it is bound by the buffer:")
            st.latex(r"\Delta\psi = \frac{q}{C} \qquad "
                     r"\frac{d\,\mathrm{pH_{in}}}{dt} = -\frac{J_\mathrm{H^+}}{N_A\,\beta\,V}")
            st.markdown(
                "β is the buffer capacity (buffer + water, plus headgroups if switched on), "
                "recalculated at the current pH. The pump runs at full speed far from equilibrium "
                "and stops when the H⁺ gradient it pumps against equals its driving energy per "
                "charge, ΔG/F (for 15 kJ/mol: 155 mV):")
            st.latex(r"J_\mathrm{pump} = J_\mathrm{max}\,\tanh\!\left("
                     r"\frac{\Delta G/F - (\Delta\psi - 2.3\,\tfrac{RT}{F}\,\Delta\mathrm{pH})}{2w}\right)")
            st.markdown(
                "Simplifications: counter-ion movement is not modelled, the outside pH stays "
                "constant, the leak is phenomenological, and the headgroups are treated as a "
                "buffer dissolved in the lumen.")


# ==================================================================
# Tab 2: valinomycin and K⁺
# ==================================================================
LOAD_DIRECTIONS = {"Discharge the gradient": "dissipate",
                   "Push positive charge into the vesicle": "in",
                   "Push positive charge out of the vesicle": "out"}
LOAD_MODELS = {"Slows down as Δψ falls": "potential", "Constant rate": "constant"}

with tab_valino:
    ctrl, main = st.columns([1, 2.8], gap="large")
    with ctrl:
        group("Vesicle")
        v_d = st.number_input("Outer diameter / nm", 5.0, 5000.0, 100.0, 10.0, key="v_d")
        v_bil = st.number_input("Bilayer thickness / nm", 0.0, 20.0, 4.5, 0.1, key="v_bil",
                                help="The lumen is the outer diameter minus two bilayers.")
        v_temp = st.number_input("Temperature / °C", 0.0, 80.0, 25.0, 1.0, key="v_temp")
        group("K⁺ gradient")
        v_Kin = st.number_input("[K⁺] inside at the start / mM", 0.001, 2000.0, 50.0, 5.0, key="v_Kin",
                                help="This internal K⁺ pool is the battery. It changes during the run.")
        v_Kout = st.number_input("[K⁺] outside / mM", 0.001, 2000.0, 5.0, 1.0, key="v_Kout",
                                 help="The outside is a large reservoir, so this stays constant.")
        group("Load")
        v_dir = st.radio("What the load does", list(LOAD_DIRECTIONS), key="v_dir",
                         help="'Discharge' moves positive charge in the direction that weakens Δψ. "
                              "The other two push charge in one fixed direction, whatever Δψ is.")
        v_model = st.radio("How fast", list(LOAD_MODELS), key="v_model",
                           help="'Slows down as Δψ falls': rate proportional to |Δψ|, like a leak.")
        v_rate = st.number_input("Load at the start / charges per second", 0.0, 1_000_000.0, 1500.0, 100.0,
                                 key="v_rate")
        v_t0 = st.number_input("Load switched on after / s", 0.0, 100000.0, 0.0, 1.0, key="v_t0")
        group("Time and threshold")
        v_T = st.number_input("Simulated time / s", 0.1, 100000.0, 60.0, 10.0, key="v_T")
        v_thr = st.number_input("Useful Δψ at least / mV", 0.0, 500.0, 50.0, 5.0, key="v_thr",
                                help="Below this value, Δψ counts as too weak to be useful.")

    r = run_valino(outer_diameter_nm=v_d, bilayer_nm=v_bil, temperature_C=v_temp, K_in_mM=v_Kin,
                   K_out_mM=v_Kout, load_direction=LOAD_DIRECTIONS[v_dir], load_model=LOAD_MODELS[v_model],
                   load_rate=v_rate, load_start_s=v_t0, duration_s=v_T)
    t, psi = r["time"], r["psi_mV"]
    finite = np.isfinite(psi)
    useful = np.zeros_like(t, dtype=bool)
    useful[finite] = np.abs(psi[finite]) >= v_thr
    useful_after = useful & (t >= v_t0)
    lifetime = t[np.flatnonzero(useful_after)[-1]] - v_t0 if useful_after.any() else None
    i_load = int(np.argmax(t >= v_t0))
    k_max = max(r["K_in_mM"].max(), v_Kout)

    with main:
        question("How long can a K⁺ gradient hold a membrane potential?")
        left, right = st.columns([1.2, 1], gap="large")
        with right:
            sentence(f"With valinomycin, Δψ equals the K⁺ Nernst potential. It starts at "
                     f"<b>{fmt_mV(r['psi_start'])}</b>, set by <span class='k'>{v_Kin:.3g} mM K⁺ inside</span> "
                     f"and <span class='k'>{v_Kout:.3g} mM outside</span>.")
            if lifetime is None:
                sentence(f"Δψ is never at least <b>{v_thr:.0f} mV</b> while the load runs, so this "
                         "gradient is too weak to be useful from the beginning.")
            elif useful[-1]:
                sentence(f"Δψ is still at least {v_thr:.0f} mV at the end: it lasts <b>longer than "
                         f"{fmt_time(t[-1] - v_t0)}</b>. Increase the simulated time to see when it drops.")
            else:
                sentence(f"Δψ stays at least {v_thr:.0f} mV for <b>{fmt_time(lifetime)}</b> after the "
                         "load is switched on.")
            sentence(f"The vesicle contains <b>{fmt_count(r['K_ions_start'])}</b> "
                     f"<span class='k'>K⁺ ions</span>; "
                     f"{fmt_count(abs(r['K_ions_start'] - r['K_ions_equilibrium']))} of them must move "
                     "until the inside matches the outside.")
        with left:
            psi_end = psi[-1]
            show(vesicle_pair(
                dict(title="Start", lines=[(f"[K⁺] {v_Kin:.3g} mM", K_COLOR), (f"Δψ {fmt_mV(r['psi_start'])}", INK)],
                     psi_mV=r["psi_start"],
                     arrows=[(r["J_K"][i_load], K_COLOR, "K⁺"), (r["J_charge"][i_load], H_COLOR, "+ charge")],
                     fill=K_LIGHT, fill_level=v_Kin / k_max),
                dict(title=f"After {fmt_time(t[-1])}",
                     lines=[(f"[K⁺] {r['K_in_mM'][-1]:.3g} mM" if np.isfinite(psi_end) else "[K⁺] ≈ 0 mM", K_COLOR),
                            (f"Δψ {fmt_mV(psi_end)}" if np.isfinite(psi_end) else "K⁺ used up",
                             INK if np.isfinite(psi_end) else ALERT)],
                     psi_mV=psi_end,
                     arrows=[(r["J_K"][-2], K_COLOR, "K⁺"), (r["J_charge"][-2], H_COLOR, "+ charge")],
                     fill=K_LIGHT, fill_level=r["K_in_mM"][-1] / k_max),
                f"outside: [K⁺] {v_Kout:.3g} mM, constant", footer_color=K_COLOR))
        if r["t_used_up"] is not None:
            st.warning(f"The K⁺ inside is used up after {fmt_time(r['t_used_up'])}. From then on, "
                       "valinomycin cannot compensate the load, so Δψ is no longer the K⁺ Nernst "
                       "potential and is not shown. The 'Pump with valinomycin' tab models what happens next.")

        def mark_load(ax):
            if v_t0 > 0:
                ax.axvline(v_t0, color=H_COLOR, linestyle="--", linewidth=1)
                ax.text(v_t0, 1.0, " load on", transform=ax.get_xaxis_transform(), color=H_COLOR,
                        fontsize=9, va="top")

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = new_axes("Δψ / mV", t[-1])
            ax.axhspan(-v_thr, v_thr, color=GRID, alpha=0.6, linewidth=0)
            label_line(ax, v_thr, f"too weak (< {v_thr:.0f} mV)", MUTED)
            ax.axhline(0, color=MUTED, linewidth=0.8)
            ax.plot(t, psi, color=INK, linewidth=2.2)
            if r["t_used_up"] is not None:
                ax.axvline(r["t_used_up"], color=ALERT, linestyle=":", linewidth=1.2)
                ax.text(r["t_used_up"], 0.5, " K⁺ used up", transform=ax.get_xaxis_transform(),
                        color=ALERT, fontsize=9, va="center")
            mark_load(ax)
            show(finish(fig, ax))
            st.caption("Δψ = ψ(inside) − ψ(outside), always equal to the K⁺ Nernst potential.")
        with c2:
            fig, ax = new_axes("[K⁺] / mM", t[-1])
            ax.plot(t, r["K_in_mM"], color=K_COLOR, linewidth=2.2, label="inside")
            ax.axhline(v_Kout, color=K_COLOR, linestyle="--", linewidth=1.2, label="outside (constant)")
            ax.set_ylim(bottom=0)
            mark_load(ax)
            show(finish(fig, ax, legend=True))
            st.caption("Δψ is zero when the inside concentration reaches the outside one.")
        c3, _ = st.columns(2)
        with c3:
            fig, ax = new_axes("Flow into the vesicle / ions s⁻¹", t[-1])
            ax.axhline(0, color=MUTED, linewidth=0.8)
            ax.plot(t, r["J_charge"], color=H_COLOR, linewidth=2.2, label="positive charge moved by the load")
            ax.plot(t, r["J_K"], color=K_COLOR, linewidth=2.2, label="K⁺ moved by valinomycin")
            mark_load(ax)
            show(finish(fig, ax, legend=True))
            st.caption("Every charge the load moves is compensated by one K⁺ moving the other way.")

        df = pd.DataFrame({"time_s": t, "delta_psi_mV": psi, "K_inside_mM": r["K_in_mM"],
                           "K_inside_ions": r["K_ions"], "charge_flow_into_vesicle_per_s": r["J_charge"],
                           "K_flow_into_vesicle_per_s": r["J_K"]})
        params = dict(outer_diameter_nm=v_d, bilayer_thickness_nm=v_bil, temperature_C=v_temp,
                      K_inside_start_mM=v_Kin, K_outside_mM=v_Kout, load=f"{v_dir}; {v_model}",
                      load_at_start_charges_per_s=v_rate, load_switched_on_s=v_t0, useful_threshold_mV=v_thr,
                      inner_volume_L=f"{r['volume_L']:.6e}",
                      note="delta_psi_mV is empty where the internal K+ is used up")
        st.download_button("Download data as CSV",
                           csv_with_header("Valinomycin K+ Nernst clamp simulation", params, df),
                           file_name=f"valinomycin_Kin{v_Kin:g}_Kout{v_Kout:g}_d{v_d:g}nm.csv",
                           mime="text/csv", key="v_csv", type="primary")

        with st.expander("How the model works"):
            st.markdown("Valinomycin is assumed to be much faster than the load, so at every moment:")
            st.latex(r"\Delta\psi = \frac{RT}{F}\,\ln\frac{[\mathrm{K^+}]_{out}}{[\mathrm{K^+}]_{in}}")
            st.markdown("Each charge the load moves is compensated by one K⁺ in the opposite direction:")
            st.latex(r"J_{\mathrm{K^+}} = -\,J_{\mathrm{charge}}\qquad "
                     r"\frac{dN_{\mathrm{K^+,in}}}{dt} = J_{\mathrm{K^+}}")
            st.markdown(
                "'Discharge' stops when [K⁺]inside = [K⁺]outside. The outside is a large reservoir "
                "with constant [K⁺]. Osmotic effects, counter-ions and volume changes are not included.")


# ==================================================================
# Tab 3: pump with valinomycin
# ==================================================================
with tab_combined:
    ctrl, main = st.columns([1, 2.8], gap="large")
    with ctrl:
        group("Vesicle")
        c_d = st.number_input("Outer diameter / nm", 20.0, 5000.0, 100.0, 10.0, key="c_d")
        group("Inside the vesicle")
        c_pH = st.number_input("pH at the start", 4.0, 10.0, 7.0, 0.1, key="c_pH")
        c_buf = st.number_input("Buffer / mM", 0.0, 2000.0, 50.0, 5.0, key="c_buf")
        c_pKa = st.number_input("Buffer pKa", 3.0, 11.0, 7.5, 0.1, key="c_pKa")
        group("K⁺")
        c_Kin = st.number_input("[K⁺] inside at the start / mM", 0.001, 2000.0, 50.0, 5.0, key="c_Kin")
        c_Kout = st.number_input("[K⁺] outside / mM", 0.001, 2000.0, 50.0, 5.0, key="c_Kout",
                                 help="Equal to the inside: no K⁺ gradient and Δψ = 0 at the start.")
        group("Proton pump (H⁺ into the vesicle)")
        c_rate = st.number_input("Maximum rate / H⁺ per second", 1.0, 100000.0, 1000.0, 100.0, key="c_rate")
        c_dG = st.number_input("Driving energy / kJ per mol H⁺", 2.0, 50.0, 15.0, 0.5, key="c_dG",
                               help="The pump stops when Δψ plus the ΔpH part reach this energy per charge.")
        c_w = st.number_input("Slowdown width / mV", 1.0, 50.0, 10.0, 1.0, key="c_w")
        group("Comparison and time")
        c_thr = st.number_input("Δψ counts as low below / mV", 1.0, 200.0, 20.0, 5.0, key="c_thr")
        c_T = st.number_input("Simulated time / s", 0.1, 10000.0, 10.0, 5.0, key="c_T")

    kw = dict(outer_diameter_nm=c_d, pH_start=c_pH, buffer_mM=c_buf, buffer_pKa=c_pKa, K_in_mM=c_Kin,
              K_out_mM=c_Kout, pump_max=c_rate, pump_dG_kJ=c_dG, pump_width_mV=c_w, duration_s=c_T)
    r = run_combined(**kw, valinomycin=True)
    r0 = run_combined(**kw, valinomycin=False)
    t = r["time"]

    def first_time(mask):
        idx = np.flatnonzero(mask)
        return t[idx[0]] if idx.size else None

    t_high = first_time(r["psi_mV"] >= c_thr)
    t_high0 = first_time(r0["psi_mV"] >= c_thr)
    running = np.flatnonzero(r["pump_fraction"] >= 0.01)
    t_stop = t[running[-1] + 1] if running.size and running[-1] + 1 < len(t) else None
    K_left = r["K_ions"][-1] / r["K_ions_start"] if r["K_ions_start"] > 0 else 0.0
    exported = r["K_ions_start"] - r["K_ions"][-1]
    i_end = len(t) - 1

    with main:
        question("How long can valinomycin keep Δψ low while a proton pump runs?")
        left, right = st.columns([1.2, 1], gap="large")
        with right:
            sentence("Without valinomycin, Δψ passes "
                     f"{c_thr:.0f} mV after <b>{fmt_time(t_high0) if t_high0 is not None else 'never'}</b> "
                     f"and ends at {fmt_mV(r0['psi_mV'][-1])}.")
            if t_high is None:
                sentence(f"With valinomycin, Δψ stays below {c_thr:.0f} mV for the <b>whole run</b> "
                         f"(it ends at {fmt_mV(r['psi_mV'][-1])}), because "
                         f"<span class='k'>K⁺</span> leaves for every <span class='h'>H⁺</span> pumped in.")
            else:
                sentence(f"With valinomycin, Δψ stays below {c_thr:.0f} mV for <b>{fmt_time(t_high)}</b>.")
            if t_stop is not None:
                pH_part, psi_part = r["pH_part_mV"][-1], r["psi_mV"][-1]
                parts = (f"It now works against {r['back_pressure_mV'][-1]:.0f} mV: "
                         f"{fmt_mV(pH_part)} from ΔpH and {fmt_mV(psi_part)} from Δψ.")
                if K_left < 0.02:
                    why = ("The internal <span class='k'>K⁺</span> was used up, so the pumped H⁺ "
                           "charged the membrane until Δψ stopped the pump. " + parts)
                elif pH_part >= psi_part:
                    why = ("The K⁺ did not run out: the lumen became too acidic "
                           f"(<span class='ph'>pH {r['pH_in'][i_end]:.2f}</span>). " + parts)
                else:
                    why = parts
                sentence(f"The pump stops after <b>{fmt_time(t_stop)}</b>. {why}")
            else:
                sentence("The pump is still running at the end. Increase the simulated time.")
            sentence(f"<b>{100 * (1 - K_left):.0f} %</b> of the <span class='k'>K⁺</span> inside was exported "
                     f"({fmt_count(exported)} of {fmt_count(r['K_ions_start'])} ions).")
        with left:
            def flux_or_zero(x):
                return x if abs(x) >= 0.01 * c_rate else 0.0
            show(vesicle_pair(
                dict(title="Start",
                     lines=[(f"[K⁺] {c_Kin:.3g} mM", K_COLOR), (f"pH {c_pH:.2f}", PH_COLOR),
                            (f"Δψ {fmt_mV(r['psi_mV'][0])}", INK)],
                     psi_mV=r["psi_mV"][0],
                     arrows=[(flux_or_zero(r["J_K"][1]), K_COLOR, "K⁺"), (flux_or_zero(r["J_H"][1]), H_COLOR, "H⁺")],
                     fill=K_LIGHT, fill_level=0.8),
                dict(title=f"After {fmt_time(t[-1])}",
                     lines=[(f"[K⁺] {r['K_in_mM'][i_end]:.3g} mM", K_COLOR), (f"pH {r['pH_in'][i_end]:.2f}", PH_COLOR),
                            (f"Δψ {fmt_mV(r['psi_mV'][i_end])}", INK)],
                     psi_mV=r["psi_mV"][i_end],
                     arrows=[(flux_or_zero(r["J_K"][i_end]), K_COLOR, "K⁺"),
                             (flux_or_zero(r["J_H"][i_end]), H_COLOR, "H⁺")],
                     fill=K_LIGHT, fill_level=0.8 * K_left),
                f"outside: [K⁺] {c_Kout:.3g} mM and pH {c_pH:.2f}, constant; valinomycin present"))

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = new_axes("Δψ / mV", t[-1])
            ax.axhline(0, color=MUTED, linewidth=0.8)
            ax.axhline(c_thr, color=MUTED, linestyle=":", linewidth=1)
            label_line(ax, c_thr, f"{c_thr:.0f} mV", MUTED)
            ax.plot(t, r0["psi_mV"], color=MUTED, linewidth=1.8, linestyle="--", label="without valinomycin")
            ax.plot(t, r["psi_mV"], color=INK, linewidth=2.2, label="with valinomycin")
            show(finish(fig, ax, legend=True))
            st.caption("Without valinomycin, a few hundred H⁺ charge the membrane within milliseconds. "
                       "With valinomycin, Δψ follows the K⁺ Nernst potential.")
        with c2:
            fig, ax = new_axes("pH inside", t[-1])
            ax.plot(t, r0["pH_in"], color=PH_COLOR, linewidth=1.5, linestyle="--", label="without valinomycin")
            ax.plot(t, r["pH_in"], color=PH_COLOR, linewidth=2.2, label="with valinomycin")
            show(finish(fig, ax, legend=True))
            st.caption("With valinomycin, the pump keeps running and acidifies the lumen until the "
                       "buffer is exhausted.")
        c3, c4 = st.columns(2)
        with c3:
            fig, ax = new_axes("K⁺ left inside / %", t[-1])
            ax.plot(t, 100 * r["K_ions"] / r["K_ions_start"], color=K_COLOR, linewidth=2.2)
            ax.set_ylim(0, 105)
            show(finish(fig, ax))
            st.caption("Each K⁺ that leaves compensates the charge of one pumped H⁺.")
        with c4:
            fig, ax = new_axes("mV", t[-1])
            ax.axhline(r["stop_mV"], color=H_COLOR, linestyle="--", linewidth=1)
            label_line(ax, r["stop_mV"], "pump stops", H_COLOR, va="bottom")
            ax.axhline(0, color=MUTED, linewidth=0.8)
            ax.plot(t, r["back_pressure_mV"], color=INK, linewidth=2.2, label="total")
            ax.plot(t, r["pH_part_mV"], color=PH_COLOR, linewidth=1.8, label="from ΔpH")
            ax.plot(t, r["psi_mV"], color=INK, linewidth=1.2, linestyle="--", label="from Δψ")
            curves = np.r_[r["back_pressure_mV"], r["pH_part_mV"], r["psi_mV"]]
            ax.set_ylim(min(0.0, np.nanmin(curves)) - 5, max(1.15 * r["stop_mV"], np.nanmax(curves) * 1.05))
            show(finish(fig, ax, legend=True))
            st.caption("What the pump works against (with valinomycin). It stops when the total "
                       "reaches its driving energy per charge.")

        df = pd.DataFrame({"time_s": t, "delta_psi_mV": r["psi_mV"], "pH_inside": r["pH_in"],
                           "K_inside_mM": r["K_in_mM"], "K_inside_ions": r["K_ions"],
                           "H_pumped_total": r["H_pumped"], "pump_H_per_s": r["J_H"],
                           "K_flow_into_vesicle_per_s": r["J_K"], "back_pressure_mV": r["back_pressure_mV"],
                           "delta_psi_without_valinomycin_mV": r0["psi_mV"],
                           "pH_inside_without_valinomycin": r0["pH_in"]})
        params = dict(outer_diameter_nm=c_d, bilayer_thickness_nm=4.5, capacitance_uF_per_cm2=1.0,
                      temperature_C=25.0, pH_start=c_pH, buffer_mM=c_buf, buffer_pKa=c_pKa,
                      K_inside_start_mM=c_Kin, K_outside_mM=c_Kout, pump_max_H_per_s=c_rate,
                      pump_driving_energy_kJ_per_mol=c_dG, pump_slowdown_width_mV=c_w,
                      pump_stopping_potential_mV=f"{r['stop_mV']:.2f}", inner_volume_L=f"{r['volume_L']:.6e}")
        st.download_button("Download data as CSV",
                           csv_with_header("Proton pump with valinomycin simulation", params, df),
                           file_name=f"pump_valinomycin_Kin{c_Kin:g}_Kout{c_Kout:g}_buffer{c_buf:g}mM.csv",
                           mime="text/csv", key="c_csv", type="primary")

        with st.expander("How the model works"):
            st.markdown(
                "Δψ comes from the net charge on the membrane: protons pumped in minus K⁺ that left. "
                "Valinomycin moves K⁺ along its electrochemical gradient, fast enough (~0.1 ms) "
                "that Δψ stays at the K⁺ Nernst potential while K⁺ is available:")
            st.latex(r"\Delta\psi = \frac{e\,(N_\mathrm{H^+,pumped} - N_\mathrm{K^+,exported})}{C}"
                     r"\qquad J_\mathrm{K^+} \propto -(\Delta\psi - E_\mathrm{K})")
            st.markdown(
                "The pump is the same as in the 'Proton pump' tab and feels both Δψ and ΔpH. "
                "The pumped protons acidify the lumen according to the buffer capacity "
                "(buffer and water). When the internal K⁺ runs out, E_K rises steeply, K⁺ export "
                "stops by itself, and the protons charge the membrane as without valinomycin.")
            st.markdown(
                "Simplifications: no proton leak, no lipid headgroup buffering, bilayer 4.5 nm, "
                "capacitance 1 µF cm⁻², 25 °C. The outside is a large reservoir with constant pH and [K⁺].")
