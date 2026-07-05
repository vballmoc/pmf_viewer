import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ------------------------------------------------------------
# Educational model:
# Reconstitution of one protein type into liposomes.
#
# Version 3:
# - Fixed orientation-class binomial reference using math.comb()
# - Protein loading compared against ideal Poisson distribution
# - Selectable occupancy class, e.g. liposomes with exactly 4 proteins
# - Orientation composition shown for that class:
#       4 out / 0 in, 3 out / 1 in, etc.
#
# This is a Monte Carlo teaching model, not a mechanistic reconstitution model.
# ------------------------------------------------------------

st.set_page_config(page_title="Liposome protein reconstitution", layout="wide")

st.title("Protein reconstitution into liposomes")

st.markdown("""
This simulation models reconstitution of **one protein type** into liposomes.

In the simplest case, proteins distribute randomly among liposomes, giving a
Poisson distribution. The model also allows two experimentally relevant deviations:

1. proteins may preferentially insert into liposomes that already contain protein;
2. protein orientation may be correlated within the same liposome.
""")

# -----------------------------
# Sidebar parameters
# -----------------------------
st.sidebar.header("Lab parameters")

protein_per_liposome = st.sidebar.number_input(
    "Protein A added per liposome",
    min_value=0.0,
    max_value=1000.0,
    value=2.0,
    step=0.1,
)

reconstitution_efficiency_percent = st.sidebar.number_input(
    "Reconstitution efficiency / %",
    min_value=0.0,
    max_value=100.0,
    value=60.0,
    step=1.0,
)

effective_lambda = protein_per_liposome * reconstitution_efficiency_percent / 100.0

st.sidebar.header("Orientation")

orientation_inside_out_percent = st.sidebar.slider(
    "Inside-out orientation / %",
    min_value=0,
    max_value=100,
    value=50,
    step=1,
    help="0 = all inside-in; 100 = all inside-out."
)

orientation_coupling_percent = st.sidebar.slider(
    "Orientation follows previous protein / %",
    min_value=0,
    max_value=100,
    value=0,
    step=1,
    help="0 = every protein orientation is independent; 100 = proteins in a liposome tend to follow the existing orientation."
)

st.sidebar.header("Non-random insertion")

occupied_preference_percent = st.sidebar.slider(
    "Preference for already occupied liposomes / %",
    min_value=0,
    max_value=100,
    value=0,
    step=1,
    help="0 = random Poisson-like loading; 100 = strong clustering into liposomes already containing A."
)

st.sidebar.header("Simulation")

n_liposomes = st.sidebar.number_input(
    "Number of simulated liposomes",
    min_value=100,
    max_value=1_000_000,
    value=100_000,
    step=10_000,
)

random_seed = st.sidebar.number_input(
    "Random seed",
    min_value=0,
    max_value=1_000_000,
    value=1,
    step=1,
)

max_bar = st.sidebar.slider(
    "Show loading histogram up to n proteins",
    min_value=5,
    max_value=50,
    value=15,
    step=1,
)

selected_class = st.sidebar.number_input(
    "Orientation analysis: occupancy class",
    min_value=1,
    max_value=100,
    value=4,
    step=1,
    help="For example, 4 means: analyze only liposomes containing exactly 4 proteins."
)

# -----------------------------
# Helper functions
# -----------------------------
def poisson_pmf(k_values, lam):
    probs = np.zeros_like(k_values, dtype=float)
    if lam == 0:
        probs[k_values == 0] = 1.0
        return probs

    probs[0] = np.exp(-lam)
    for i in range(1, len(k_values)):
        probs[i] = probs[i - 1] * lam / i
    return probs


def binomial_pmf_for_orientation(n, p):
    """Probability of k inside-out proteins among n proteins."""
    return np.array([
        math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
        for k in range(n + 1)
    ])


def simulate_reconstitution(
    n_lipo,
    lambda_eff,
    p_inside_out,
    occupied_preference,
    orientation_coupling,
    seed,
):
    rng = np.random.default_rng(seed)

    n_inserted = rng.poisson(lambda_eff * n_lipo)

    counts = np.zeros(n_lipo, dtype=np.int32)
    inside_out_counts = np.zeros(n_lipo, dtype=np.int32)
    inside_in_counts = np.zeros(n_lipo, dtype=np.int32)

    if n_inserted == 0:
        return counts, inside_out_counts, inside_in_counts, 0

    occupied_indices = []

    for _ in range(n_inserted):
        use_occupied = (
            rng.random() < occupied_preference
            and len(occupied_indices) > 0
        )

        if use_occupied:
            idx = occupied_indices[rng.integers(0, len(occupied_indices))]
        else:
            idx = rng.integers(0, n_lipo)

        was_empty = counts[idx] == 0
        counts[idx] += 1

        if was_empty:
            occupied_indices.append(idx)
            is_inside_out = rng.random() < p_inside_out
        else:
            if rng.random() < orientation_coupling:
                if inside_out_counts[idx] > inside_in_counts[idx]:
                    is_inside_out = True
                elif inside_in_counts[idx] > inside_out_counts[idx]:
                    is_inside_out = False
                else:
                    is_inside_out = rng.random() < p_inside_out
            else:
                is_inside_out = rng.random() < p_inside_out

        if is_inside_out:
            inside_out_counts[idx] += 1
        else:
            inside_in_counts[idx] += 1

    return counts, inside_out_counts, inside_in_counts, n_inserted


# -----------------------------
# Run simulation
# -----------------------------
p_inside_out = orientation_inside_out_percent / 100.0
occupied_preference = occupied_preference_percent / 100.0
orientation_coupling = orientation_coupling_percent / 100.0

counts, inside_out_counts, inside_in_counts, n_inserted = simulate_reconstitution(
    int(n_liposomes),
    effective_lambda,
    p_inside_out,
    occupied_preference,
    orientation_coupling,
    int(random_seed),
)

# Ideal Poisson comparison
k_values = np.arange(0, max_bar + 1)
ideal_poisson = poisson_pmf(k_values, effective_lambda)

# Simulated loading histogram
hist_counts = np.array([(counts == k).mean() for k in k_values])
above_max_fraction = (counts > max_bar).mean()

# Summary metrics
empty_fraction = (counts == 0).mean()
single_fraction = (counts == 1).mean()
multi_fraction = (counts >= 2).mean()
mean_inserted = counts.mean()
var_inserted = counts.var()

occupied = counts > 0
mean_per_occupied = counts[occupied].mean() if occupied.any() else 0.0

only_inside_out = ((inside_out_counts > 0) & (inside_in_counts == 0)).mean()
only_inside_in = ((inside_in_counts > 0) & (inside_out_counts == 0)).mean()
mixed_orientation = ((inside_out_counts > 0) & (inside_in_counts > 0)).mean()

if occupied.any():
    occupied_only_inside_out = ((inside_out_counts > 0) & (inside_in_counts == 0) & occupied).sum() / occupied.sum()
    occupied_only_inside_in = ((inside_in_counts > 0) & (inside_out_counts == 0) & occupied).sum() / occupied.sum()
    occupied_mixed = ((inside_out_counts > 0) & (inside_in_counts > 0) & occupied).sum() / occupied.sum()
else:
    occupied_only_inside_out = 0.0
    occupied_only_inside_in = 0.0
    occupied_mixed = 0.0

# Orientation distribution for selected occupancy class
selected_class = int(selected_class)
class_mask = counts == selected_class
n_class = class_mask.sum()

if n_class > 0:
    out_in_class = inside_out_counts[class_mask]
    orientation_class_fractions = np.array([
        (out_in_class == k).mean()
        for k in range(selected_class + 1)
    ])
else:
    orientation_class_fractions = np.zeros(selected_class + 1)

orientation_independent_reference = binomial_pmf_for_orientation(selected_class, p_inside_out)

# -----------------------------
# Metrics display
# -----------------------------
st.subheader("Main parameters")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Added A/liposome", f"{protein_per_liposome:.2f}")
with col2:
    st.metric("Efficiency", f"{reconstitution_efficiency_percent:.1f}%")
with col3:
    st.metric("Effective λ", f"{effective_lambda:.2f}")
with col4:
    st.metric("Inserted proteins", f"{n_inserted:,}")

st.subheader("Distribution outcome")

col5, col6, col7, col8 = st.columns(4)

with col5:
    st.metric("Empty liposomes", f"{empty_fraction*100:.1f}%")
with col6:
    st.metric("Exactly one A", f"{single_fraction*100:.1f}%")
with col7:
    st.metric("Two or more A", f"{multi_fraction*100:.1f}%")
with col8:
    st.metric("Mean A/liposome", f"{mean_inserted:.2f}")

col9, col10, col11, col12 = st.columns(4)

with col9:
    st.metric("Variance", f"{var_inserted:.2f}")
with col10:
    st.metric("Mean A/occupied liposome", f"{mean_per_occupied:.2f}")
with col11:
    st.metric("Preference occupied", f"{occupied_preference_percent:.0f}%")
with col12:
    st.metric("Orientation coupling", f"{orientation_coupling_percent:.0f}%")

# -----------------------------
# Plots
# -----------------------------
left, right = st.columns(2)

with left:
    fig, ax = plt.subplots()
    width = 0.4
    ax.bar(k_values - width/2, ideal_poisson * 100, width=width, label="ideal Poisson")
    ax.bar(k_values + width/2, hist_counts * 100, width=width, label="simulation")
    ax.set_xlabel("Number of protein A per liposome")
    ax.set_ylabel("Liposomes / %")
    ax.set_title("Protein loading distribution")
    ax.legend()
    if above_max_fraction > 0:
        ax.text(
            0.98,
            0.95,
            f">{max_bar}: {above_max_fraction*100:.2f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
        )
    st.pyplot(fig)

with right:
    labels = ["empty", "inside-out only", "inside-in only", "mixed orientation"]
    values = [
        empty_fraction * 100,
        only_inside_out * 100,
        only_inside_in * 100,
        mixed_orientation * 100,
    ]
    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylabel("All liposomes / %")
    ax.set_title("Orientation classes")
    ax.tick_params(axis="x", rotation=25)
    st.pyplot(fig)

fig, ax = plt.subplots()
labels_occ = ["inside-out only", "inside-in only", "mixed orientation"]
values_occ = [
    occupied_only_inside_out * 100,
    occupied_only_inside_in * 100,
    occupied_mixed * 100,
]
ax.bar(labels_occ, values_occ)
ax.set_ylabel("Occupied liposomes / %")
ax.set_title("Orientation classes among occupied liposomes")
ax.tick_params(axis="x", rotation=20)
st.pyplot(fig)

# -----------------------------
# New selected-class orientation analysis
# -----------------------------
st.subheader("Orientation distribution within one occupancy class")

st.markdown(
    f"Selected class: liposomes containing exactly **{selected_class}** protein(s). "
    f"Number of simulated liposomes in this class: **{n_class:,}**."
)

labels_class = [
    f"{k} out / {selected_class-k} in"
    for k in range(selected_class + 1)
]

fig, ax = plt.subplots()
x = np.arange(selected_class + 1)
width = 0.4

ax.bar(
    x - width/2,
    orientation_independent_reference * 100,
    width=width,
    label="independent binomial expectation",
)
ax.bar(
    x + width/2,
    orientation_class_fractions * 100,
    width=width,
    label="simulation",
)

ax.set_xticks(x)
ax.set_xticklabels(labels_class, rotation=30, ha="right")
ax.set_ylabel(f"Liposomes in class {selected_class} / %")
ax.set_title(f"Orientation composition among liposomes with {selected_class} protein(s)")
ax.legend()

if n_class == 0:
    ax.text(
        0.5,
        0.5,
        "No liposomes in this class.\nIncrease simulation size or choose another class.",
        transform=ax.transAxes,
        ha="center",
        va="center",
    )

st.pyplot(fig)

# Show direct table-like output in markdown
if n_class > 0:
    st.markdown("Direct simulation outcome for this class:")
    rows = []
    for k, frac in enumerate(orientation_class_fractions):
        rows.append(
            f"- **{k} inside-out / {selected_class-k} inside-in**: "
            f"{frac*100:.2f}% "
            f"({int(round(frac*n_class)):,} liposomes)"
        )
    st.markdown("\n".join(rows))

# -----------------------------
# Teaching notes
# -----------------------------
st.subheader("Teaching interpretation")

if occupied_preference_percent == 0:
    preference_text = (
        "With zero occupied-liposome preference, the simulated loading closely follows "
        "the ideal Poisson distribution."
    )
else:
    preference_text = (
        "With occupied-liposome preference, proteins are more likely to insert into "
        "liposomes that already contain protein. This produces more empty liposomes "
        "and more highly occupied liposomes than an ideal Poisson distribution."
    )

if orientation_coupling_percent == 0:
    coupling_text = (
        "With zero orientation coupling, each protein orientation is chosen independently "
        "from the global orientation bias."
    )
else:
    coupling_text = (
        "With orientation coupling, later proteins tend to follow the orientation already "
        "present in the same liposome. This shifts the selected-class distribution away "
        "from the binomial expectation."
    )

st.markdown(f"""
The experimentally chosen protein/liposome ratio and reconstitution efficiency combine into:

`effective λ = added A/liposome × efficiency = {effective_lambda:.2f}`

{preference_text}

{coupling_text}
""")

st.subheader("Important simplifications")

st.markdown("""
- Liposomes are assumed to be identical in size.
- Protein insertion events are treated as discrete stochastic events.
- Reconstitution efficiency reduces the effective number of inserted proteins.
- The occupied-liposome preference is a phenomenological clustering parameter.
- Orientation coupling is phenomenological and does not model a molecular mechanism.
- This is intended for teaching and intuition, not quantitative fitting.
""")
