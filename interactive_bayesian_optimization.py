# Interactive Visualization Tools - interactive explainers for ML methods
# Copyright (C) 2026  Inbum Chung
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Interactive Bayesian Optimization Explorer

Run with:
    streamlit run interactive_bayesian_optimization.py
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel

st.set_page_config(page_title="Bayesian Optimization Explorer", layout="wide")

# --------------------------------------------------------------------------------------
# Palette (validated: references/palette.md - categorical order, single-hue sequential)
# --------------------------------------------------------------------------------------

COLOR_TRUTH = "#ffffff"       # bright/white - reads clearly against a dark chart surface
COLOR_MEAN = "#2a78d6"        # categorical slot 1 (blue) - GP prediction
COLOR_BAND = "rgba(42,120,214,0.16)"
COLOR_SAMPLE = "#eb6834"      # categorical slot 2 (orange) - observed data
COLOR_CANDIDATE = "#eda100"   # categorical slot 4 (yellow) - next-point suggestion
COLOR_ACQ = "#1baf7a"         # categorical slot 3 (aqua) - acquisition function
SEQ_TRUTH_MEAN = "Blues"      # single-hue sequential: function value
SEQ_STD = "Oranges"           # single-hue sequential: uncertainty
SEQ_ACQ = "Greens"            # single-hue sequential: acquisition
PLOT_TEMPLATE = "plotly_white"

# --------------------------------------------------------------------------------------
# Safe expression evaluation for custom ground-truth functions
# --------------------------------------------------------------------------------------

_SAFE_FUNCS = {
    name: getattr(np, name)
    for name in [
        "sin", "cos", "tan", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh",
        "exp", "log", "log2", "log10", "sqrt", "abs", "power", "maximum", "minimum",
        "pi", "e", "clip", "where", "sign",
    ]
}


def safe_eval(expr, **vars_):
    namespace = dict(_SAFE_FUNCS)
    namespace.update(vars_)
    return eval(expr, {"__builtins__": {}}, namespace)


def validate_expr(expr, dim):
    try:
        if dim == 1:
            safe_eval(expr, x=np.array([0.1, 0.5]))
        else:
            safe_eval(expr, x1=np.array([0.1, 0.5]), x2=np.array([0.2, 0.4]))
        return True, ""
    except Exception as e:
        return False, str(e)


# --------------------------------------------------------------------------------------
# Ground-truth test functions
# --------------------------------------------------------------------------------------

def forrester(x):
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4)


def sine_quadratic(x):
    return np.sin(3 * x) + 0.3 * x ** 2


def branin(x1, x2):
    a, b, c, r, s, t = 1.0, 5.1 / (4 * np.pi ** 2), 5.0 / np.pi, 6.0, 10.0, 1.0 / (8 * np.pi)
    return a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s


def six_hump_camel(x1, x2):
    return (4 - 2.1 * x1 ** 2 + x1 ** 4 / 3) * x1 ** 2 + x1 * x2 + (-4 + 4 * x2 ** 2) * x2 ** 2


GT_1D = {
    "Forrester (classic 1D test function)": (0.0, 1.0),
    "Sine + quadratic": (-2.0, 2.0),
    "Custom expression": (-2.0, 2.0),
}

GT_2D = {
    "Branin (rescaled)": ((-5.0, 10.0), (0.0, 15.0)),
    "Six-Hump Camel": ((-2.0, 2.0), (-1.0, 1.0)),
    "Custom expression": ((-2.0, 2.0), (-2.0, 2.0)),
}


def make_truth_fn(dim, name, custom_expr=None):
    if dim == 1:
        if name == "Forrester (classic 1D test function)":
            return lambda X: forrester(X[:, 0])
        if name == "Sine + quadratic":
            return lambda X: sine_quadratic(X[:, 0])
        return lambda X: safe_eval(custom_expr, x=X[:, 0])
    else:
        if name == "Branin (rescaled)":
            return lambda X: branin(X[:, 0], X[:, 1])
        if name == "Six-Hump Camel":
            return lambda X: six_hump_camel(X[:, 0], X[:, 1])
        return lambda X: safe_eval(custom_expr, x1=X[:, 0], x2=X[:, 1])


# --------------------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------------------

def sample_uniform(bounds, n, rng):
    dim = len(bounds)
    u = rng.uniform(size=(n, dim))
    for i, (lo, hi) in enumerate(bounds):
        u[:, i] = lo + u[:, i] * (hi - lo)
    return u


def sample_lhs(bounds, n, seed):
    dim = len(bounds)
    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    u = sampler.random(n)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    return qmc.scale(u, lo, hi)


# --------------------------------------------------------------------------------------
# GP model + acquisition
# --------------------------------------------------------------------------------------

def fit_gp(X, y, dim, cfg):
    base_cls = RBF if cfg["kernel_name"] == "RBF" else Matern
    base_kwargs = {"length_scale": cfg["length_scale"]}
    if base_cls is Matern:
        base_kwargs["nu"] = 2.5

    if cfg["hp_mode"] == "Manual":
        base = base_cls(length_scale_bounds="fixed", **base_kwargs)
        kernel = ConstantKernel(cfg["amplitude"], constant_value_bounds="fixed") * base
        alpha = max(cfg["gp_noise"], 1e-10)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=False,
                                       optimizer=None, random_state=0)
    else:
        base = base_cls(length_scale_bounds=(1e-2, 1e2), **base_kwargs)
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * base
        alpha = max(cfg["noise_std"] ** 2, 1e-8)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True,
                                       n_restarts_optimizer=6, random_state=0)
    gp.fit(X, y)
    return gp


def expected_improvement(mu, sigma, y_best, xi=0.01):
    sigma = np.maximum(sigma, 1e-9)
    imp = y_best - mu - xi
    z = imp / sigma
    ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
    return np.where(sigma < 1e-9, 0.0, ei)


def probability_of_improvement(mu, sigma, y_best, xi=0.01):
    sigma = np.maximum(sigma, 1e-9)
    z = (y_best - mu - xi) / sigma
    return norm.cdf(z)


ACQUISITION_FUNCTIONS = {
    "Expected Improvement": expected_improvement,
    "Probability of Improvement": probability_of_improvement,
}


# --------------------------------------------------------------------------------------
# Session state (only domain-defining config lives here; GP/kernel knobs are read live
# from the sidebar every run, so tweaking them never wipes collected samples)
# --------------------------------------------------------------------------------------

def reset_experiment(dim, bounds, truth_name, custom_expr, seed):
    ss = st.session_state
    ss.active_dim = dim
    ss.active_bounds = bounds
    ss.active_truth_name = truth_name
    ss.active_custom_expr = custom_expr
    ss.truth_fn = make_truth_fn(dim, truth_name, custom_expr)
    ss.rng = np.random.default_rng(seed)
    ss.X = None
    ss.y = None
    ss.history = []
    ss.initialized = False
    ss.manual_init_points = []


def ensure_state():
    if "active_dim" not in st.session_state:
        reset_experiment(1, [GT_1D["Forrester (classic 1D test function)"]],
                          "Forrester (classic 1D test function)", "sin(3*x) + 0.3*x**2", 42)


def observe(X_new, noise_std):
    ss = st.session_state
    y_new = ss.truth_fn(X_new)
    if noise_std > 0:
        y_new = y_new + ss.rng.normal(0, noise_std, size=y_new.shape)
    return y_new


def add_points(X_new, y_new):
    ss = st.session_state
    if ss.X is None:
        ss.X, ss.y = X_new, y_new
    else:
        ss.X = np.vstack([ss.X, X_new])
        ss.y = np.concatenate([ss.y, y_new])


def log_history(grid, dim, cfg):
    ss = st.session_state
    if ss.X is None or len(ss.X) < 2:
        return
    gp = fit_gp(ss.X, ss.y, dim, cfg)
    mu, std = gp.predict(grid, return_std=True)
    truth = ss.truth_fn(grid)
    rmse = float(np.sqrt(np.mean((mu - truth) ** 2)))
    mean_var = float(np.mean(std ** 2))
    ss.history.append({
        "iteration": len(ss.X),
        "rmse": rmse,
        "mean_var": mean_var,
        "best_y": float(np.min(ss.y)),
    })


# --------------------------------------------------------------------------------------
# Plotting - 1D
# --------------------------------------------------------------------------------------

def plot_1d(gp, grid, show_truth=True, candidate_x=None, candidate_label=None,
            acq_values=None, acq_name=None):
    ss = st.session_state
    x = grid[:, 0]

    # Always computed (regardless of show_truth) so the y-axis range is stable and
    # never jumps when the ground truth is toggled on/off.
    truth_full = ss.truth_fn(grid)
    range_values = [truth_full]

    # Always two rows - even in Manual mode with no acquisition curve - so the top
    # plot's size/proportions never shift depending on which mode is selected.
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.06,
    )

    if show_truth:
        fig.add_trace(go.Scatter(x=x, y=truth_full, mode="lines", name="Ground truth",
                                  line=dict(color=COLOR_TRUTH, dash="solid", width=4)), row=1, col=1)

    if gp is not None:
        mu, std = gp.predict(grid, return_std=True)
        upper, lower = mu + 1.96 * std, mu - 1.96 * std
        range_values += [upper, lower]
        fig.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]),
                                  y=np.concatenate([upper, lower[::-1]]),
                                  fill="toself", fillcolor=COLOR_BAND,
                                  line=dict(color="rgba(255,255,255,0)"),
                                  name="95% CI", showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=mu, mode="lines", name="GP mean",
                                  line=dict(color=COLOR_MEAN, dash="dash", width=3)), row=1, col=1)

    if ss.X is not None:
        range_values.append(ss.y)
        fig.add_trace(go.Scatter(x=ss.X[:, 0], y=ss.y, mode="markers", name="Samples",
                                  marker=dict(color=COLOR_SAMPLE, size=10, symbol="circle",
                                              line=dict(color="white", width=1.5))), row=1, col=1)

    if candidate_x is not None:
        if show_truth:
            cy = ss.truth_fn(np.array([[candidate_x]]))[0]
        elif gp is not None:
            cy = gp.predict(np.array([[candidate_x]]))[0]
        else:
            cy = 0.0
        fig.add_trace(go.Scatter(x=[candidate_x], y=[cy], mode="markers",
                                  name=candidate_label or "Candidate",
                                  marker=dict(color=COLOR_CANDIDATE, size=16, symbol="star",
                                              line=dict(color="black", width=1.5))), row=1, col=1)
        fig.add_vline(x=candidate_x, line=dict(color="gray", dash="dot"), row=1, col=1)
        fig.add_vline(x=candidate_x, line=dict(color="gray", dash="dot"), row=2, col=1)

    if acq_values is not None:
        fig.add_trace(go.Scatter(x=x, y=acq_values, mode="lines", name=acq_name,
                                  line=dict(color=COLOR_ACQ, width=3), fill="tozeroy",
                                  fillcolor="rgba(27,175,122,0.15)"), row=2, col=1)
        fig.update_yaxes(title_text=acq_name, row=2, col=1)
    else:
        # An invisible trace keeps row 2 a real, properly framed subplot (with its
        # own axis box) even though there's nothing to plot in it yet.
        fig.add_trace(go.Scatter(x=[x.min(), x.max()], y=[0, 0], mode="lines",
                                  line=dict(color="rgba(0,0,0,0)"), showlegend=False,
                                  hoverinfo="skip"), row=2, col=1)
        fig.update_yaxes(title_text="Acquisition", range=[0, 1], row=2, col=1)
        fig.add_annotation(text="Pick Expected Improvement or Probability of Improvement to see it here",
                            xref="x2 domain", yref="y2 domain", x=0.5, y=0.5,
                            showarrow=False, font=dict(size=12, color="gray"))
    fig.update_xaxes(title_text="x", row=2, col=1)

    all_y = np.concatenate([np.ravel(v) for v in range_values])
    pad = 0.05 * (all_y.max() - all_y.min() + 1e-9)
    fig.update_yaxes(title_text="f(x)", range=[all_y.min() - pad, all_y.max() + pad], row=1, col=1)
    fig.update_layout(height=580, margin=dict(l=10, r=10, t=30, b=10),
                       template=PLOT_TEMPLATE,
                       legend=dict(orientation="h", yanchor="bottom", y=1.02),
                       font=dict(size=14))
    return fig


# --------------------------------------------------------------------------------------
# Plotting - 2D
# --------------------------------------------------------------------------------------

def build_2d_grid(bounds, n=80):
    x1 = np.linspace(bounds[0][0], bounds[0][1], n)
    x2 = np.linspace(bounds[1][0], bounds[1][1], n)
    X1, X2 = np.meshgrid(x1, x2)
    grid = np.column_stack([X1.ravel(), X2.ravel()])
    return x1, x2, X1, X2, grid


def plot_2d_contours(gp, x1, x2, X1, X2, grid, show_truth=True, candidate=None,
                      acq_grid=None, acq_name=None):
    ss = st.session_state

    # Ground truth is always evaluated so hidden/shown never changes which of the
    # fixed four panels exist - only whether that one panel's data is drawn.
    truth_z = ss.truth_fn(grid).reshape(X1.shape)
    mu_z = std_z = None
    if gp is not None:
        mu, std = gp.predict(grid, return_std=True)
        mu_z, std_z = mu.reshape(X1.shape), std.reshape(X1.shape)
    acq_z = acq_grid.reshape(X1.shape) if acq_grid is not None else None

    # Fixed 2x2 arrangement: truth + acquisition on top (the two things you compare
    # to decide "where next"), GP mean + std on the bottom (what the model believes).
    slots = [
        ("Ground truth" if show_truth else "Ground truth (hidden)",
         truth_z if show_truth else None, SEQ_TRUTH_MEAN),
        (acq_name or "Acquisition (pick EI or PI)", acq_z, SEQ_ACQ),
        ("GP mean", mu_z, SEQ_TRUTH_MEAN),
        ("GP std (uncertainty)", std_z, SEQ_STD),
    ]
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    H, V = 0.11, 0.18
    row_h = (1 - V) / 2

    def colorbar_for(row, col):
        # Route col-1 colorbars into the LEFT margin and col-2 colorbars into the
        # RIGHT margin - never into the gap between columns - so they can never
        # overlap the neighboring panel.
        x, xanchor = (-0.02, "right") if col == 1 else (1.02, "left")
        y = 1 - row_h / 2 if row == 1 else row_h / 2
        return dict(x=x, xanchor=xanchor, y=y, len=row_h * 0.85, yanchor="middle",
                     thickness=12, tickfont=dict(size=10))

    fig = make_subplots(rows=2, cols=2, subplot_titles=[s[0] for s in slots],
                         horizontal_spacing=H, vertical_spacing=V)

    for (row, col), (name, z, colorscale) in zip(positions, slots):
        if z is not None:
            fig.add_trace(go.Contour(x=x1, y=x2, z=z, colorscale=colorscale,
                                      contours=dict(showlabels=False, coloring="heatmap"),
                                      line_smoothing=0, ncontours=28,
                                      showscale=True, colorbar=colorbar_for(row, col),
                                      name=name), row=row, col=col)
        else:
            fig.update_xaxes(range=[x1.min(), x1.max()], row=row, col=col)
            fig.update_yaxes(range=[x2.min(), x2.max()], row=row, col=col)
        if ss.X is not None:
            fig.add_trace(go.Scatter(x=ss.X[:, 0], y=ss.X[:, 1], mode="markers",
                                      marker=dict(color=COLOR_SAMPLE, size=8,
                                                  line=dict(color="white", width=1.5)),
                                      showlegend=False), row=row, col=col)
        if candidate is not None:
            fig.add_trace(go.Scatter(x=[candidate[0]], y=[candidate[1]], mode="markers",
                                      marker=dict(color=COLOR_CANDIDATE, size=14, symbol="star",
                                                  line=dict(color="black", width=1.5)),
                                      showlegend=False), row=row, col=col)

    fig.update_layout(height=820, margin=dict(l=80, r=80, t=50, b=10),
                       template=PLOT_TEMPLATE, font=dict(size=13))
    return fig


def plot_2d_surfaces(gp, x1, x2, X1, X2, grid, show_truth=True):
    ss = st.session_state

    # Ground truth is always evaluated so the z-axis range never depends on the toggle.
    truth_z = ss.truth_fn(grid).reshape(X1.shape)
    mu_z = None
    if gp is not None:
        mu, _ = gp.predict(grid, return_std=True)
        mu_z = mu.reshape(X1.shape)

    range_values = [truth_z]
    if mu_z is not None:
        range_values.append(mu_z)
    if ss.y is not None:
        range_values.append(ss.y)
    all_z = np.concatenate([np.ravel(v) for v in range_values])
    pad = 0.05 * (all_z.max() - all_z.min() + 1e-9)
    z_range = [all_z.min() - pad, all_z.max() + pad]

    def crisp_cscale_range(z):
        # A little headroom past the data range so a flat/near-flat surface still
        # shows a splash of color, but not so much that everything washes toward
        # one muddy middle tone - most of the colorscale's contrast stays in play.
        zmin, zmax = float(z.min()), float(z.max())
        cpad = 0.12 * (zmax - zmin + 1e-9)
        return zmin - cpad, zmax + cpad

    mesh_x = dict(show=True, color="rgba(20,20,20,0.45)", width=1.5)
    mesh_y = dict(show=True, color="rgba(20,20,20,0.45)", width=1.5)

    fig = go.Figure()

    if show_truth:
        cmin, cmax = crisp_cscale_range(truth_z)
        fig.add_trace(go.Surface(
            x=x1, y=x2, z=truth_z, colorscale=SEQ_TRUTH_MEAN, cmin=cmin, cmax=cmax,
            opacity=0.94, showscale=True, name="Ground truth", showlegend=True,
            contours=dict(x=mesh_x, y=mesh_y),
            colorbar=dict(title="Truth", x=1.02, y=0.75, len=0.42, thickness=14),
        ))
    if mu_z is not None:
        cmin, cmax = crisp_cscale_range(mu_z)
        fig.add_trace(go.Surface(
            x=x1, y=x2, z=mu_z, colorscale=SEQ_STD, cmin=cmin, cmax=cmax,
            opacity=0.94, showscale=True, name="GP mean", showlegend=True,
            contours=dict(x=mesh_x, y=mesh_y),
            colorbar=dict(title="GP mean", x=1.02, y=0.25, len=0.42, thickness=14),
        ))
    if ss.X is not None:
        fig.add_trace(go.Scatter3d(x=ss.X[:, 0], y=ss.X[:, 1], z=ss.y, mode="markers",
                                    name="Samples", showlegend=True,
                                    marker=dict(color=COLOR_ACQ, size=5,
                                                line=dict(color="black", width=1))))

    axis_style = dict(showbackground=True, backgroundcolor="rgb(235,235,238)",
                       gridcolor="rgb(190,190,195)", showgrid=True, zeroline=False)
    fig.update_layout(
        height=600, margin=dict(l=0, r=90, t=30, b=0), template=PLOT_TEMPLATE,
        scene=dict(aspectmode="cube",
                   xaxis=dict(title="x1", **axis_style),
                   yaxis=dict(title="x2", **axis_style),
                   zaxis=dict(title="f", range=z_range, **axis_style)),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0.02),
    )
    return fig


# --------------------------------------------------------------------------------------
# Sidebar - configuration
# --------------------------------------------------------------------------------------

ensure_state()
ss = st.session_state

st.title("Bayesian Optimization Explorer")
st.caption("Configure a ground-truth function, sample it, fit a Gaussian Process, "
           "and watch how Expected Improvement, Probability of Improvement, or manual "
           "picks refine the model.")

with st.sidebar:
    st.header("1. Problem setup")
    dim_choice = st.radio("Design space dimension", [1, 2], horizontal=True,
                           format_func=lambda d: f"{d}D")

    if dim_choice == 1:
        truth_name = st.selectbox("Ground truth function", list(GT_1D.keys()))
        default_bounds = GT_1D[truth_name]
        custom_expr = None
        if truth_name == "Custom expression":
            custom_expr = st.text_input("f(x) =", value="sin(3*x) + 0.3*x**2")
        c1, c2 = st.columns(2)
        lo = c1.number_input("Lower bound", value=float(default_bounds[0]))
        hi = c2.number_input("Upper bound", value=float(default_bounds[1]))
        bounds = [(lo, hi)]
    else:
        truth_name = st.selectbox("Ground truth function", list(GT_2D.keys()))
        default_bounds = GT_2D[truth_name]
        custom_expr = None
        if truth_name == "Custom expression":
            custom_expr = st.text_input("f(x1, x2) =", value="sin(x1) + cos(x2) + 0.1*x1*x2")
        st.markdown("**x1 bounds**")
        c1, c2 = st.columns(2)
        lo1 = c1.number_input("x1 lower", value=float(default_bounds[0][0]))
        hi1 = c2.number_input("x1 upper", value=float(default_bounds[0][1]))
        st.markdown("**x2 bounds**")
        c3, c4 = st.columns(2)
        lo2 = c3.number_input("x2 lower", value=float(default_bounds[1][0]))
        hi2 = c4.number_input("x2 upper", value=float(default_bounds[1][1]))
        bounds = [(lo1, hi1), (lo2, hi2)]

    expr_ok = True
    if custom_expr:
        expr_ok, err = validate_expr(custom_expr, dim_choice)
        if not expr_ok:
            st.error(f"Invalid expression: {err}")

    seed = st.number_input("Random seed", value=42, step=1)

    if st.button("Apply settings & reset experiment", type="primary", disabled=not expr_ok):
        reset_experiment(dim_choice, bounds, truth_name, custom_expr, int(seed))
        st.rerun()

    st.divider()
    st.header("2. GP model")
    st.caption("These apply live to the current data - no reset needed.")
    kernel_name = st.selectbox("Kernel", ["RBF", "Matern(nu=2.5)"])
    hp_mode = st.radio("Hyperparameters", ["Auto (optimize)", "Manual"], horizontal=True)
    noise_std = st.slider("Observation noise (std, used when sampling)", 0.0, 1.0, 0.0, 0.01)

    if hp_mode == "Manual":
        log_amp = st.slider("Signal variance: log10(amplitude)", -2.0, 3.0, 0.0, 0.1)
        amplitude = 10 ** log_amp
        length_scale = []
        for i in range(ss.active_dim):
            active_lo, active_hi = ss.active_bounds[i]
            rng_i = max(active_hi - active_lo, 1e-3)
            default_ls = float(np.clip(rng_i / 4, 0.01, 2 * rng_i))
            ls = st.slider(f"Length scale (x{i+1})", 0.01, float(2 * rng_i), default_ls)
            length_scale.append(ls)
        log_noise = st.slider("GP noise: log10(alpha)", -8.0, 0.0, -6.0, 0.5)
        gp_noise = 10 ** log_noise
    else:
        amplitude, length_scale, gp_noise = 1.0, [1.0] * ss.active_dim, 1e-6

model_cfg = {
    "kernel_name": kernel_name,
    "hp_mode": hp_mode,
    "noise_std": noise_std,
    "amplitude": amplitude,
    "length_scale": length_scale,
    "gp_noise": gp_noise,
}

dim = ss.active_dim
bounds = ss.active_bounds


# --------------------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------------------

tab_setup, tab_optimize, tab_history = st.tabs(
    ["1. Initial sampling", "2. Optimize", "3. History & data"]
)

# ---- Tab 1: initial sampling -----------------------------------------------------------
with tab_setup:
    if not ss.initialized:
        sample_type = st.radio("Sampling type", ["Uniform random", "Latin Hypercube (LHS)", "Manual"],
                                horizontal=True)

        if sample_type in ("Uniform random", "Latin Hypercube (LHS)"):
            n_init = st.slider("Number of initial points", 2, 30, 5)
            if st.button("Generate initial samples"):
                if sample_type == "Uniform random":
                    X_init = sample_uniform(bounds, n_init, ss.rng)
                else:
                    X_init = sample_lhs(bounds, n_init, int(seed))
                y_init = observe(X_init, noise_std)
                add_points(X_init, y_init)
                ss.initialized = True
                st.rerun()

        else:  # Manual
            st.write("Pick coordinates below, preview the point, then add it. Repeat, then finalize.")
            if dim == 1:
                xm = st.slider("x", float(bounds[0][0]), float(bounds[0][1]),
                                float((bounds[0][0] + bounds[0][1]) / 2))
                candidate = np.array([xm])
            else:
                c1, c2 = st.columns(2)
                x1m = c1.slider("x1", float(bounds[0][0]), float(bounds[0][1]),
                                 float((bounds[0][0] + bounds[0][1]) / 2))
                x2m = c2.slider("x2", float(bounds[1][0]), float(bounds[1][1]),
                                 float((bounds[1][0] + bounds[1][1]) / 2))
                candidate = np.array([x1m, x2m])

            preview_y = ss.truth_fn(candidate.reshape(1, -1))[0]
            st.info(f"Candidate point: {np.round(candidate, 3).tolist()}  ->  f = {preview_y:.4f}")

            cc1, cc2, cc3 = st.columns(3)
            if cc1.button("Add point"):
                ss.manual_init_points.append(candidate.copy())
            if cc2.button("Clear all points"):
                ss.manual_init_points = []
            n_added = len(ss.manual_init_points)
            cc3.write(f"**{n_added}** point(s) added")

            if n_added > 0:
                st.dataframe(np.array(ss.manual_init_points), width="stretch")

            if st.button("Finalize initial samples", disabled=n_added < 2):
                X_init = np.array(ss.manual_init_points)
                y_init = observe(X_init, noise_std)
                add_points(X_init, y_init)
                ss.initialized = True
                st.rerun()
    else:
        st.success(f"Initialized with {len(ss.X)} sample(s). Switch to the **2. Optimize** tab to continue.")
        if st.button("Restart sampling (keep current problem settings)"):
            reset_experiment(ss.active_dim, ss.active_bounds, ss.active_truth_name,
                              ss.active_custom_expr, int(seed))
            st.rerun()


# ---- Tab 2: optimize (controls + plot side by side, no scrolling) ---------------------
with tab_optimize:
    if not ss.initialized:
        st.info("Generate initial samples in the **1. Initial sampling** tab first.")
    else:
        gp = fit_gp(ss.X, ss.y, dim, model_cfg)

        if dim == 1:
            grid = np.linspace(bounds[0][0], bounds[0][1], 400).reshape(-1, 1)
        else:
            x1g, x2g, X1, X2, grid = build_2d_grid(bounds, n=80)

        if len(ss.history) == 0 or ss.history[-1]["iteration"] != len(ss.X):
            log_history(grid, dim, model_cfg)

        latest = ss.history[-1]
        dcol1, dcol2, dcol3, dcol4 = st.columns(4)
        dcol1.caption(f"Samples so far &nbsp; **{latest['iteration']}**")
        dcol2.caption(f"Mean GP variance &nbsp; **{latest['mean_var']:.3g}**")
        dcol3.caption(f"RMSE vs ground truth &nbsp; **{latest['rmse']:.3g}**")
        dcol4.caption(f"Best f(x) found &nbsp; **{latest['best_y']:.3g}**")
        st.caption("RMSE vs ground truth is only computable here because this is a synthetic "
                   "benchmark with a known function - shown for teaching purposes only.")
        st.divider()

        left, right = st.columns([1, 2.3], gap="medium")

        with left:
            st.subheader("Controls")
            show_truth = st.toggle("Show ground truth", value=True)
            next_mode = st.radio("Next point selection",
                                  ["Expected Improvement", "Probability of Improvement", "Manual"])

            candidate = None
            acq_values_1d = None
            acq_grid_2d = None
            acq_name = None

            if next_mode in ACQUISITION_FUNCTIONS:
                mu, std = gp.predict(grid, return_std=True)
                y_best = np.min(ss.y)
                acq_name = next_mode
                acq = ACQUISITION_FUNCTIONS[acq_name](mu, std, y_best)
                best_idx = int(np.argmax(acq))
                candidate = grid[best_idx]
                if dim == 1:
                    acq_values_1d = acq
                else:
                    acq_grid_2d = acq
                st.info(f"{acq_name} suggests: {np.round(candidate, 3).tolist()}\n\n"
                        f"{acq_name} = {acq[best_idx]:.4g}")
                add_label = f"Add {acq_name}-suggested point & refit"
            else:
                if dim == 1:
                    xm = st.slider("Next x", float(bounds[0][0]), float(bounds[0][1]),
                                   float((bounds[0][0] + bounds[0][1]) / 2), key="next_x_1d")
                    candidate = np.array([xm])
                else:
                    x1m = st.slider("Next x1", float(bounds[0][0]), float(bounds[0][1]),
                                     float((bounds[0][0] + bounds[0][1]) / 2), key="next_x1_2d")
                    x2m = st.slider("Next x2", float(bounds[1][0]), float(bounds[1][1]),
                                     float((bounds[1][0] + bounds[1][1]) / 2), key="next_x2_2d")
                    candidate = np.array([x1m, x2m])
                mu_c, std_c = gp.predict(candidate.reshape(1, -1), return_std=True)
                st.info(f"GP mean = {mu_c[0]:.4f}\n\nGP std = {std_c[0]:.4f}")
                add_label = "Add manual point & refit"

            if st.button(add_label, type="primary"):
                X_new = candidate.reshape(1, -1)
                y_new = observe(X_new, noise_std)
                add_points(X_new, y_new)
                log_history(grid, dim, model_cfg)
                st.rerun()

        with right:
            if dim == 1:
                fig = plot_1d(gp, grid, show_truth=show_truth, candidate_x=candidate[0],
                              candidate_label=f"{acq_name} suggestion" if acq_name else "Manual candidate",
                              acq_values=acq_values_1d, acq_name=acq_name)
                st.plotly_chart(fig, width="stretch")
            else:
                tab_contour, tab_surface = st.tabs(["2D contours", "3D surfaces"])
                with tab_contour:
                    fig2d = plot_2d_contours(gp, x1g, x2g, X1, X2, grid, show_truth=show_truth,
                                              candidate=candidate, acq_grid=acq_grid_2d, acq_name=acq_name)
                    st.plotly_chart(fig2d, width="stretch")
                with tab_surface:
                    fig3d = plot_2d_surfaces(gp, x1g, x2g, X1, X2, grid, show_truth=show_truth)
                    st.plotly_chart(fig3d, width="stretch")


# ---- Tab 3: history & raw data ----------------------------------------------------------
with tab_history:
    if not ss.initialized:
        st.info("Nothing to show yet - generate initial samples first.")
    else:
        hist = ss.history
        iters = [h["iteration"] for h in hist]

        st.subheader("Convergence history")
        hc1, hc2 = st.columns(2)
        with hc1:
            fig_rmse = go.Figure(go.Scatter(x=iters, y=[h["rmse"] for h in hist],
                                             mode="lines+markers", name="RMSE",
                                             line=dict(color=COLOR_MEAN, width=3),
                                             marker=dict(size=7)))
            fig_rmse.update_layout(title="RMSE vs ground truth", xaxis_title="Number of samples",
                                    yaxis_title="RMSE", height=340, template=PLOT_TEMPLATE,
                                    margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
            st.plotly_chart(fig_rmse, width="stretch")
        with hc2:
            fig_var = go.Figure(go.Scatter(x=iters, y=[h["mean_var"] for h in hist],
                                            mode="lines+markers", name="Mean variance",
                                            line=dict(color=COLOR_SAMPLE, width=3),
                                            marker=dict(size=7)))
            fig_var.update_layout(title="Mean GP variance", xaxis_title="Number of samples",
                                   yaxis_title="Mean variance", height=340, template=PLOT_TEMPLATE,
                                   margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
            st.plotly_chart(fig_var, width="stretch")

        st.subheader("Raw sample data")
        cols = [f"x{i+1}" for i in range(dim)] if dim > 1 else ["x"]
        st.dataframe(
            {**{c: ss.X[:, i] for i, c in enumerate(cols)}, "y": ss.y},
            width="stretch",
        )
