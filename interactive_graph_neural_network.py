# Interactive Visualization Tools - interactive explainers for ML methods
# Copyright (C) 2026  In-Bum Chung
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
Interactive Graph Neural Network Explorer

Step through message passing and aggregation on a fixed 7-node graph, one node
and one layer at a time.

Run with:
    streamlit run interactive_graph_neural_network.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Graph Neural Network Explorer", layout="wide")

# --------------------------------------------------------------------------------------
# Palette (validated: references/palette.md - categorical order, diverging for signed data)
# --------------------------------------------------------------------------------------

COLOR_FOCUS = "#eb6834"       # categorical slot 2 (orange) - the node being updated
COLOR_NEIGHBOR = "#2a78d6"    # categorical slot 1 (blue)   - contributing neighbours
COLOR_MESSAGE = "#1baf7a"     # categorical slot 3 (aqua)   - messages and aggregate
COLOR_SELF = "#eda100"        # categorical slot 4 (yellow) - the node's own state
COLOR_EDGE = "#8e99a4"
SEQ_FEATURE = "RdBu"          # diverging: feature values are signed, zero is meaningful
SEQ_HOPS = "Blues"            # single-hue sequential: hop distance


def _dark_theme():
    """The theme actually applied in the browser, which the viewer can toggle. Falls
    back to the configured base, then to light, so a missing API never breaks startup."""
    for probe in (lambda: st.context.theme.type, lambda: st.get_option("theme.base")):
        try:
            value = probe()
        except Exception:
            continue
        if value:
            return str(value).lower() == "dark"
    return False


DARK = _dark_theme()
PLOT_TEMPLATE = "plotly_dark" if DARK else "plotly_white"
TEXT = "#fafafa" if DARK else "#2b3138"
TEXT_MUTED = "#b6bec7" if DARK else "#4a5560"
LABEL_BG = "rgba(14,17,23,0.72)" if DARK else "rgba(255,255,255,0.78)"

# --------------------------------------------------------------------------------------
# The fixed graph
#
#   0 - 1        4            two triangles joined by the bridge edge 2-3, plus a
#    \ /        / \           leaf (6) four hops from node 0 - far enough that no
#     2 ----- 3 - 5 - 6       3-layer receptive field rooted in the left triangle
#                             ever reaches it.
# --------------------------------------------------------------------------------------

N_NODES = 7

EDGES = [
    (0, 1), (0, 2), (1, 2),   # triangle A: {0, 1, 2}
    (2, 3),                   # bridge
    (3, 4), (3, 5), (4, 5),   # triangle B: {3, 4, 5}
    (5, 6),                   # leaf
]


def spring_layout(n_nodes, edges, seed=3, iterations=300):
    """Fruchterman-Reingold, then PCA-rotated and rescaled into a wide box so the
    picture lands the same way every time."""
    rng = np.random.default_rng(seed)
    pos = rng.normal(0.0, 1.0, (n_nodes, 2))
    A = np.zeros((n_nodes, n_nodes))
    for a, b in edges:
        A[a, b] = A[b, a] = 1.0
    k = 1.0 / np.sqrt(n_nodes)
    temp, cool = 0.1, 0.1 / (iterations + 1)
    for _ in range(iterations):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.clip(np.linalg.norm(delta, axis=-1), 0.01, None)
        force = k ** 2 / dist ** 2 - A * dist / k        # repel everyone, attract edges
        np.fill_diagonal(force, 0.0)
        disp = (delta * force[..., None]).sum(axis=1)
        length = np.clip(np.linalg.norm(disp, axis=1, keepdims=True), 0.01, None)
        pos = pos + disp / length * np.minimum(length, temp)
        temp -= cool

    pos = pos - pos.mean(axis=0)
    _, _, vt = np.linalg.svd(pos, full_matrices=False)   # longest axis -> horizontal
    pos = pos @ vt.T
    span = pos[:, 0].max() - pos[:, 0].min()
    pos = pos * (5.6 / max(span, 1e-9))
    return pos - pos.mean(axis=0)


POS = spring_layout(N_NODES, EDGES)
X_RANGE = [POS[:, 0].min() - 1.05, POS[:, 0].max() + 1.05]
Y_RANGE = [POS[:, 1].min() - 0.62, POS[:, 1].max() + 0.62]

def _point_to_segment(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(float(np.dot(ab, ab)), 1e-9), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def label_positions(pos, edges):
    """Put each node's label on whichever side is emptiest, so text never lands on a
    neighbouring node or on an edge."""
    candidates = [("top center", np.array([0.0, 0.45])),
                  ("bottom center", np.array([0.0, -0.45])),
                  ("middle right", np.array([0.75, 0.0])),
                  ("middle left", np.array([-0.75, 0.0]))]
    out = []
    for v in range(len(pos)):
        best, best_score = candidates[0][0], -1.0
        for name, off in candidates:
            anchor = pos[v] + off
            score = min([np.linalg.norm(anchor - pos[u]) for u in range(len(pos)) if u != v]
                        + [_point_to_segment(anchor, pos[a], pos[b]) for a, b in edges])
            if score > best_score + 1e-9:
                best, best_score = name, score
        out.append(best)
    return out


LABEL_POS = label_positions(POS, EDGES)

# Default node features: a smooth gradient along the graph in dimension 0, so that
# over-smoothing (every node collapsing towards the mean) is easy to see.
DEFAULT_X = np.array([
    [1.0, 0.0, 0.5],
    [0.8, 0.2, 0.0],
    [0.5, 0.4, 1.0],
    [0.0, 0.8, 0.5],
    [-0.4, 0.6, 0.0],
    [-0.7, 0.3, 1.0],
    [-1.0, 0.0, 0.5],
])

# Edge features start varied rather than all-ones, so the edge-mode mechanisms visibly
# do something before you touch anything. Up to three components per edge.
MAX_EDGE_DIM = 3
DEFAULT_EDGE_FEAT = np.round(
    np.random.default_rng(11).uniform(0.3, 1.7, (len(EDGES), MAX_EDGE_DIM)), 2
)
SELF_LOOP_EDGE_FEAT = 1.0     # an edge from a node to itself carries all-ones

EDGE_INDEX = {}
for _i, (_a, _b) in enumerate(EDGES):
    EDGE_INDEX[(_a, _b)] = _i
    EDGE_INDEX[(_b, _a)] = _i


def neighbor_lists(self_loops):
    nbrs = [[] for _ in range(N_NODES)]
    for a, b in EDGES:
        nbrs[a].append(b)
        nbrs[b].append(a)
    out = []
    for v in range(N_NODES):
        ordered = sorted(nbrs[v])
        out.append(([v] + ordered) if self_loops else ordered)
    return out


def hop_distances(source):
    dist = np.full(N_NODES, -1)
    dist[source] = 0
    frontier = [source]
    nbrs = neighbor_lists(False)
    while frontier:
        nxt = []
        for v in frontier:
            for u in nbrs[v]:
                if dist[u] < 0:
                    dist[u] = dist[v] + 1
                    nxt.append(u)
        frontier = nxt
    return dist


# --------------------------------------------------------------------------------------
# Layer configuration
# --------------------------------------------------------------------------------------

MSG_LINEAR = "linear"
MSG_GCN = "gcn (degree-normalised)"
MSG_ATTN = "attention"
MSG_DIFF = "difference"

MSG_OPTIONS = [MSG_LINEAR, MSG_GCN, MSG_ATTN, MSG_DIFF]
AGG_OPTIONS = ["sum", "mean", "max"]
UPD_OPTIONS = ["agg only", "self + agg", "residual"]
ACT_OPTIONS = ["identity", "relu", "tanh"]
EDGE_OPTIONS = ["scale", "concat", "gate"]

PRESETS = {
    "Custom": None,
    "GCN (Kipf & Welling)": dict(msg=MSG_GCN, agg="sum", upd="self + agg", act="relu"),
    "GraphSAGE (mean)": dict(msg=MSG_LINEAR, agg="mean", upd="self + agg", act="relu"),
    "GAT (attention)": dict(msg=MSG_ATTN, agg="sum", upd="agg only", act="relu"),
    "GIN-like (sum)": dict(msg=MSG_LINEAR, agg="sum", upd="self + agg", act="relu"),
    "Max-pool aggregator": dict(msg=MSG_LINEAR, agg="max", upd="self + agg", act="relu"),
}

STAGE_NAMES = ["1. Gather", "2. Message", "3. Aggregate", "4. Update"]


def activate(name, z):
    if name == "relu":
        return np.maximum(z, 0.0)
    if name == "tanh":
        return np.tanh(z)
    return z


def make_weights(dim, kind, seed, edge_dim=1):
    """B_edge and W_gate map an edge feature in R^{edge_dim} into message space R^{dim}."""
    if kind == "identity":
        W = np.eye(dim)
        W_self = np.eye(dim)
        B_edge = np.ones((dim, edge_dim)) / edge_dim
        W_gate = np.ones((dim, edge_dim)) / edge_dim
    else:
        rng = np.random.default_rng(seed)
        W = np.round(rng.normal(0.0, 0.8, (dim, dim)), 2)
        W_self = np.round(rng.normal(0.0, 0.8, (dim, dim)), 2)
        B_edge = np.round(rng.normal(0.0, 0.8, (dim, edge_dim)), 2)
        W_gate = np.round(rng.normal(0.0, 0.8, (dim, edge_dim)), 2)
    a_vec = np.ones(2 * dim) / dim          # attention score vector a in R^{2d}
    return dict(W=W, W_self=W_self, B_edge=B_edge, W_gate=W_gate, a_vec=a_vec)


def attention_scores(v, nbrs, H, W, a_vec, slope=0.2):
    """GAT: e_vu = LeakyReLU(a . [W h_v ; W h_u]), then softmax over N(v)."""
    hv = W @ H[v]
    raw = []
    for u in nbrs:
        z = np.concatenate([hv, W @ H[u]])
        s = float(a_vec @ z)
        raw.append(s if s > 0 else slope * s)
    raw = np.asarray(raw)
    ex = np.exp(raw - raw.max())
    return ex / ex.sum(), raw


def edge_feature(u, v, edge_vals):
    """The feature vector carried by edge (u, v); self-loops carry all-ones."""
    if u == v:
        return np.full(edge_vals.shape[1], SELF_LOOP_EDGE_FEAT)
    return np.asarray(edge_vals[EDGE_INDEX[(u, v)]], dtype=float)


def compute_node(v, nbrs, H, deg, cfg, wts, edge_vals):
    """One node's update, broken into the four stages the stepper walks through."""
    W, W_self = wts["W"], wts["W_self"]
    dim = H.shape[1]
    rows = []

    if cfg["msg"] == MSG_ATTN and nbrs:
        alphas, raw_scores = attention_scores(v, nbrs, H, W, wts["a_vec"])
    else:
        alphas, raw_scores = None, None

    for k, u in enumerate(nbrs):
        h_u = H[u]
        if cfg["msg"] == MSG_DIFF:
            transformed = W @ (h_u - H[v])
        else:
            transformed = W @ h_u

        e_val, gate = None, None
        after_edge = transformed
        if cfg["edge_mode"]:
            e_val = edge_feature(u, v, edge_vals)
            if cfg["edge_mech"] == "scale":
                # broadcasts: one shared scale when edge_dim == 1, else one per dimension
                after_edge = e_val * transformed
            elif cfg["edge_mech"] == "concat":
                after_edge = transformed + wts["B_edge"] @ e_val
            else:  # gate
                gate = 1.0 / (1.0 + np.exp(-(wts["W_gate"] @ e_val)))
                after_edge = gate * transformed

        if cfg["msg"] == MSG_GCN:
            coef = 1.0 / float(np.sqrt(deg[u] * deg[v]))
        elif cfg["msg"] == MSG_ATTN:
            coef = float(alphas[k])
        else:
            coef = 1.0

        rows.append(dict(
            u=u, h_u=h_u, transformed=transformed, e_val=e_val, gate=gate,
            after_edge=after_edge, coef=coef, msg=coef * after_edge,
            score=None if raw_scores is None else float(raw_scores[k]),
        ))

    M = np.array([r["msg"] for r in rows]) if rows else np.zeros((0, dim))
    winners = None
    if len(rows) == 0:
        agg = np.zeros(dim)
    elif cfg["agg"] == "sum":
        agg = M.sum(axis=0)
    elif cfg["agg"] == "mean":
        agg = M.mean(axis=0)
    else:
        agg = M.max(axis=0)
        winners = [rows[int(i)]["u"] for i in M.argmax(axis=0)]

    self_term = W_self @ H[v]
    if cfg["upd"] == "agg only":
        pre = agg
        out = activate(cfg["act"], pre)
    elif cfg["upd"] == "self + agg":
        pre = self_term + agg
        out = activate(cfg["act"], pre)
    else:  # residual
        pre = self_term + agg
        out = H[v] + activate(cfg["act"], pre)

    return dict(v=v, nbrs=list(nbrs), rows=rows, agg=agg, winners=winners,
                self_term=self_term, pre=pre, out=out, h_v=H[v])


def run_forward(X, cfg, wts, edge_vals):
    nbrs = neighbor_lists(cfg["self_loops"])
    deg = np.array([len(n) for n in nbrs], dtype=float)
    layers, H = [], X.copy()
    for _ in range(cfg["n_layers"]):
        H_in = H.copy()
        H_out = np.zeros_like(H_in)
        recs = {}
        for v in range(N_NODES):
            rec = compute_node(v, nbrs[v], H_in, deg, cfg, wts, edge_vals)
            recs[v] = rec
            H_out[v] = rec["out"]
        layers.append(dict(H_in=H_in, H_out=H_out, nodes=recs))
        H = H_out
    return layers


def mean_pairwise_distance(H):
    total, count = 0.0, 0
    for i in range(N_NODES):
        for j in range(i + 1, N_NODES):
            total += float(np.linalg.norm(H[i] - H[j]))
            count += 1
    return total / count


def fmt_vec(v, nd=3):
    return "[" + ", ".join(f"{x:.{nd}f}" for x in np.atleast_1d(v)) + "]"


def fmt_mat(M, nd=2):
    return "\n".join("  [" + "  ".join(f"{x:>6.{nd}f}" for x in row) + "]" for row in np.atleast_2d(M))


# --------------------------------------------------------------------------------------
# Graph drawing
# --------------------------------------------------------------------------------------

NODE_PAD = 0.17     # data units: roughly the radius of a node disc, so lines stop clear of it


def _shorten(p0, p1, pad=NODE_PAD):
    d = p1 - p0
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return p0, p1
    d = d / n
    return p0 + d * pad, p1 - d * pad


def graph_figure(values, focus=None, highlight=None, arrows=None, colorscale=SEQ_FEATURE,
                 reverse=True, cbar_title="value", height=430, fade=True, node_text=None,
                 edge_labels=None):
    """One picture of the graph: node colour = `values`, optional focus/neighbour rings
    and message arrows."""
    highlight = set(highlight or [])
    arrows = arrows or []
    involved = set(highlight) | ({focus} if focus is not None else set())

    fig = go.Figure()

    # Split the edges in two: the ones carrying a message into the focus node are drawn
    # at full strength, everything else is pushed back. Both are clipped to stop at the
    # rim of each node disc.
    def _segments(pairs):
        xs, ys = [], []
        for a, b in pairs:
            p0, p1 = _shorten(POS[a].astype(float), POS[b].astype(float), pad=NODE_PAD)
            xs += [p0[0], p1[0], None]
            ys += [p0[1], p1[1], None]
        return xs, ys

    stepping = bool(fade and involved)
    active = [(a, b) for a, b in EDGES
              if involved and (a == focus or b == focus) and a in involved and b in involved]
    resting = [e for e in EDGES if e not in active]

    if resting:
        rx, ry = _segments(resting)
        fig.add_trace(go.Scatter(x=rx, y=ry, mode="lines", hoverinfo="skip",
                                 line=dict(color=COLOR_EDGE, width=1.4),
                                 opacity=0.45 if stepping else 1.0, showlegend=False))
    if active:
        ax_, ay_ = _segments(active)
        fig.add_trace(go.Scatter(x=ax_, y=ay_, mode="lines", hoverinfo="skip",
                                 line=dict(color=COLOR_NEIGHBOR, width=3.2),
                                 opacity=1.0, showlegend=False))

    # message arrows
    for ar in arrows:
        p0, p1 = _shorten(POS[ar["u"]].astype(float), POS[ar["v"]].astype(float))
        fig.add_annotation(x=p1[0], y=p1[1], ax=p0[0], ay=p0[1],
                           xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2, arrowsize=0.9,
                           arrowwidth=ar.get("w", 2.0),
                           arrowcolor=ar.get("color", COLOR_MESSAGE), opacity=1.0)

    if edge_labels:
        for (a, b), text in edge_labels.items():
            mid = (POS[a] + POS[b]) / 2.0
            fig.add_annotation(x=mid[0], y=mid[1], text=text, showarrow=False,
                               font=dict(size=10, color=TEXT_MUTED),
                               bgcolor=LABEL_BG, borderpad=1)

    line_colors, line_widths, sizes, text_colors = [], [], [], []
    for v in range(N_NODES):
        idle = bool(fade and involved and v not in involved)
        if v == focus:
            line_colors.append(COLOR_FOCUS)
            line_widths.append(4.0)
        elif v in highlight:
            line_colors.append(COLOR_NEIGHBOR)
            line_widths.append(3.0)
        else:
            line_colors.append(COLOR_EDGE)     # neutral: legible on light and dark alike
            line_widths.append(1.5)
        # every disc stays fully opaque - a translucent one shows the edge underneath it
        sizes.append(18 if idle else 27)
        text_colors.append(TEXT_MUTED if idle else TEXT)

    labels = node_text if node_text is not None else [str(v) for v in range(N_NODES)]
    hover = [f"node {v}<br>{cbar_title} = {values[v]:.3f}" for v in range(N_NODES)]

    fig.add_trace(go.Scatter(
        x=POS[:, 0], y=POS[:, 1], mode="markers+text",
        text=labels, textposition=LABEL_POS,
        textfont=dict(size=12, color=text_colors),
        hovertext=hover, hoverinfo="text",
        marker=dict(size=sizes, color=values, colorscale=colorscale, reversescale=reverse,
                    line=dict(color=line_colors, width=line_widths),
                    colorbar=dict(title=cbar_title, thickness=12, len=0.75)),
        showlegend=False,
    ))

    fig.update_layout(
        template=PLOT_TEMPLATE, height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(visible=False, range=X_RANGE, fixedrange=True),
        yaxis=dict(visible=False, range=Y_RANGE, fixedrange=True,
                   scaleanchor="x", scaleratio=1),
        plot_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT),
    )
    return fig


def contribution_figure(rec, cfg, dim_names):
    """Stacked bars: how each neighbour's message contributes to the aggregate."""
    fig = go.Figure()
    if not rec["rows"]:
        return fig
    shades = ["#2a78d6", "#5b9ae4", "#8cbcf0", "#b9d6f7", "#1a5aa8", "#0f3f7a"]
    for i, r in enumerate(rec["rows"]):
        fig.add_trace(go.Bar(
            name=f"from {r['u']}", x=dim_names, y=r["msg"],
            marker_color=shades[i % len(shades)],
            hovertemplate="from %{fullData.name}<br>%{x} = %{y:.3f}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=dim_names, y=rec["agg"], mode="markers+text",
        text=[f"{x:.2f}" for x in rec["agg"]], textposition="middle right",
        marker=dict(symbol="diamond", size=13, color=COLOR_MESSAGE),
        name=f"aggregate ({cfg['agg']})",
        hovertemplate="aggregate %{x} = %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        template=PLOT_TEMPLATE, barmode="relative", height=300, bargap=0.55,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="value", legend=dict(orientation="h", y=1.12, x=0),
    )
    return fig


DIM_SHADES = ["#2a78d6", "#7fb0e8", "#c2d9f5"]   # single-hue ramp: one bar per dimension


def update_stages(rec, cfg):
    """The quantities the update combines, in the order it combines them."""
    act = cfg["act"]
    if cfg["upd"] == "agg only":
        return [("aggregate a", rec["agg"]), (f"{act}(a)", rec["out"])]
    if cfg["upd"] == "self + agg":
        return [("W_self h", rec["self_term"]), ("aggregate a", rec["agg"]),
                ("sum", rec["pre"]), (f"{act}(sum)", rec["out"])]
    return [("h (old)", rec["h_v"]), ("W_self h", rec["self_term"]),
            ("aggregate a", rec["agg"]), ("sum", rec["pre"]),
            (f"h + {act}(sum)", rec["out"])]


def update_figure(rec, cfg, dim_names):
    """Read left to right: each group is one quantity in the update equation, and the
    bars inside a group are the feature dimensions."""
    stages = update_stages(rec, cfg)
    xs = [name for name, _ in stages]
    fig = go.Figure()
    for i, dname in enumerate(dim_names):
        fig.add_trace(go.Bar(
            name=dname, x=xs, y=[vals[i] for _, vals in stages],
            marker_color=DIM_SHADES[i % len(DIM_SHADES)],
            text=[f"{vals[i]:.2f}" for _, vals in stages], textposition="outside",
            textfont=dict(size=10), cliponaxis=False,
            hovertemplate="%{x}<br>" + dname + " = %{y:.3f}<extra></extra>",
        ))
    # everything left of the line is intermediate; the last group is what leaves the layer
    fig.add_vline(x=len(xs) - 1.5, line_width=1, line_dash="dot", line_color="#9aa4ae")
    fig.update_layout(
        template=PLOT_TEMPLATE, barmode="group", height=330,
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis_title="value",
        legend=dict(orientation="h", y=1.16, x=0, title="dimension"),
    )
    return fig


# --------------------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------------------

def ensure_state(dim, edge_dim):
    if "X" not in st.session_state or st.session_state["X"].shape[1] != dim:
        st.session_state["X"] = DEFAULT_X[:, :dim].copy()
    if "E" not in st.session_state or st.session_state["E"].shape[1] != edge_dim:
        st.session_state["E"] = DEFAULT_EDGE_FEAT[:, :edge_dim].copy()
    st.session_state.setdefault("step", 0)
    st.session_state.setdefault("focus", 3)
    st.session_state["focus"] = int(np.clip(st.session_state["focus"], 0, N_NODES - 1))


# --------------------------------------------------------------------------------------
# Sidebar: configuration
# --------------------------------------------------------------------------------------

st.title("Graph Neural Network Explorer")
st.caption("A fixed 7-node graph. Set the node features, then walk one node's update "
           "through gather -> message -> aggregate -> update, one layer at a time.")

with st.sidebar:
    st.header("1. Problem setup")
    mode = st.radio("Features", ["Node features only", "Node + edge features"],
                    help="Edge mode gives every edge its own scalar feature that enters "
                         "the message function.")
    edge_mode = mode.startswith("Node +")
    dim = st.slider("Feature dimension d", 1, 3, 2,
                    help="Numbers per node. Small on purpose: you should be able to "
                         "redo the arithmetic by hand.")
    n_layers = st.slider("Layers", 1, 3, 2,
                         help="Each layer is one hop of information flow.")
    self_loops = st.checkbox("Include self-loop in N(v)", value=False,
                             help="If on, a node sends a message to itself as well.")

    st.header("2. Layer definition")
    preset_name = st.selectbox("Preset", list(PRESETS.keys()), index=0)
    preset = PRESETS[preset_name]
    locked = preset is not None

    msg_fn = st.selectbox("Message function", MSG_OPTIONS,
                          index=MSG_OPTIONS.index(preset["msg"]) if locked else 0,
                          disabled=locked)
    agg_fn = st.selectbox("Aggregation", AGG_OPTIONS,
                          index=AGG_OPTIONS.index(preset["agg"]) if locked else 0,
                          disabled=locked)
    upd_fn = st.selectbox("Update", UPD_OPTIONS,
                          index=UPD_OPTIONS.index(preset["upd"]) if locked else 1,
                          disabled=locked)
    act_fn = st.selectbox("Activation", ACT_OPTIONS,
                          index=ACT_OPTIONS.index(preset["act"]) if locked else 0,
                          disabled=locked)
    edge_dim = st.slider("Edge feature dimension d_e", 1, MAX_EDGE_DIM, 1,
                         disabled=not edge_mode,
                         help="Numbers carried by each edge. `concat` and `gate` map them "
                              "into message space with a matrix, so any width works.")
    # `scale` multiplies elementwise, so it needs one shared number or one per dimension
    mech_options = [m for m in EDGE_OPTIONS
                    if m != "scale" or edge_dim == 1 or edge_dim == dim]
    edge_mech = st.selectbox("Edge-feature mechanism", mech_options, index=0,
                             disabled=not edge_mode,
                             help="scale: m = e * (W h_u), elementwise.  "
                                  "concat: m = W h_u + B_e e.  "
                                  "gate: m = sigmoid(W_g e) * (W h_u).")
    if edge_mode and "scale" not in mech_options:
        st.caption(f"`scale` is hidden: it multiplies elementwise, so it needs "
                   f"d_e = 1 or d_e = d (= {dim}).")

    st.header("3. Weights")
    w_kind = st.radio("Weight matrices", ["identity", "random (seeded)"], index=0,
                      help="Identity keeps the arithmetic checkable by hand. "
                           "Random shows what a trained-looking layer does.")
    w_seed = st.number_input("Seed", 0, 9999, 0, disabled=(w_kind == "identity"))

    st.divider()
    if st.button("Reset features to defaults", width="stretch"):
        st.session_state["X"] = DEFAULT_X[:, :dim].copy()
        st.session_state["E"] = DEFAULT_EDGE_FEAT[:, :edge_dim].copy()

cfg = dict(msg=msg_fn, agg=agg_fn, upd=upd_fn, act=act_fn, n_layers=n_layers,
           self_loops=self_loops, edge_mode=edge_mode, edge_mech=edge_mech)

ensure_state(dim, edge_dim)
wts = make_weights(dim, "identity" if w_kind == "identity" else "random", int(w_seed),
                   edge_dim)
E_NAMES = [f"e{i}" for i in range(edge_dim)]
DIM_NAMES = [f"h{i}" for i in range(dim)]

if st.session_state["step"] >= 4 * n_layers:
    st.session_state["step"] = 4 * n_layers - 1

tab_setup, tab_step, tab_layers, tab_math = st.tabs(
    ["1. Graph & features", "2. Step through a node", "3. All layers", "4. What the layer computes"]
)

# --------------------------------------------------------------------------------------
# Tab 1: graph and editable features
# --------------------------------------------------------------------------------------

with tab_setup:
    # Narrow numeric columns keep the tables and the graph on one row, no scrolling.
    table_h = 35 * (N_NODES + 1) + 3
    # tall enough that the drawing is limited by the column width, not the height, so it
    # fills the box instead of sitting letterboxed between two margins
    GRAPH_H = 260
    if edge_mode:
        c_nodes, c_edges, c_graph = st.columns([1.0, 1.0, 2.2], gap="medium")
    else:
        c_nodes, c_graph = st.columns([0.95, 2.4], gap="medium")
        c_edges = None

    with c_nodes:
        st.markdown("**Node features**")
        st.caption("Edit any value.")
        df = pd.DataFrame(st.session_state["X"], columns=DIM_NAMES)
        df.insert(0, "node", np.arange(N_NODES))
        edited = st.data_editor(
            df, key=f"node_editor_{dim}", hide_index=True, height=table_h,
            width="stretch", disabled=["node"],
            column_config={
                "node": st.column_config.NumberColumn("node", width="small"),
                **{c: st.column_config.NumberColumn(c, format="%.2f", step=0.1,
                                                    width="small") for c in DIM_NAMES},
            },
        )
        st.session_state["X"] = edited[DIM_NAMES].to_numpy(dtype=float)

    if edge_mode:
        with c_edges:
            st.markdown("**Edge features**")
            st.caption("One scalar per edge.")
            edf = pd.DataFrame(st.session_state["E"], columns=E_NAMES)
            edf.insert(0, "edge", [f"{a}-{b}" for a, b in EDGES])
            eedit = st.data_editor(
                edf, key=f"edge_editor_{edge_dim}", hide_index=True,
                height=35 * (len(EDGES) + 1) + 3,
                width="stretch", disabled=["edge"],
                column_config={
                    "edge": st.column_config.TextColumn("edge", width="small"),
                    **{c: st.column_config.NumberColumn(c, format="%.2f", step=0.1,
                                                        width="small") for c in E_NAMES},
                },
            )
            st.session_state["E"] = eedit[E_NAMES].to_numpy(dtype=float)

            with st.popover("How edge features enter a GNN layer", width="stretch"):
                st.markdown(
                    "`scale`, `concat` and `gate` are this app's labels rather than "
                    "standard names, but each matches real operators, and they differ in "
                    "**where** the edge acts. Two of them change *how much* of a message "
                    "arrives: `scale` is a plain weighted adjacency, which is all a "
                    "spectral operator such as GCN can accept (hence `edge_weight`, never "
                    "`edge_attr`), and `gate` is the same idea with a learned per-channel "
                    "sigmoid, as in CGConv and GatedGCN. `concat` instead changes *what* "
                    "is sent, adding the edge's own projected vector into the message - "
                    "exactly GINEConv, and the same additive projection that GAT and "
                    "TransformerConv use for their `edge_dim` argument. A fourth family, "
                    "left out here, is edge-conditioned weights (NNConv, the original "
                    "MPNN, and RGCN for discrete relation types), where each edge "
                    "generates its own weight matrix: the most general and by far the "
                    "most expensive. Most classic operators support none of this, partly "
                    "because their benchmarks - citation and social graphs - have edges "
                    "carrying nothing but existence, and partly for a real structural "
                    "reason: a spectral derivation is built on a scalar-weighted "
                    "Laplacian, where a vector per edge simply has no place. Edge support "
                    "therefore tracks the domain rather than the era, and is routine for "
                    "molecules, knowledge graphs, road networks and physics simulation. "
                    "One simplification here is that GatedGCN, MPNN and graph transformers "
                    "give edges their own hidden state, updated every layer, whereas this "
                    "app holds them fixed."
                )

    with c_graph:
        gcols = st.columns([1, 1, 1.4])
        with gcols[0]:
            color_dim = st.selectbox("Colour nodes by", DIM_NAMES, index=0,
                                     key=f"setup_dim_{dim}")
        edge_label_dim = 0
        if edge_mode and edge_dim > 1:
            with gcols[1]:
                edge_label_dim = E_NAMES.index(
                    st.selectbox("Label edges with", E_NAMES, index=0,
                                 key=f"setup_edim_{edge_dim}"))
        ci = DIM_NAMES.index(color_dim)
        labels = [f"{v}<br>{st.session_state['X'][v, ci]:.2f}" for v in range(N_NODES)]
        elabels = None
        if edge_mode:
            elabels = {(a, b): f"{st.session_state['E'][i, edge_label_dim]:.2f}"
                       for i, (a, b) in enumerate(EDGES)}
        st.plotly_chart(
            graph_figure(st.session_state["X"][:, ci], cbar_title=color_dim,
                         node_text=labels, fade=False, edge_labels=elabels,
                         height=GRAPH_H),
            width="stretch",
        )
        deg = np.array([len(n) for n in neighbor_lists(self_loops)])
        st.caption("Degrees " + ("(with self-loop): " if self_loops else ": ")
                   + ", ".join(f"{v}:{deg[v]}" for v in range(N_NODES))
                   + ("  |  edge features default to random values; set them all to 1 "
                      "to recover the node-features-only case." if edge_mode else ""))

X = st.session_state["X"]
E = st.session_state["E"]
layers = run_forward(X, cfg, wts, E)

# --------------------------------------------------------------------------------------
# Tab 2: the stepper
# --------------------------------------------------------------------------------------

with tab_step:
    c_focus, c_back, c_next, c_reset, c_label = st.columns([1.3, 0.8, 0.8, 0.9, 3])
    with c_focus:
        st.session_state["focus"] = st.selectbox(
            "Focus node", list(range(N_NODES)),
            index=st.session_state["focus"], key="focus_select",
        )
    # Moving the step in an on_click callback (rather than after st.button returns)
    # keeps the disabled flags honest: the callback runs before the rerun draws the
    # buttons, so the index can never walk outside [0, 4 * n_layers).
    last_step = 4 * n_layers - 1

    def move_step(delta):
        st.session_state["step"] = int(np.clip(st.session_state["step"] + delta, 0, last_step))

    def restart_steps():
        st.session_state["step"] = 0

    with c_back:
        st.write("")
        st.button("Back", width="stretch", on_click=move_step, args=(-1,),
                  disabled=st.session_state["step"] == 0)
    with c_next:
        st.write("")
        st.button("Next", width="stretch", on_click=move_step, args=(1,),
                  disabled=st.session_state["step"] >= last_step)
    with c_reset:
        st.write("")
        st.button("Restart", width="stretch", on_click=restart_steps)

    step = st.session_state["step"]
    layer_i, stage = step // 4, step % 4
    v = st.session_state["focus"]
    rec = layers[layer_i]["nodes"][v]
    H_in = layers[layer_i]["H_in"]

    with c_label:
        st.write("")
        st.markdown(f"**Layer {layer_i + 1} of {n_layers}  |  {STAGE_NAMES[stage]}**  "
                    f"&nbsp;&nbsp; node **{v}**, {len(rec['nbrs'])} neighbour(s)")
    st.progress((step + 1) / (4 * n_layers))

    gcol, dcol = st.columns([1.55, 1], gap="large")

    # ---- the picture ----
    with gcol:
        show_dim = st.selectbox("Colour nodes by", DIM_NAMES, index=0, key=f"step_dim_{dim}")
        di = DIM_NAMES.index(show_dim)
        vals = H_in[:, di]
        labels = [f"{n}<br>{vals[n]:.2f}" for n in range(N_NODES)]
        arrows = []
        if stage >= 1 and rec["rows"]:
            mags = np.array([float(np.linalg.norm(r["msg"])) for r in rec["rows"]])
            scale = mags.max() if mags.max() > 1e-9 else 1.0
            for r, m in zip(rec["rows"], mags):
                if r["u"] == v:
                    continue
                arrows.append(dict(u=r["u"], v=v, w=1.2 + 2.8 * m / scale,
                                   color=COLOR_MESSAGE))
        if stage == 3:
            labels = [f"{n}<br>{vals[n]:.2f}" for n in range(N_NODES)]
            labels[v] = f"{v}<br>{vals[v]:.2f} -> {rec['out'][di]:.2f}"
        st.plotly_chart(
            graph_figure(vals, focus=v, highlight=[r["u"] for r in rec["rows"]],
                         arrows=arrows, cbar_title=f"{show_dim} (layer {layer_i} input)",
                         node_text=labels, height=300),
            width="stretch",
        )
        if stage >= 1 and rec["rows"]:
            st.caption("Arrow thickness is proportional to the length of the message "
                       "travelling along that edge.")

    # ---- the numbers ----
    with dcol:
        if stage == 0:
            st.markdown("#### Gather")
            st.markdown(
                f"Node **{v}** looks at its neighbourhood "
                f"$N({v}) = \\{{{', '.join(str(u) for u in rec['nbrs'])}\\}}$"
                + ("  (self-loop included)" if self_loops else "")
            )
            g = pd.DataFrame({
                "neighbour u": [r["u"] for r in rec["rows"]],
                "h_u (layer input)": [fmt_vec(r["h_u"]) for r in rec["rows"]],
            })
            if edge_mode:
                g["e(u,v)"] = [fmt_vec(edge_feature(r["u"], v, E), 2) for r in rec["rows"]]
            st.dataframe(g, hide_index=True, width="stretch")
            st.info(f"Own state  h_{v} = {fmt_vec(rec['h_v'])}  (not used yet at this stage)")
            st.caption("Nothing has been computed yet. This stage is only about scope: "
                       "which nodes are allowed to speak to node "
                       f"{v} in this layer.")

        elif stage == 1:
            st.markdown("#### Message")
            st.markdown("Every neighbour sends its **own vector**, not a number:")
            if cfg["msg"] == MSG_DIFF:
                st.latex(r"m_{u \to v} = c_{uv}\, W\,(h_u - h_v)")
            else:
                st.latex(r"m_{u \to v} = c_{uv}\, W h_u")
            cols = {
                "u": [r["u"] for r in rec["rows"]],
                "W h_u": [fmt_vec(r["transformed"]) for r in rec["rows"]],
            }
            if edge_mode:
                cols["e"] = [fmt_vec(r["e_val"], 2) for r in rec["rows"]]
                cols[f"after edge ({edge_mech})"] = [fmt_vec(r["after_edge"]) for r in rec["rows"]]
            if cfg["msg"] == MSG_ATTN:
                cols["score"] = [f"{r['score']:.3f}" for r in rec["rows"]]
            cols["coefficient c"] = [f"{r['coef']:.3f}" for r in rec["rows"]]
            cols["message m"] = [fmt_vec(r["msg"]) for r in rec["rows"]]
            st.dataframe(pd.DataFrame(cols), hide_index=True, width="stretch")
            if cfg["msg"] == MSG_GCN:
                st.caption("c = 1/sqrt(deg(u) deg(v)): high-degree neighbours are damped, "
                           "so a hub does not drown out everyone else.")
            elif cfg["msg"] == MSG_ATTN:
                st.caption("c = softmax over the neighbourhood, so the coefficients sum "
                           "to 1 and are *learned* from the features rather than fixed "
                           "by the degrees.")
            elif cfg["msg"] == MSG_DIFF:
                st.caption("Messages carry the *difference* to the centre node, so a "
                           "neighbourhood that already agrees with v sends nearly nothing.")
            else:
                st.caption("c = 1: every neighbour speaks at full volume, so the result "
                           "grows with degree.")

        elif stage == 2:
            st.markdown("#### Aggregate")
            st.latex({
                "sum": r"a_v = \sum_{u \in N(v)} m_{u \to v}",
                "mean": r"a_v = \frac{1}{|N(v)|}\sum_{u \in N(v)} m_{u \to v}",
                "max": r"a_v = \max_{u \in N(v)} m_{u \to v} \quad\text{(elementwise)}",
            }[cfg["agg"]])
            st.plotly_chart(contribution_figure(rec, cfg, DIM_NAMES),
                            width="stretch")
            st.success(f"aggregate a_{v} = {fmt_vec(rec['agg'])}")
            if cfg["agg"] == "max" and rec["winners"]:
                st.caption("Winning neighbour per dimension: " +
                           ", ".join(f"{DIM_NAMES[i]} <- node {u}"
                                     for i, u in enumerate(rec["winners"])))
            st.caption("The messages arrive as an unordered pile. Shuffling them changes "
                       "nothing, which is exactly why a GNN can be applied to a graph "
                       "with no canonical node ordering.")
            if st.button("Shuffle the neighbour order", key="shuffle"):
                order = np.random.permutation(len(rec["rows"]))
                st.write("Shuffled order: " +
                         " , ".join(f"m from {rec['rows'][i]['u']}" for i in order))
                shuffled = np.array([rec["rows"][int(i)]["msg"] for i in order])
                again = (shuffled.sum(0) if cfg["agg"] == "sum" else
                         shuffled.mean(0) if cfg["agg"] == "mean" else shuffled.max(0))
                st.write(f"Aggregate is still {fmt_vec(again)}")

        else:
            st.markdown("#### Update")
            st.latex({
                "agg only": r"h_v' = \sigma\!\left(a_v\right)",
                "self + agg": r"h_v' = \sigma\!\left(W_{self} h_v + a_v\right)",
                "residual": r"h_v' = h_v + \sigma\!\left(W_{self} h_v + a_v\right)",
            }[cfg["upd"]])
            st.plotly_chart(update_figure(rec, cfg, DIM_NAMES), width="stretch")
            st.caption({
                "agg only": "Left group: the aggregate. Right group: the same numbers "
                            f"after {cfg['act']}. Bars within a group are the feature "
                            "dimensions, so this is the whole update in two steps.",
                "self + agg": "Read the groups left to right: the node's own transformed "
                              "state, the aggregate from its neighbours, the two added "
                              "together, and that sum after "
                              f"{cfg['act']}. Groups 1 + 2 = group 3. Bars within a group "
                              "are the feature dimensions.",
                "residual": "Read left to right: the old embedding, the node's own "
                            "transformed state, the aggregate, those last two added, and "
                            "finally the old embedding plus "
                            f"{cfg['act']}(sum). Bars within a group are the feature "
                            "dimensions.",
            }[cfg["upd"]] + " Only the group right of the dotted line leaves this layer.")
            box = [f"h_{v} (in)        = {fmt_vec(rec['h_v'])}"]
            if cfg["upd"] != "agg only":
                box.append(f"W_self h_{v}     = {fmt_vec(rec['self_term'])}")
            box += [
                f"aggregate a_{v}  = {fmt_vec(rec['agg'])}",
                f"pre-activation  = {fmt_vec(rec['pre'])}",
                f"h_{v} (out)       = {fmt_vec(rec['out'])}",
            ]
            st.code("\n".join(box), language="text")
            if layer_i + 1 < n_layers:
                st.caption(f"This output becomes the input of layer {layer_i + 2}. "
                           "Press Next to keep going.")
            else:
                st.caption("Last layer: this is node "
                           f"{v}'s final embedding. It has now seen everything within "
                           f"{n_layers} hop(s).")

# --------------------------------------------------------------------------------------
# Tab 3: all nodes, all layers
# --------------------------------------------------------------------------------------

with tab_layers:
    st.subheader("Every node, every layer")
    di = DIM_NAMES.index(st.selectbox("Dimension", DIM_NAMES, index=0, key=f"ov_dim_{dim}"))

    stages = [X] + [lay["H_out"] for lay in layers]
    grid = np.array([[stages[L][n, di] for L in range(len(stages))] for n in range(N_NODES)])

    hcol, scol = st.columns([1.2, 1], gap="large")
    with hcol:
        fig = go.Figure(go.Heatmap(
            z=grid, x=["input"] + [f"layer {i + 1}" for i in range(n_layers)],
            y=[f"node {n}" for n in range(N_NODES)],
            colorscale=SEQ_FEATURE, reversescale=True,
            colorbar=dict(title=DIM_NAMES[di], thickness=12),
            hovertemplate="%{y} at %{x}: %{z:.3f}<extra></extra>",
        ))
        fig.update_layout(template=PLOT_TEMPLATE, height=400,
                          margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("Columns drift towards a common value as depth grows. That is "
                   "over-smoothing, and it is why most GNNs are 2-3 layers deep.")

    with scol:
        spread = [mean_pairwise_distance(s) for s in stages]
        sfig = go.Figure(go.Scatter(
            x=list(range(len(spread))), y=spread, mode="lines+markers",
            line=dict(color=COLOR_MESSAGE, width=2.5), marker=dict(size=9),
            hovertemplate="after %{x} layer(s): %{y:.3f}<extra></extra>",
        ))
        sfig.update_layout(template=PLOT_TEMPLATE, height=400,
                           margin=dict(l=10, r=10, t=30, b=10),
                           xaxis_title="layers applied",
                           yaxis_title="mean pairwise distance between embeddings")
        st.plotly_chart(sfig, width="stretch")
        st.caption("If this curve falls towards zero, every node is converging to the "
                   "same embedding and node-level predictions become impossible.")

    st.divider()
    st.subheader(f"Receptive field of node {st.session_state['focus']}")
    rcol, tcol = st.columns([1.3, 1], gap="large")
    dist = hop_distances(st.session_state["focus"])
    with rcol:
        capped = np.where(dist < 0, 99, dist).astype(float)
        labels = [f"{n}<br>{int(dist[n])} hop" if dist[n] >= 0 else f"{n}<br>-"
                  for n in range(N_NODES)]
        st.plotly_chart(
            graph_figure(capped, focus=st.session_state["focus"], colorscale=SEQ_HOPS,
                         reverse=True, cbar_title="hops", node_text=labels, fade=False,
                         height=300),
            width="stretch",
        )
    with tcol:
        inside = [n for n in range(N_NODES) if 0 <= dist[n] <= n_layers]
        outside = [n for n in range(N_NODES) if dist[n] > n_layers or dist[n] < 0]
        st.markdown(f"With **{n_layers}** layer(s), node "
                    f"**{st.session_state['focus']}** can see:")
        st.success("Inside the receptive field: " + ", ".join(map(str, inside)))
        if outside:
            st.warning("Never reached: " + ", ".join(map(str, outside)))
            st.caption("Those nodes could change arbitrarily and this node's embedding "
                       "would not move. That is under-reaching - the mirror image of "
                       "over-smoothing, and the reason depth is a real trade-off.")
        else:
            st.caption("Every node is within reach. Add fewer layers to see "
                       "under-reaching.")

# --------------------------------------------------------------------------------------
# Tab 4: the layer written out
# --------------------------------------------------------------------------------------

with tab_math:
    st.subheader("The layer you have configured")
    mcol, wcol = st.columns([1.2, 1], gap="large")

    with mcol:
        st.markdown("**Message**")
        base = r"W h_u" if cfg["msg"] != MSG_DIFF else r"W (h_u - h_v)"
        if edge_mode:
            body = {
                "scale": r"e_{uv} \odot " + base,
                "concat": base + r" + B_e\, e_{uv}",
                "gate": r"\sigma(W_g\, e_{uv}) \odot " + base,
            }[edge_mech]
        else:
            body = base
        coef = {
            MSG_LINEAR: r"1",
            MSG_GCN: r"\frac{1}{\sqrt{d_u d_v}}",
            MSG_ATTN: r"\alpha_{uv}",
            MSG_DIFF: r"1",
        }[cfg["msg"]]
        st.latex(r"m_{u \to v} = " + coef + r"\cdot\left(" + body + r"\right)")
        if cfg["msg"] == MSG_ATTN:
            st.latex(r"\alpha_{uv} = \mathrm{softmax}_{u \in N(v)}"
                     r"\big(\mathrm{LeakyReLU}(a^\top [W h_v \,\|\, W h_u])\big)")

        st.markdown("**Aggregate**")
        st.latex({
            "sum": r"a_v = \sum_{u \in N(v)} m_{u \to v}",
            "mean": r"a_v = \tfrac{1}{|N(v)|}\sum_{u \in N(v)} m_{u \to v}",
            "max": r"a_v = \max_{u \in N(v)} m_{u \to v}",
        }[cfg["agg"]])

        st.markdown("**Update**")
        st.latex({
            "agg only": r"h_v^{(l+1)} = \sigma(a_v)",
            "self + agg": r"h_v^{(l+1)} = \sigma(W_{self} h_v^{(l)} + a_v)",
            "residual": r"h_v^{(l+1)} = h_v^{(l)} + \sigma(W_{self} h_v^{(l)} + a_v)",
        }[cfg["upd"]])
        st.caption(f"with sigma = {cfg['act']}, applied for {n_layers} layer(s), "
                   f"N(v) {'including' if self_loops else 'excluding'} v itself"
                   + (f", and edge features in R^{edge_dim}." if edge_mode else "."))
        if edge_mode and edge_mech == "concat":
            st.caption("Writing it as a concatenation is the same thing: [W | B_e] "
                       "applied to [h_u ; e_uv] expands to exactly this sum.")

    with wcol:
        st.markdown("**Weights in use**")
        extra = ""
        if edge_mode and edge_mech == "concat":
            extra = f"\n\nB_e     (edge -> message, {dim}x{edge_dim})\n{fmt_mat(wts['B_edge'])}"
        elif edge_mode and edge_mech == "gate":
            extra = f"\n\nW_g     (edge -> gate, {dim}x{edge_dim})\n{fmt_mat(wts['W_gate'])}"
        st.code(f"W       (message)\n{fmt_mat(wts['W'])}\n\n"
                f"W_self  (update)\n{fmt_mat(wts['W_self'])}" + extra,
                language="text")
        if w_kind == "identity":
            st.caption("Identity weights: the layer does no mixing of dimensions, so "
                       "every number on screen can be checked by hand. Switch to random "
                       "weights to see what a trained layer would do.")

    st.divider()
    st.markdown("**Things worth trying**")
    st.markdown(
        "- Set every node feature to `1.0` and pick *linear + sum*. The aggregate is now "
        "literally each node's **degree** - that is where graph structure enters the maths.\n"
        "- Give nodes 3 and 5 the same feature vector. Node 4 now sees the multiset "
        "`{A, A}` and node 6 sees `{A}`. Under *mean* their aggregates are identical; "
        "under *sum* they differ by a factor of two. That single example is the whole "
        "expressivity argument behind GIN.\n"
        "- Push to 3 layers with *mean* + *agg only* and watch the spread curve in tab 3 "
        "fall. Switch the update to *residual* and it stops falling, because each layer "
        "now adds to the previous embedding instead of replacing it.\n"
        "- In edge mode with *scale*, set the bridge edge `2-3` to `0`. It goes mute and you "
        "have split the graph in two without touching the topology.\n"
        "- Raise the edge dimension to 3 with *concat*: every edge now injects its own "
        "vector B_e e into each message it carries, on top of what the sender said.\n"
        "- Compare *gcn* and *attention* on node 2 (degree 3) versus node 6 (degree 1) - "
        "fixed normalisation versus coefficients that depend on the features."
    )
