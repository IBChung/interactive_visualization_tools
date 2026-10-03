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
Interactive Optimization Algorithms Explorer

Step through derivative-based and nature-inspired optimization methods on 1D and
2D test functions, one iteration (or generation) at a time.

Run with:
    streamlit run interactive_optimization_algorithms.py
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Optimization Algorithms Explorer", layout="wide")

# --------------------------------------------------------------------------------------
# Palette (validated: references/palette.md - categorical order). Dark mode is the
# assumed default background; a viewer who switches the Streamlit theme is detected
# and respected, but an undetectable theme falls back to dark rather than light.
# --------------------------------------------------------------------------------------

def _dark_theme():
    for probe in (lambda: st.context.theme.type, lambda: st.get_option("theme.base")):
        try:
            value = probe()
        except Exception:
            continue
        if value:
            return str(value).lower() == "dark"
    return True


DARK = _dark_theme()
PLOT_TEMPLATE = "plotly_dark" if DARK else "plotly_white"
TEXT = "#fafafa" if DARK else "#2b3138"
TEXT_MUTED = "#b6bec7" if DARK else "#4a5560"

COLOR_POINT = "#eb6834"      # categorical slot 2 (orange) - current position / particle
COLOR_TRAJ = "#2a78d6"       # categorical slot 1 (blue)   - path taken / parent 1 / x_r2
COLOR_GRAD = "#1baf7a"       # categorical slot 3 (aqua)   - gradient step / mutant / result
COLOR_BEST = "#eda100"       # categorical slot 4 (yellow) - best-so-far / parent 2 / x_r3
COLOR_GHOST = "rgba(42,120,214,0.35)"
COLOR_TARGET = "#ffffff"
SEQ_FIT = "Blues"
SEQ_LANDSCAPE = "Greys"       # background only - kept hue-neutral so every bold
                              # categorical colour above stays legible on top of it

# --------------------------------------------------------------------------------------
# Safe expression evaluation for custom test functions
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
# Test functions
# --------------------------------------------------------------------------------------

def quadratic_bowl(x):
    return (x - 2.0) ** 2 + 1.0


def double_well(x):
    return 0.05 * x ** 4 - x ** 2 + 0.5 * x + 5.0


def rugged_1d(x):
    return x ** 2 / 20.0 + 2.0 * np.sin(3.0 * x)


def sphere2d(x1, x2):
    return x1 ** 2 + x2 ** 2


def rosenbrock2d(x1, x2):
    return (1.0 - x1) ** 2 + 100.0 * (x2 - x1 ** 2) ** 2


def rastrigin2d(x1, x2):
    return 20.0 + (x1 ** 2 - 10.0 * np.cos(2 * np.pi * x1)) \
                + (x2 ** 2 - 10.0 * np.cos(2 * np.pi * x2))


FUNCS_1D = {
    "Quadratic bowl (convex)": (-5.0, 5.0),
    "Double well (two minima)": (-4.0, 4.0),
    "Rugged (many local minima)": (-6.0, 6.0),
    "Custom expression": (-5.0, 5.0),
}

FUNCS_2D = {
    "Sphere (convex bowl)": ((-5.0, 5.0), (-5.0, 5.0)),
    "Rosenbrock (banana valley)": ((-2.0, 2.0), (-1.0, 3.0)),
    "Rastrigin (many local minima)": ((-5.12, 5.12), (-5.12, 5.12)),
    "Custom expression": ((-5.0, 5.0), (-5.0, 5.0)),
}

FUNC_LATEX_1D = {
    "Quadratic bowl (convex)": r"f(x) = (x-2)^2+1",
    "Double well (two minima)": r"f(x)=0.05x^4-x^2+0.5x+5",
    "Rugged (many local minima)": r"f(x)=\tfrac{x^2}{20}+2\sin(3x)",
}

FUNC_LATEX_2D = {
    "Sphere (convex bowl)": r"f(x_1,x_2)=x_1^2+x_2^2",
    "Rosenbrock (banana valley)": r"f(x_1,x_2)=(1-x_1)^2+100(x_2-x_1^2)^2",
    "Rastrigin (many local minima)": r"f(x_1,x_2)=20+\sum_{i=1}^{2}\left[x_i^2-10\cos(2\pi x_i)\right]",
}


def make_truth_fn(dim, name, custom_expr=None):
    if dim == 1:
        if name == "Quadratic bowl (convex)":
            return lambda X: quadratic_bowl(X[:, 0])
        if name == "Double well (two minima)":
            return lambda X: double_well(X[:, 0])
        if name == "Rugged (many local minima)":
            return lambda X: rugged_1d(X[:, 0])
        return lambda X: safe_eval(custom_expr, x=X[:, 0])
    else:
        if name == "Sphere (convex bowl)":
            return lambda X: sphere2d(X[:, 0], X[:, 1])
        if name == "Rosenbrock (banana valley)":
            return lambda X: rosenbrock2d(X[:, 0], X[:, 1])
        if name == "Rastrigin (many local minima)":
            return lambda X: rastrigin2d(X[:, 0], X[:, 1])
        return lambda X: safe_eval(custom_expr, x1=X[:, 0], x2=X[:, 1])


def build_2d_grid(bounds, n=100):
    x1 = np.linspace(bounds[0][0], bounds[0][1], n)
    x2 = np.linspace(bounds[1][0], bounds[1][1], n)
    X1, X2 = np.meshgrid(x1, x2)
    grid = np.column_stack([X1.ravel(), X2.ravel()])
    return x1, x2, X1, X2, grid


def init_population(bounds, n, seed):
    rng = np.random.default_rng(seed)
    d = len(bounds)
    u = rng.uniform(size=(n, d))
    for i, (lo, hi) in enumerate(bounds):
        u[:, i] = lo + u[:, i] * (hi - lo)
    return u


# --------------------------------------------------------------------------------------
# Numerical differentiation (central differences) - used by all derivative-based
# methods, including custom expressions, so no hand-derived gradient is needed.
# --------------------------------------------------------------------------------------

def numgrad(f_point, x, h=1e-4):
    d = len(x)
    g = np.zeros(d)
    for i in range(d):
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        g[i] = (f_point(xp) - f_point(xm)) / (2 * h)
    return g


def numhess(f_point, x, h=1e-3):
    d = len(x)
    H = np.zeros((d, d))
    fx = f_point(x)
    for i in range(d):
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        H[i, i] = (f_point(xp) - 2 * fx + f_point(xm)) / h ** 2
    for i in range(d):
        for j in range(i + 1, d):
            xpp, xpm, xmp, xmm = x.copy(), x.copy(), x.copy(), x.copy()
            xpp[i] += h; xpp[j] += h
            xpm[i] += h; xpm[j] -= h
            xmp[i] -= h; xmp[j] += h
            xmm[i] -= h; xmm[j] -= h
            val = (f_point(xpp) - f_point(xpm) - f_point(xmp) + f_point(xmm)) / (4 * h ** 2)
            H[i, j] = H[j, i] = val
    return H


# --------------------------------------------------------------------------------------
# Derivative-based algorithms. Each returns a list of step dicts; a "step" is one
# point in the trajectory (step 0 is the starting point).
# --------------------------------------------------------------------------------------

GRAD_TOL = 1e-6


def _diverged(x, bounds, factor=4.0):
    for xi, (lo, hi) in zip(x, bounds):
        span = hi - lo
        if not np.isfinite(xi) or xi < lo - factor * span or xi > hi + factor * span:
            return True
    return False


def run_gd(f_point, x0, bounds, lr, max_iter):
    x = np.array(x0, dtype=float)
    g = numgrad(f_point, x)
    steps = [dict(x=x.copy(), fx=float(f_point(x)), grad=g.copy(), note="")]
    for _ in range(max_iter):
        if np.linalg.norm(g) < GRAD_TOL:
            break
        x = x - lr * g
        if _diverged(x, bounds):
            steps.append(dict(x=x.copy(), fx=np.nan, grad=g.copy(), note="diverged"))
            break
        g = numgrad(f_point, x)
        steps.append(dict(x=x.copy(), fx=float(f_point(x)), grad=g.copy(), note=""))
    return steps


def run_newton(f_point, x0, bounds, alpha, damping, max_iter):
    x = np.array(x0, dtype=float)
    d = len(x)
    span = np.array([hi - lo for lo, hi in bounds])
    g = numgrad(f_point, x)
    H = numhess(f_point, x)
    note = "indefinite Hessian - step may not descend" if np.any(np.linalg.eigvalsh((H + H.T) / 2) < 0) else ""
    steps = [dict(x=x.copy(), fx=float(f_point(x)), grad=g.copy(), hess=H.copy(), note=note)]
    for _ in range(max_iter):
        if np.linalg.norm(g) < GRAD_TOL:
            break
        H_reg = H + damping * np.eye(d)
        try:
            step = np.linalg.solve(H_reg, g)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(H_reg) @ g
        step = np.clip(alpha * step, -2 * span, 2 * span)
        x = x - step
        if _diverged(x, bounds):
            steps.append(dict(x=x.copy(), fx=np.nan, grad=g.copy(), hess=H.copy(), note="diverged"))
            break
        g = numgrad(f_point, x)
        H = numhess(f_point, x)
        note = "indefinite Hessian - step may not descend" if np.any(np.linalg.eigvalsh((H + H.T) / 2) < 0) else ""
        steps.append(dict(x=x.copy(), fx=float(f_point(x)), grad=g.copy(), hess=H.copy(), note=note))
    return steps


def run_adam(f_point, x0, bounds, lr, beta1, beta2, eps, max_iter):
    x = np.array(x0, dtype=float)
    d = len(x)
    m, v = np.zeros(d), np.zeros(d)
    g = numgrad(f_point, x)
    steps = [dict(x=x.copy(), fx=float(f_point(x)), grad=g.copy(), m=m.copy(), v=v.copy(), note="")]
    for k in range(1, max_iter + 1):
        if np.linalg.norm(g) < GRAD_TOL:
            break
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        mhat = m / (1 - beta1 ** k)
        vhat = v / (1 - beta2 ** k)
        x = x - lr * mhat / (np.sqrt(vhat) + eps)
        if _diverged(x, bounds):
            steps.append(dict(x=x.copy(), fx=np.nan, grad=g.copy(), m=m.copy(), v=v.copy(), note="diverged"))
            break
        g = numgrad(f_point, x)
        steps.append(dict(x=x.copy(), fx=float(f_point(x)), grad=g.copy(), m=m.copy(), v=v.copy(), note=""))
    return steps


DERIV_EQUATIONS = {
    "Gradient Descent": [r"x_{k+1} = x_k - \eta\,\nabla f(x_k)"],
    "Newton-Raphson": [r"x_{k+1} = x_k - \alpha\,[\mathbf{H}f(x_k)+\lambda I]^{-1}\nabla f(x_k)"],
    "Adam": [
        r"m_{k+1}=\beta_1 m_k+(1-\beta_1)\nabla f(x_k)",
        r"v_{k+1}=\beta_2 v_k+(1-\beta_2)\nabla f(x_k)^2",
        r"x_{k+1}=x_k-\eta\,\dfrac{\hat m_{k+1}}{\sqrt{\hat v_{k+1}}+\epsilon}",
    ],
}


# --------------------------------------------------------------------------------------
# Nature-inspired algorithms. Each returns a "history" list, one entry per
# generation (entry 0 is the initial population). Every entry records the
# population, its fitness, the running best, and - from generation 1 onward -
# one "event" per individual describing exactly how it was produced, which the
# plot uses to draw the update mechanism for a chosen individual.
# --------------------------------------------------------------------------------------

def run_ga(truth_fn, bounds, pop_size, cx_rate, mut_rate, mut_sigma, elite_count,
           tournament_k, max_gen, seed):
    rng = np.random.default_rng(seed)
    d = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    pop = lo + rng.uniform(size=(pop_size, d)) * (hi - lo)
    fit = truth_fn(pop)
    best_i = int(np.argmin(fit))
    best_val, best_pos = float(fit[best_i]), pop[best_i].copy()
    history = [dict(pop=pop.copy(), fit=fit.copy(), best_val=best_val, best_pos=best_pos.copy(),
                     mean_val=float(fit.mean()), events=None)]

    for _ in range(max_gen):
        order = np.argsort(fit)
        elite_idx = order[:elite_count]
        children, events = [], []
        for k in range(pop_size):
            if k < elite_count:
                child = pop[int(elite_idx[k])].copy()
                events.append(dict(elite=True, post_mut=child.copy()))
                children.append(child)
                continue
            cand1 = rng.integers(0, pop_size, size=tournament_k)
            cand2 = rng.integers(0, pop_size, size=tournament_k)
            p1 = int(cand1[np.argmin(fit[cand1])])
            p2 = int(cand2[np.argmin(fit[cand2])])
            P1, P2 = pop[p1], pop[p2]
            if rng.random() < cx_rate:
                alpha = rng.uniform(-0.25, 1.25, size=d)
                pre_mut = P1 + alpha * (P2 - P1)
            else:
                pre_mut = P1.copy()
            post_mut = pre_mut.copy()
            mut_mask = rng.random(d) < mut_rate
            post_mut[mut_mask] += rng.normal(0, mut_sigma, size=d)[mut_mask]
            post_mut = np.clip(post_mut, lo, hi)
            events.append(dict(elite=False, p1_pos=P1.copy(), p2_pos=P2.copy(),
                                pre_mut=pre_mut.copy(), post_mut=post_mut.copy()))
            children.append(post_mut)
        pop = np.array(children)
        fit = truth_fn(pop)
        gi = int(np.argmin(fit))
        if fit[gi] < best_val:
            best_val, best_pos = float(fit[gi]), pop[gi].copy()
        history.append(dict(pop=pop.copy(), fit=fit.copy(), best_val=best_val, best_pos=best_pos.copy(),
                             mean_val=float(fit.mean()), events=events))
    return history


def run_pso(truth_fn, bounds, pop_size, w, c1, c2, vmax_frac, max_gen, seed):
    rng = np.random.default_rng(seed)
    d = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    span = hi - lo
    x = lo + rng.uniform(size=(pop_size, d)) * span
    v = rng.uniform(-1, 1, size=(pop_size, d)) * span * 0.1
    fit = truth_fn(x)
    pbest, pbest_val = x.copy(), fit.copy()
    g_i = int(np.argmin(pbest_val))
    gbest, gbest_val = pbest[g_i].copy(), float(pbest_val[g_i])
    vmax = vmax_frac * span

    history = [dict(pop=x.copy(), fit=fit.copy(), best_val=gbest_val, best_pos=gbest.copy(),
                     mean_val=float(fit.mean()), events=None)]

    for _ in range(max_gen):
        events = []
        x_new, v_new = np.zeros_like(x), np.zeros_like(v)
        for i in range(pop_size):
            r1, r2 = rng.random(d), rng.random(d)
            inertia = w * v[i]
            cognitive = c1 * r1 * (pbest[i] - x[i])
            social = c2 * r2 * (gbest - x[i])
            vi_new = np.clip(inertia + cognitive + social, -vmax, vmax)
            xi_new = np.clip(x[i] + vi_new, lo, hi)
            events.append(dict(x=x[i].copy(), pbest=pbest[i].copy(), gbest=gbest.copy(),
                                inertia=inertia.copy(), cognitive=cognitive.copy(), social=social.copy()))
            x_new[i], v_new[i] = xi_new, vi_new
        x, v = x_new, v_new
        fit = truth_fn(x)
        improved = fit < pbest_val
        pbest[improved] = x[improved]
        pbest_val[improved] = fit[improved]
        g_i = int(np.argmin(pbest_val))
        if pbest_val[g_i] < gbest_val:
            gbest_val, gbest = float(pbest_val[g_i]), pbest[g_i].copy()
        history.append(dict(pop=x.copy(), fit=fit.copy(), best_val=gbest_val, best_pos=gbest.copy(),
                             mean_val=float(fit.mean()), events=events))
    return history


def run_de(truth_fn, bounds, pop_size, F, CR, max_gen, seed):
    rng = np.random.default_rng(seed)
    d = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    x = lo + rng.uniform(size=(pop_size, d)) * (hi - lo)
    fit = truth_fn(x)
    b_i = int(np.argmin(fit))
    best_val, best_pos = float(fit[b_i]), x[b_i].copy()

    history = [dict(pop=x.copy(), fit=fit.copy(), best_val=best_val, best_pos=best_pos.copy(),
                     mean_val=float(fit.mean()), events=None)]

    for _ in range(max_gen):
        new_x, new_fit, events = x.copy(), fit.copy(), []
        for i in range(pop_size):
            idxs = [j for j in range(pop_size) if j != i]
            r1, r2, r3 = rng.choice(idxs, 3, replace=False)
            mutant = np.clip(x[r1] + F * (x[r2] - x[r3]), lo, hi)
            trial = x[i].copy()
            j_rand = rng.integers(0, d)
            cross_mask = rng.random(d) < CR
            cross_mask[j_rand] = True
            trial[cross_mask] = mutant[cross_mask]
            f_trial = float(truth_fn(trial.reshape(1, -1))[0])
            accepted = f_trial <= fit[i]
            if accepted:
                new_x[i], new_fit[i] = trial, f_trial
            events.append(dict(x_r1=x[r1].copy(), x_r2=x[r2].copy(), x_r3=x[r3].copy(),
                                mutant=mutant.copy(), trial=trial.copy(), x_i=x[i].copy(),
                                accepted=accepted))
        x, fit = new_x, new_fit
        gi = int(np.argmin(fit))
        if fit[gi] < best_val:
            best_val, best_pos = float(fit[gi]), x[gi].copy()
        history.append(dict(pop=x.copy(), fit=fit.copy(), best_val=best_val, best_pos=best_pos.copy(),
                             mean_val=float(fit.mean()), events=events))
    return history


NATURE_EQUATIONS = {
    "Genetic Algorithm": [
        r"\text{child} = p_1 + \alpha\,(p_2-p_1), \quad \alpha \sim U(-0.25,\,1.25)",
        r"\text{gene}_j \mathrel{+}= \mathcal{N}(0,\sigma^2) \text{ with probability } p_{mut}",
    ],
    "Particle Swarm Optimization": [
        r"v_{k+1}=w\,v_k+c_1 r_1\,(p_{best}-x_k)+c_2 r_2\,(g_{best}-x_k)",
        r"x_{k+1}=x_k+v_{k+1}",
    ],
    "Differential Evolution": [
        r"v = x_{r1}+F\,(x_{r2}-x_{r3})",
        r"\text{trial}_j = v_j \text{ if } \text{rand}_j<CR \text{ else } x_{i,j}",
    ],
}

ALGO_EXPLAIN_DERIV = {
    "Gradient Descent": """
- **Always goes downhill.** Each step moves in the direction of steepest descent, -grad f(x),
  scaled by the learning rate eta. Because -grad f always points toward *lower* f locally,
  Gradient Descent can only converge to minima (or get stuck on flat plateaus/ridges) -
  never maxima.
- **The learning rate is the whole story.** Too small and convergence crawls; too large and
  it overshoots, oscillates, or diverges outright - try it on the rugged function.
- **No curvature information.** It doesn't know whether it's in a steep, narrow valley or a
  wide, flat one - it takes a step based only on the local slope, which is why it can zig-zag
  down narrow valleys (see Rosenbrock in 2D).
""",
    "Newton-Raphson": """
- **It solves grad f(x) = 0, not "go downhill."** The step jumps straight to the stationary
  point of a local quadratic approximation of f. That stationary point is a minimum only if
  the local curvature (the Hessian) is positive definite there.
- **This is exactly why it can converge to maxima or saddle points** - if the Hessian is
  negative definite or indefinite near the current point, the quadratic model's stationary
  point is a maximum (or saddle), and Newton heads there with no hesitation. Watch for the
  "indefinite Hessian" warning above - that's your signal it might be happening.
- **Very fast near a minimum** (quadratic convergence - correct digits roughly double each
  step), which is the main reason to tolerate the maxima risk at all.
- **Damping (lambda) and step size (alpha)** are the safety valves: damping pushes the
  Hessian toward positive-definite (trading speed for safety), and step size shrinks each
  jump so it can't overshoot as badly.
""",
    "Adam": """
- **Still a descent method.** Adam rescales each coordinate of the gradient by a running
  estimate of its own typical magnitude, but the direction is still built from grad f, so -
  like Gradient Descent - it's structurally biased toward going downhill, not toward maxima.
- **m and v are running averages** of the gradient and its square (first and second moment),
  with beta1/beta2 controlling how much history they remember. The bias-correction terms
  (dividing by 1-beta^k) compensate for m and v both starting at zero.
- **The step size adapts per coordinate** - a dimension with a small, consistent gradient
  gets a relatively larger effective step than one with a large or noisy gradient, which is
  why Adam handles the narrow Rosenbrock valley more gracefully than plain Gradient Descent.
""",
}

ALGO_EXPLAIN_NATURE = {
    "Genetic Algorithm": """
- **The dashed best-so-far line** tracks the best fitness ever found, generation by
  generation - not any one individual's path. Individuals don't persist; a whole new
  population is produced every generation.
- **The mechanism diagram** (pick a child above, once you've stepped past generation 0) shows
  one child's construction: the dotted line connects its two tournament-selected parents, blend
  crossover produces an intermediate point (white x), and mutation then nudges it to its final
  position (green star). An elite child skips all of this - it's a verbatim copy of a top
  individual from the previous generation.
- **Crossover rate** controls how often blending happens at all (vs. just copying parent 1).
  **Mutation rate/strength** control how often, and how far, each gene gets perturbed.
  **Elitism** guarantees the best individual(s) are never lost to bad luck. **Tournament size**
  sets selection pressure - larger tournaments favor fitter parents more strongly.
""",
    "Particle Swarm Optimization": """
- **The three chained arrows** are a literal vector decomposition of one particle's velocity
  update: inertia (carries forward its previous motion), cognitive (pulls toward its own
  personal best), and social (pulls toward the swarm's global best) - drawn head-to-tail, so
  the last arrowhead lands exactly on the particle's new position.
- **Inertia (w)** controls how much old momentum carries over - high values explore more, low
  values settle faster. **Cognitive/social coefficients (c1/c2)** balance "trust your own
  experience" against "trust the swarm". **Max speed** caps how far a particle can move in one
  generation, preventing wild overshoots.
- Unlike GA and DE, PSO keeps the same particles across generations - only their positions and
  velocities evolve - and the global best can only ever improve, which is why its best-so-far
  curve tends to look like a clean staircase rather than a jagged descent.
""",
    "Differential Evolution": """
- **The mechanism diagram** draws the mutation formula as geometry: the dashed line is the
  difference x_r2 - x_r3, and the solid arrow from x_r1 adds F times that difference to reach
  the mutant - exactly v = x_r1 + F * (x_r2 - x_r3).
- **Crossover (CR)** then mixes the mutant into a trial vector gene-by-gene against the
  original x_i, and **selection is greedy**: the trial replaces x_i only if it's at least as
  good, so no individual's fitness can ever get worse from one generation to the next.
- **F** controls how large a step the mutation takes (too high and it behaves almost randomly;
  too low and the population can't escape a cluster). **CR** controls how much of the mutant
  actually makes it into the trial versus staying as the original value.
""",
}


# --------------------------------------------------------------------------------------
# Plotting helpers. `_xy` maps any design-space point to the 2D coordinates the
# figure is drawn in: (x, f(x)) for a 1D problem, (x1, x2) for a 2D problem - so
# every overlay below is written once and works for both dimensions.
# --------------------------------------------------------------------------------------

def _xy(dim, p, truth_fn):
    if dim == 1:
        return float(p[0]), float(truth_fn(np.asarray(p, dtype=float).reshape(1, -1))[0])
    return float(p[0]), float(p[1])


def _add_point(fig, p, color, symbol="circle", size=12, name=None, line_color="black"):
    fig.add_trace(go.Scatter(x=[p[0]], y=[p[1]], mode="markers",
                              marker=dict(color=color, size=size, symbol=symbol,
                                          line=dict(color=line_color, width=1.5)),
                              name=name or "", showlegend=bool(name)))


def _add_arrow(fig, p0, p1, color, width=2.5):
    fig.add_annotation(x=p1[0], y=p1[1], ax=p0[0], ay=p0[1], xref="x", yref="y",
                        axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1,
                        arrowwidth=width, arrowcolor=color)


def _add_line(fig, p0, p1, color, dash="dot", width=1.5):
    fig.add_trace(go.Scatter(x=[p0[0], p1[0]], y=[p0[1], p1[1]], mode="lines",
                              line=dict(color=color, dash=dash, width=width),
                              showlegend=False, hoverinfo="skip"))


def _base_figure(dim, grid_info):
    if dim == 1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=grid_info["x"], y=grid_info["truth"], mode="lines",
                                  line=dict(color=TEXT, width=3), name="f(x)"))
    else:
        fig = go.Figure(go.Contour(x=grid_info["x1"], y=grid_info["x2"], z=grid_info["truth"],
                                    colorscale=SEQ_LANDSCAPE, reversescale=True,
                                    contours=dict(coloring="heatmap", showlabels=False),
                                    ncontours=24, showscale=False, name="f"))
    return fig


def _finish_figure(fig, dim, grid_info, height=560):
    # Ranges are pinned to the problem's own domain rather than left to autorange, so a
    # divergent step (still finite, but far outside the domain) can't stretch the whole
    # plot until the landscape itself is unreadable.
    if dim == 1:
        x_range = [float(grid_info["x"].min()), float(grid_info["x"].max())]
        pad = 0.08 * (float(grid_info["truth"].max()) - float(grid_info["truth"].min()) + 1e-9)
        y_range = [float(grid_info["truth"].min()) - pad, float(grid_info["truth"].max()) + pad]
    else:
        x_range = [float(grid_info["x1"].min()), float(grid_info["x1"].max())]
        y_range = [float(grid_info["x2"].min()), float(grid_info["x2"].max())]
    fig.update_xaxes(range=x_range)
    fig.update_yaxes(range=y_range)
    fig.update_layout(
        xaxis_title=("x" if dim == 1 else "x1"),
        yaxis_title=("f(x)" if dim == 1 else "x2"),
        template=PLOT_TEMPLATE, height=height, margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02), font=dict(size=13),
    )
    return fig


def plot_single_point(dim, grid_info, truth_fn, steps, idx, global_min):
    fig = _base_figure(dim, grid_info)
    _add_point(fig, _xy(dim, global_min[0], truth_fn), COLOR_TARGET, symbol="star",
               size=14, name="global min (grid search)")

    pts = [s for s in steps[:idx + 1] if np.all(np.isfinite(s["x"])) and np.isfinite(s["fx"])]
    xy = [_xy(dim, s["x"], truth_fn) for s in pts]
    if len(xy) >= 2:
        fig.add_trace(go.Scatter(x=[p[0] for p in xy], y=[p[1] for p in xy], mode="lines+markers",
                                  line=dict(color=COLOR_TRAJ, width=2),
                                  marker=dict(color=COLOR_TRAJ, size=7), name="trajectory"))
        _add_arrow(fig, xy[-2], xy[-1], COLOR_GRAD)
    if xy:
        _add_point(fig, xy[-1], COLOR_POINT, symbol="circle", size=17, name="current position")
    return _finish_figure(fig, dim, grid_info)


def overlay_ga(fig, dim, truth_fn, event):
    if event["elite"]:
        _add_point(fig, _xy(dim, event["post_mut"], truth_fn), COLOR_BEST, symbol="diamond",
                   size=16, name="elite carry-over")
        return
    p1 = _xy(dim, event["p1_pos"], truth_fn)
    p2 = _xy(dim, event["p2_pos"], truth_fn)
    pre = _xy(dim, event["pre_mut"], truth_fn)
    post = _xy(dim, event["post_mut"], truth_fn)
    _add_line(fig, p1, p2, COLOR_TRAJ, dash="dot")
    _add_point(fig, p1, COLOR_TRAJ, symbol="circle", size=13, name="parent 1")
    _add_point(fig, p2, COLOR_BEST, symbol="circle", size=13, name="parent 2")
    _add_point(fig, pre, "#ffffff", symbol="x", size=11, name="after crossover")
    if not np.allclose(event["pre_mut"], event["post_mut"]):
        _add_arrow(fig, pre, post, COLOR_GRAD)
    _add_point(fig, post, COLOR_GRAD, symbol="star", size=16, name="child (final)")


def overlay_pso(fig, dim, truth_fn, event):
    x = np.asarray(event["x"], dtype=float)
    p0 = _xy(dim, x, truth_fn)
    p_in = _xy(dim, x + event["inertia"], truth_fn)
    p_cog = _xy(dim, x + event["inertia"] + event["cognitive"], truth_fn)
    p_soc = _xy(dim, x + event["inertia"] + event["cognitive"] + event["social"], truth_fn)
    _add_point(fig, p0, COLOR_POINT, symbol="circle", size=13, name="particle (before)")
    _add_arrow(fig, p0, p_in, COLOR_TRAJ)
    _add_arrow(fig, p_in, p_cog, COLOR_POINT)
    _add_arrow(fig, p_cog, p_soc, COLOR_BEST)
    _add_point(fig, _xy(dim, event["pbest"], truth_fn), COLOR_POINT, symbol="diamond",
               size=13, name="personal best")
    _add_point(fig, _xy(dim, event["gbest"], truth_fn), COLOR_BEST, symbol="diamond",
               size=14, name="global best")
    _add_point(fig, p_soc, COLOR_GRAD, symbol="star", size=17, name="particle (after)")


def overlay_de(fig, dim, truth_fn, event):
    r1 = _xy(dim, event["x_r1"], truth_fn)
    r2 = _xy(dim, event["x_r2"], truth_fn)
    r3 = _xy(dim, event["x_r3"], truth_fn)
    mutant = _xy(dim, event["mutant"], truth_fn)
    trial = _xy(dim, event["trial"], truth_fn)
    xi = _xy(dim, event["x_i"], truth_fn)
    _add_line(fig, r3, r2, COLOR_TRAJ, dash="dash")
    _add_point(fig, r2, COLOR_TRAJ, symbol="circle", size=12, name="x_r2")
    _add_point(fig, r3, COLOR_BEST, symbol="circle", size=12, name="x_r3")
    _add_point(fig, r1, "#ffffff", symbol="square", size=12, name="x_r1 (base)")
    _add_arrow(fig, r1, mutant, COLOR_GRAD)
    _add_point(fig, mutant, COLOR_GRAD, symbol="diamond", size=14, name="mutant")
    _add_point(fig, xi, COLOR_POINT, symbol="circle", size=12, name="target x_i")
    sym = "star" if event["accepted"] else "x"
    _add_point(fig, trial, "#ffffff", symbol=sym, size=15,
               name="trial (accepted)" if event["accepted"] else "trial (rejected)")


OVERLAYS = {
    "Genetic Algorithm": overlay_ga,
    "Particle Swarm Optimization": overlay_pso,
    "Differential Evolution": overlay_de,
}


def plot_population(dim, grid_info, truth_fn, history, gen_idx, event_idx, algo, global_min):
    cur = history[gen_idx]
    prev = history[gen_idx - 1] if gen_idx > 0 else None
    fig = _base_figure(dim, grid_info)
    _add_point(fig, _xy(dim, global_min[0], truth_fn), COLOR_TARGET, symbol="star",
               size=14, name="global min (grid search)")

    if prev is not None:
        pxy = [_xy(dim, p, truth_fn) for p in prev["pop"]]
        fig.add_trace(go.Scatter(x=[p[0] for p in pxy], y=[p[1] for p in pxy], mode="markers",
                                  marker=dict(color=COLOR_GHOST, size=5), name="previous generation"))

    cxy = [_xy(dim, p, truth_fn) for p in cur["pop"]]
    fig.add_trace(go.Scatter(x=[p[0] for p in cxy], y=[p[1] for p in cxy], mode="markers",
                              marker=dict(color=cur["fit"], colorscale=SEQ_FIT, reversescale=True,
                                          size=9, line=dict(color="white", width=1)),
                              name="current population"))

    best_traj = [_xy(dim, h["best_pos"], truth_fn) for h in history[:gen_idx + 1]]
    if len(best_traj) >= 2:
        fig.add_trace(go.Scatter(x=[p[0] for p in best_traj], y=[p[1] for p in best_traj],
                                  mode="lines", line=dict(color=COLOR_BEST, width=4, dash="dash"),
                                  name="best-so-far path"))
    _add_point(fig, best_traj[-1], COLOR_BEST, symbol="diamond", size=16, name="best-so-far")

    if gen_idx > 0 and cur["events"] is not None and event_idx is not None:
        OVERLAYS[algo](fig, dim, truth_fn, cur["events"][event_idx])

    return _finish_figure(fig, dim, grid_info)


def plot_progress_deriv(steps):
    iters = list(range(len(steps)))
    fx = [s["fx"] for s in steps]
    gnorm = [float(np.linalg.norm(s["grad"])) for s in steps]
    fig = make_subplots(rows=1, cols=2, subplot_titles=["f(x) per iteration", "||grad f(x)|| per iteration"])
    fig.add_trace(go.Scatter(x=iters, y=fx, mode="lines+markers",
                              line=dict(color=COLOR_TRAJ, width=3), marker=dict(size=6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=iters, y=gnorm, mode="lines+markers",
                              line=dict(color=COLOR_GRAD, width=3), marker=dict(size=6)), row=1, col=2)
    fig.update_yaxes(type="log", row=1, col=2)
    fig.update_xaxes(title_text="iteration")
    fig.update_layout(template=PLOT_TEMPLATE, height=360, showlegend=False,
                       margin=dict(l=10, r=10, t=40, b=10))
    return fig


def plot_progress_pop(history):
    gens = list(range(len(history)))
    best = [h["best_val"] for h in history]
    mean = [h["mean_val"] for h in history]
    diversity = [float(np.mean(np.std(h["pop"], axis=0))) for h in history]
    fig = make_subplots(rows=1, cols=2,
                         subplot_titles=["best-so-far & population mean", "population diversity"])
    fig.add_trace(go.Scatter(x=gens, y=best, name="best-so-far",
                              line=dict(color=COLOR_BEST, width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=gens, y=mean, name="population mean",
                              line=dict(color=COLOR_TRAJ, width=2, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=gens, y=diversity, showlegend=False,
                              line=dict(color=COLOR_GRAD, width=3)), row=1, col=2)
    fig.update_yaxes(title_text="mean std across dimensions", row=1, col=2)
    fig.update_xaxes(title_text="generation")
    fig.update_layout(template=PLOT_TEMPLATE, height=360,
                       legend=dict(orientation="h", y=1.18, x=0.0),
                       margin=dict(l=10, r=10, t=40, b=10))
    return fig


def fmt_vec(v, nd=4):
    return "[" + ", ".join(f"{x:.{nd}f}" for x in np.atleast_1d(v)) + "]"


def slider_input(label, min_value, max_value, value, step, key, help=None, fmt=None):
    """A slider paired with a number input, kept in sync through session state - so a
    parameter can be dragged roughly or typed exactly, whichever a slide bar alone can't do."""
    is_int = isinstance(min_value, int) and isinstance(max_value, int) and isinstance(value, int)
    cast = int if is_int else float
    skey, ikey = f"{key}__sl", f"{key}__in"

    current = cast(np.clip(st.session_state.get(skey, value), min_value, max_value))
    st.session_state[skey] = current
    st.session_state[ikey] = current

    def _from_slider():
        st.session_state[ikey] = st.session_state[skey]

    def _from_input():
        st.session_state[skey] = st.session_state[ikey]

    c1, c2 = st.columns([2.3, 1], gap="small")
    with c1:
        st.slider(label, min_value, max_value, step=step, key=skey, help=help, on_change=_from_slider)
    with c2:
        st.number_input(label, min_value, max_value, step=step, key=ikey, on_change=_from_input,
                         label_visibility="collapsed", format=fmt)
    return st.session_state[skey]


# --------------------------------------------------------------------------------------
# Sidebar - problem and method configuration
# --------------------------------------------------------------------------------------

st.title("Optimization Algorithms Explorer")
st.caption("Watch an optimizer move through a design space one step at a time - a single "
           "point tracing a path for derivative-based methods, a population evolving "
           "generation by generation for nature-inspired ones.")

with st.sidebar:
    st.header("1. Problem setup")
    dim = st.radio("Design space dimension", [1, 2], horizontal=True, format_func=lambda d: f"{d}D")

    if dim == 1:
        fname = st.selectbox("Test function", list(FUNCS_1D.keys()))
        default_bounds = FUNCS_1D[fname]
        custom_expr = None
        if fname == "Custom expression":
            custom_expr = st.text_input("f(x) =", value="x**2 + 2*sin(3*x)")
        c1, c2 = st.columns(2)
        lo = c1.number_input("Lower bound", value=float(default_bounds[0]))
        hi = c2.number_input("Upper bound", value=float(default_bounds[1]))
        bounds = [(lo, hi)]
    else:
        fname = st.selectbox("Test function", list(FUNCS_2D.keys()))
        default_bounds = FUNCS_2D[fname]
        custom_expr = None
        if fname == "Custom expression":
            custom_expr = st.text_input("f(x1, x2) =", value="x1**2 + x2**2 + 2*sin(x1)*cos(x2)")
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
        expr_ok, err = validate_expr(custom_expr, dim)
        if not expr_ok:
            st.error(f"Invalid expression: {err}")

    seed = st.number_input("Random seed", value=42, step=1)

    st.divider()
    st.header("2. Method")
    category = st.radio("Approach", ["Derivative-based", "Nature-inspired"])

    if category == "Derivative-based":
        algo = st.selectbox("Algorithm", ["Gradient Descent", "Newton-Raphson", "Adam"])
        for eq in DERIV_EQUATIONS[algo]:
            st.latex(eq)

        if algo == "Gradient Descent":
            lr = slider_input("Learning rate (eta)", 0.001, 2.0, 0.1, 0.001, key="lr_gd", fmt="%.3f")
        elif algo == "Newton-Raphson":
            newton_alpha = slider_input("Step size (alpha)", 0.05, 1.5, 1.0, 0.05,
                                         key="newton_alpha", fmt="%.2f")
            log_damp = slider_input("Hessian damping: log10(lambda)", -6.0, 1.0, -3.0, 0.5,
                                     key="newton_log_damp", fmt="%.1f")
            newton_damping = 10 ** log_damp
        else:
            lr = slider_input("Learning rate (eta)", 0.001, 2.0, 0.2, 0.001, key="lr_adam", fmt="%.3f")
            beta1 = slider_input("beta1", 0.5, 0.999, 0.9, 0.001, key="adam_beta1", fmt="%.3f")
            beta2 = slider_input("beta2", 0.8, 0.9999, 0.999, 0.0001, key="adam_beta2", fmt="%.4f")
            log_eps = slider_input("log10(epsilon)", -10.0, -2.0, -8.0, 0.5, key="adam_log_eps", fmt="%.1f")
            adam_eps = 10 ** log_eps
        max_iter = slider_input("Max iterations", 5, 300, 60, 5, key="max_iter")

        st.header("3. Starting point")
        start_mode = st.radio("Start", ["Manual", "Random (seeded)"], horizontal=True)
        if dim == 1:
            if start_mode == "Manual":
                x0 = np.array([slider_input("x0", float(bounds[0][0]), float(bounds[0][1]),
                                             float((bounds[0][0] + bounds[0][1]) / 2), 0.01,
                                             key="x0_1d", fmt="%.3f")])
            else:
                x0 = init_population(bounds, 1, int(seed))[0]
        else:
            if start_mode == "Manual":
                x1_0 = slider_input("x1_0", float(bounds[0][0]), float(bounds[0][1]),
                                     float((bounds[0][0] + bounds[0][1]) / 2), 0.01,
                                     key="x1_0_2d", fmt="%.3f")
                x2_0 = slider_input("x2_0", float(bounds[1][0]), float(bounds[1][1]),
                                     float((bounds[1][0] + bounds[1][1]) / 2), 0.01,
                                     key="x2_0_2d", fmt="%.3f")
                x0 = np.array([x1_0, x2_0])
            else:
                x0 = init_population(bounds, 1, int(seed))[0]
    else:
        algo = st.selectbox("Algorithm", ["Genetic Algorithm", "Particle Swarm Optimization",
                                           "Differential Evolution"])
        for eq in NATURE_EQUATIONS[algo]:
            st.latex(eq)

        pop_size = slider_input("Population size", 6, 80, 24, 2, key="pop_size")
        max_gen = slider_input("Max generations", 5, 150, 40, 5, key="max_gen")

        if algo == "Genetic Algorithm":
            cx_rate = slider_input("Crossover rate", 0.0, 1.0, 0.8, 0.05, key="ga_cx_rate", fmt="%.2f")
            mut_rate = slider_input("Mutation rate (per gene)", 0.0, 1.0, 0.2, 0.05,
                                     key="ga_mut_rate", fmt="%.2f")
            mut_sigma = slider_input("Mutation strength (sigma)", 0.01, 2.0, 0.3, 0.01,
                                      key="ga_mut_sigma", fmt="%.2f")
            elite_count = slider_input("Elitism (carried over unchanged)", 0, 5, 1, 1, key="ga_elite_count")
            tournament_k = slider_input("Tournament size", 2, 6, 3, 1, key="ga_tournament_k")
        elif algo == "Particle Swarm Optimization":
            pso_w = slider_input("Inertia weight (w)", 0.0, 1.2, 0.7, 0.05, key="pso_w", fmt="%.2f")
            pso_c1 = slider_input("Cognitive coefficient (c1)", 0.0, 3.0, 1.5, 0.1, key="pso_c1", fmt="%.2f")
            pso_c2 = slider_input("Social coefficient (c2)", 0.0, 3.0, 1.5, 0.1, key="pso_c2", fmt="%.2f")
            pso_vmax_frac = slider_input("Max speed (fraction of domain)", 0.02, 1.0, 0.2, 0.02,
                                          key="pso_vmax_frac", fmt="%.2f")
        else:
            de_F = slider_input("Differential weight (F)", 0.1, 2.0, 0.6, 0.05, key="de_F", fmt="%.2f")
            de_CR = slider_input("Crossover rate (CR)", 0.0, 1.0, 0.9, 0.05, key="de_CR", fmt="%.2f")

if not expr_ok:
    st.stop()

truth_fn = make_truth_fn(dim, fname, custom_expr)


def f_point(x):
    return float(truth_fn(np.asarray(x, dtype=float).reshape(1, -1))[0])


# --------------------------------------------------------------------------------------
# Landscape + global minimum (dense grid search, works for any function including
# custom expressions - no analytic minimum needed)
# --------------------------------------------------------------------------------------

if dim == 1:
    xg = np.linspace(bounds[0][0], bounds[0][1], 400)
    truth_g = truth_fn(xg.reshape(-1, 1))
    gi = int(np.argmin(truth_g))
    global_min = (np.array([xg[gi]]), float(truth_g[gi]))
    grid_info = dict(x=xg, truth=truth_g)
else:
    x1g, x2g, X1, X2, grid = build_2d_grid(bounds, n=100)
    truth_grid = truth_fn(grid).reshape(X1.shape)
    flat_i = int(np.argmin(truth_grid))
    gidx = np.unravel_index(flat_i, truth_grid.shape)
    global_min = (np.array([X1[gidx], X2[gidx]]), float(truth_grid[gidx]))
    grid_info = dict(x1=x1g, x2=x2g, X1=X1, X2=X2, truth=truth_grid)

# --------------------------------------------------------------------------------------
# Run the chosen algorithm
# --------------------------------------------------------------------------------------

if category == "Derivative-based":
    if algo == "Gradient Descent":
        steps = run_gd(f_point, x0, bounds, lr, max_iter)
    elif algo == "Newton-Raphson":
        steps = run_newton(f_point, x0, bounds, newton_alpha, newton_damping, max_iter)
    else:
        steps = run_adam(f_point, x0, bounds, lr, beta1, beta2, adam_eps, max_iter)
    last = len(steps) - 1
else:
    if algo == "Genetic Algorithm":
        history = run_ga(truth_fn, bounds, pop_size, cx_rate, mut_rate, mut_sigma,
                          elite_count, tournament_k, max_gen, int(seed))
    elif algo == "Particle Swarm Optimization":
        history = run_pso(truth_fn, bounds, pop_size, pso_w, pso_c1, pso_c2,
                           pso_vmax_frac, max_gen, int(seed))
    else:
        history = run_de(truth_fn, bounds, pop_size, de_F, de_CR, max_gen, int(seed))
    last = len(history) - 1

if "opt_step" not in st.session_state:
    st.session_state.opt_step = 0
st.session_state.opt_step = int(np.clip(st.session_state.opt_step, 0, last))


def move_step(delta):
    st.session_state.opt_step = int(np.clip(st.session_state.opt_step + delta, 0, last))


def restart_step():
    st.session_state.opt_step = 0


def jump_end():
    st.session_state.opt_step = last


# --------------------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------------------

tab_problem, tab_optimize, tab_history = st.tabs(["1. Problem", "2. Optimize", "3. History"])

# ---- Tab 1: problem ---------------------------------------------------------------
with tab_problem:
    if fname == "Custom expression":
        st.code(f"f = {custom_expr}", language="text")
    else:
        st.latex((FUNC_LATEX_1D if dim == 1 else FUNC_LATEX_2D)[fname])

    fig_land = _base_figure(dim, grid_info)
    _add_point(fig_land, _xy(dim, global_min[0], truth_fn), COLOR_TARGET, symbol="star",
               size=16, name="global min (grid search)")
    st.plotly_chart(_finish_figure(fig_land, dim, grid_info, height=520), width="stretch")
    st.caption(f"Global minimum located by dense grid search: x = {fmt_vec(global_min[0])}, "
               f"f = {global_min[1]:.5g}. This is an approximation for teaching purposes, not "
               "an analytic result - its precision is limited by the grid resolution.")

# ---- Tab 2: optimize ---------------------------------------------------------------
with tab_optimize:
    bcol1, bcol2, bcol3, bcol4, bcol5 = st.columns([0.8, 0.8, 0.8, 1.2, 3])
    bcol1.button("Back", width="stretch", on_click=move_step, args=(-1,),
                 disabled=st.session_state.opt_step == 0)
    bcol2.button("Next", width="stretch", on_click=move_step, args=(1,),
                 disabled=st.session_state.opt_step >= last)
    bcol3.button("Restart", width="stretch", on_click=restart_step)
    bcol4.button("Jump to end", width="stretch", on_click=jump_end)
    idx = st.session_state.opt_step
    unit = "Iteration" if category == "Derivative-based" else "Generation"
    with bcol5:
        st.write("")
        st.markdown(f"**{unit} {idx} of {last}**")
    st.progress((idx + 1) / (last + 1))

    left, right = st.columns([1, 1.6], gap="medium")

    if category == "Derivative-based":
        cur = steps[idx]
        gnorm = float(np.linalg.norm(cur["grad"]))

        with left:
            st.subheader(algo)
            for eq in DERIV_EQUATIONS[algo]:
                st.latex(eq)
            mcols = st.columns(2)
            mcols[0].metric("f(x)", f"{cur['fx']:.5g}" if np.isfinite(cur["fx"]) else "-")
            mcols[1].metric("||grad f(x)||", f"{gnorm:.5g}")
            if cur["note"]:
                st.warning(cur["note"])
            elif gnorm < GRAD_TOL:
                st.success("Converged: gradient norm below tolerance.")
            lines = [f"x        = {fmt_vec(cur['x'])}", f"grad f   = {fmt_vec(cur['grad'])}"]
            if algo == "Newton-Raphson":
                lines.append(f"hess f   =\n{cur['hess']}")
            elif algo == "Adam":
                lines += [f"m        = {fmt_vec(cur['m'])}", f"v        = {fmt_vec(cur['v'])}"]
            st.code("\n".join(lines), language="text")

        with right:
            st.plotly_chart(plot_single_point(dim, grid_info, truth_fn, steps, idx, global_min),
                             width="stretch")
            st.caption("The trajectory line traces every visited point; the arrow is the most "
                       "recent step.")
            with st.expander(f"How {algo} works"):
                st.markdown(ALGO_EXPLAIN_DERIV[algo])
    else:
        cur = history[idx]
        diversity = float(np.mean(np.std(cur["pop"], axis=0)))

        with left:
            st.subheader(algo)
            for eq in NATURE_EQUATIONS[algo]:
                st.latex(eq)
            mcols = st.columns(2)
            mcols[0].metric("Best-so-far f", f"{cur['best_val']:.5g}")
            mcols[1].metric("Population mean f", f"{cur['mean_val']:.5g}")
            st.caption(f"Population diversity (mean std across dimensions): {diversity:.4g}")

            label = {"Genetic Algorithm": "child", "Particle Swarm Optimization": "particle",
                      "Differential Evolution": "individual"}[algo]
            event_idx = None
            if idx > 0:
                n_events = len(cur["events"])
                if "event_idx" in st.session_state:
                    st.session_state.event_idx = min(st.session_state.event_idx, n_events - 1)
                event_idx = st.number_input(f"Illustrate update for {label} #", 0, n_events - 1,
                                             key="event_idx")
                ev = cur["events"][event_idx]
                if algo == "Genetic Algorithm":
                    if ev["elite"]:
                        st.info("This individual is an elite carry-over: copied unchanged, "
                                "no crossover or mutation applied.")
                    else:
                        st.code(f"parent 1   = {fmt_vec(ev['p1_pos'])}\n"
                                f"parent 2   = {fmt_vec(ev['p2_pos'])}\n"
                                f"crossover  = {fmt_vec(ev['pre_mut'])}\n"
                                f"mutated    = {fmt_vec(ev['post_mut'])}", language="text")
                elif algo == "Particle Swarm Optimization":
                    st.code(f"x (before) = {fmt_vec(ev['x'])}\n"
                            f"inertia    = {fmt_vec(ev['inertia'])}\n"
                            f"cognitive  = {fmt_vec(ev['cognitive'])}\n"
                            f"social     = {fmt_vec(ev['social'])}\n"
                            f"personal best = {fmt_vec(ev['pbest'])}\n"
                            f"global best   = {fmt_vec(ev['gbest'])}", language="text")
                else:
                    st.code(f"x_r1 (base) = {fmt_vec(ev['x_r1'])}\n"
                            f"x_r2        = {fmt_vec(ev['x_r2'])}\n"
                            f"x_r3        = {fmt_vec(ev['x_r3'])}\n"
                            f"mutant      = {fmt_vec(ev['mutant'])}\n"
                            f"trial       = {fmt_vec(ev['trial'])}\n"
                            f"target x_i  = {fmt_vec(ev['x_i'])}\n"
                            f"accepted    = {ev['accepted']}", language="text")
            else:
                st.info("Step forward one generation to see the update mechanism illustrated.")

        with right:
            st.plotly_chart(plot_population(dim, grid_info, truth_fn, history, idx, event_idx,
                                             algo, global_min), width="stretch")
            st.caption("Faded dots are the previous generation; bright dots are the current "
                       "one, coloured by fitness. The dashed line traces the best point found "
                       "so far across generations.")
            with st.expander(f"How {algo} works"):
                st.markdown(ALGO_EXPLAIN_NATURE[algo])

# ---- Tab 3: history ------------------------------------------------------------------
with tab_history:
    if category == "Derivative-based":
        st.subheader("Convergence")
        st.plotly_chart(plot_progress_deriv(steps), width="stretch")
        st.caption(f"Ran {len(steps) - 1} iteration(s) "
                   + ("(stopped early)" if len(steps) - 1 < max_iter else "(reached the iteration cap)")
                   + ".")
    else:
        st.subheader("Convergence")
        st.plotly_chart(plot_progress_pop(history), width="stretch")
        st.caption("Diversity falling toward zero means the population has converged onto a "
                   "single region of the design space - compare it against the best-so-far "
                   "curve to see whether that region is actually the optimum.")
