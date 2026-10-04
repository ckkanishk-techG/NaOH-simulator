r"""Differentiable (JAX) likelihood of the reduced model: gradients w.r.t. all parameters, Gauss-Newton fitting
and NUTS (NumPyro). Optional - the NumPy/emcee path always works.

All experiments are simulated in one ``vmap``-ed ``lax.scan`` (padded to the longest duration); channel readings
are interpolated onto the fixed-step grid with precomputed indices; thermocouple lag is an exact IIR filter.
Only parameters with a direct entry in the model constant vector are supported here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core import l1_fast as lf
from ..core.params import ParamSet
from .calibrate import FitResult, Predictor
from .data import Experiment, ParamSpace

K_MAP = {"k25": "k25", "Ea": "Ea", "n_oh": "n", "tau_ind": "tau", "k_mt": "kmt", "ec_k_diss": "kd",
         "ec_k_pass": "kp", "ec_eps_al": "eps_al", "ec_aAl_floor": "a_floor"}


class JaxPosterior:
    def __init__(self, exps: list[Experiment], names: list[str], base: ParamSet | None = None,
                 model: str = "empirical", dt: float = 2.0, model_err_rel: float = 0.0) -> None:
        import jax
        import jax.numpy as jnp

        lf._jax_funcs()  # enables x64
        bad = [n for n in names if n not in K_MAP]
        if bad:
            raise ValueError(f"JAX path supports only {sorted(K_MAP)}; got {bad}")
        self.exps, self.names, self.dt, self.model_err_rel = exps, list(names), dt, model_err_rel
        self.base = base or ParamSet()
        self.space = ParamSpace.for_experiments(names, exps)
        self.k_idx = np.array([lf.KI[K_MAP[n]] for n in names])
        self.k_base = np.array([lf.build_constants(e.scenario, self.base, model, dt) for e in exps])
        self.y0 = np.array([lf.initial_state(e.scenario) for e in exps])
        self.nsteps = int(round(max(e.scenario.duration_s for e in exps) / dt))
        self._chan = []  # per channel: (exp index, col, idx, w, y, sigma, gain, offset, drift_h, tau, weight, t)
        col = {"n_h2": 5, "T": 2, "Tw": 3}
        for ie, e in enumerate(exps):
            for c in e.channels:
                pos = np.clip(c.t / dt, 0, self.nsteps - 1e-9)
                i0 = np.floor(pos).astype(int)
                w = pos - i0
                s = c.sensor
                self._chan.append((ie, col[c.kind], i0, w, c.y, c.sigma, s.gain if s else 1.0, s.offset if s else 0.0,
                                   s.drift_per_h if s else 0.0, s.tau_s if s else 0.0, e.weight, c.t))
        _, integ = lf._jax_funcs()
        self._integ = lf._JAX_CACHE["int"]
        self._vm = jax.vmap(lambda y0, k: self._integ(y0, k, dt, self.nsteps))
        self.jax, self.jnp = jax, jnp
        self.resid = jax.jit(self._resid)
        self.logpost = jax.jit(self._logpost)

    # --- pieces ---------------------------------------------------------------------------------
    def natural(self, x: Any) -> Any:
        jnp = self.jnp
        sp = self.space
        u = x * sp.scale + sp.centre
        if sp.shear is not None:
            from ..constants import R, T_REF

            ik, ie, inn, t_c, c_c = sp.shear
            term = -u[ie] / R * (1.0 / t_c - 1.0 / T_REF) + u[inn] * np.log(c_c)
            u = u.at[ik].add(term)
        return jnp.where(sp.is_log, jnp.exp(jnp.minimum(u, 700.0)), u), u  # magic: exp guard

    def _lag(self, y: Any, tau: float) -> Any:
        jnp, dt = self.jnp, self.dt
        if tau <= 0.0:
            return y
        a = float(np.exp(-dt / tau))
        c = tau * (1.0 - a) / dt

        def body(prev: Any, yy: Any) -> tuple[Any, Any]:
            out = a * prev[0] + (1.0 - c) * yy + (-a + c) * prev[1]
            return (out, yy), out

        _, ys = self.jax.lax.scan(body, (y[0], y[0]), y[1:])
        return jnp.concatenate([y[:1], ys])

    def _resid(self, x: Any, noise: float = 1.0) -> tuple[Any, Any]:
        jnp = self.jnp
        theta, _ = self.natural(x)
        ks = jnp.asarray(self.k_base).at[:, self.k_idx].set(theta[None, :])
        out = self._vm(jnp.asarray(self.y0), ks)  # (nexp, nsteps+1, 6)
        res, logs = [], 0.0
        for ie, colx, i0, w, y, sg, gain, off, drift, tau, wt, t in self._chan:
            series = out[ie, :, colx]
            if tau > 0 or gain != 1.0 or off != 0.0 or drift != 0.0:
                grid_t = jnp.arange(series.shape[0]) * self.dt
                series = gain * self._lag(series, tau) + off + drift * grid_t / 3600.0
            pr = series[i0] * (1 - w) + series[i0 + 1] * w
            s = noise * jnp.sqrt(sg**2 + (self.model_err_rel * pr) ** 2)
            res.append(jnp.sqrt(wt) * (y - pr) / s)
            logs = logs + jnp.sum(jnp.log(s))
        return jnp.concatenate(res), logs

    def _logpost(self, x: Any) -> Any:
        jnp = self.jnp
        sp = self.space
        r, logs = self._resid(x)
        _, u = self.natural(x)
        inb = jnp.all((u >= sp.ulo) & (u <= sp.uhi))
        pri = 0.0
        for i, e in enumerate(sp.entries):
            if e.dist in ("norm", "logn") and e.sd:
                pri = pri - 0.5 * ((u[i] - sp.centre_u[i]) / sp.scale[i]) ** 2
        return jnp.where(inb, pri - 0.5 * jnp.sum(r * r) - logs, -1.0e12)  # magic: soft wall outside bounds

    # --- API ---------------------------------------------------------------------------------------
    def residual_jacobian(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        jac = self.jax.jacfwd(lambda xx: self._resid(xx)[0])(self.jnp.asarray(x))
        return np.asarray(self.resid(self.jnp.asarray(x))[0]), np.asarray(jac)

    def fit_least_squares(self, x0: np.ndarray | None = None, max_nfev: int = 100) -> FitResult:
        """Bounded trust-region least squares with exact JAX Jacobians (forward-mode autodiff)."""
        from scipy.optimize import least_squares

        x_start = np.zeros(len(self.names)) if x0 is None else np.asarray(x0, float)
        cache: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}

        def both(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            key = np.asarray(x).tobytes()
            if key not in cache:
                cache.clear()
                cache[key] = self.residual_jacobian(np.asarray(x))
            return cache[key]

        sol = least_squares(lambda x: both(x)[0], x_start, jac=lambda x: both(x)[1],
                            bounds=(self.space.lo, self.space.hi), x_scale="jac", max_nfev=max_nfev,
                            xtol=1e-12, ftol=1e-12, gtol=1e-12)
        r, j = sol.fun, sol.jac
        n = len(r)
        chi2 = float(r @ r)
        cov = np.linalg.pinv(j.T @ j)
        return FitResult(self.names, self.space.from_u(sol.x), cov, chi2, n - len(self.names), n,
                         chi2 + 2 * len(self.names), chi2 + len(self.names) * np.log(n), j, r, bool(sol.success),
                         self.space, {"backend": "jax", "nfev": sol.nfev})

    def run_nuts(self, x0: np.ndarray, num_warmup: int = 300, num_samples: int = 400, num_chains: int = 2,
                 seed: int = 0) -> dict[str, Any]:
        """NUTS via NumPyro (requires ``numpyro``). Returns samples (x coordinates), natural params, R-hat, ESS."""
        import numpyro
        import numpyro.distributions as dist
        from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
        from numpyro.infer import MCMC, NUTS

        jnp = self.jnp
        sp = self.space

        def model() -> None:
            x = numpyro.sample("x", dist.Uniform(jnp.asarray(sp.lo), jnp.asarray(sp.hi)).to_event(1))
            numpyro.factor("ll", self._logpost(x))

        mcmc = MCMC(NUTS(model, init_strategy=numpyro.infer.init_to_value(values={"x": jnp.asarray(x0)})),
                    num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, progress_bar=False,
                    chain_method="sequential")
        mcmc.run(self.jax.random.PRNGKey(seed))
        xs = np.asarray(mcmc.get_samples(group_by_chain=True)["x"])  # (chains, samples, dim)
        nat = {n: np.empty(xs.shape[:2]) for n in self.names}
        for c in range(xs.shape[0]):
            for s in range(xs.shape[1]):
                for n, v in sp.from_u(xs[c, s]).items():
                    nat[n][c, s] = v
        return {"x": xs, "natural": nat, "rhat": np.asarray(split_gelman_rubin(xs)),
                "ess": np.asarray(effective_sample_size(xs))}


def gradient_of_outputs(sc: Any, names: list[str], base: ParamSet | None = None, model: str = "empirical",
                        dt: float = 4.0) -> dict[str, np.ndarray]:
    """d(final H2 moles, peak T)/d(parameter) by forward-mode autodiff (JAX) at the base parameters."""
    import jax
    import jax.numpy as jnp

    lf._jax_funcs()
    base = base or ParamSet()
    k0 = lf.build_constants(sc, base, model, dt)
    idx = np.array([lf.KI[K_MAP[n]] for n in names])
    nsteps = int(round(sc.duration_s / dt))
    y0 = jnp.asarray(lf.initial_state(sc))
    integ = lf._JAX_CACHE["int"]

    def outputs(theta: Any) -> Any:
        k = jnp.asarray(k0).at[idx].set(theta)
        o = integ(y0, k, dt, nsteps)
        return jnp.stack([o[-1, 5], jnp.max(o[:, 2])])

    th0 = jnp.asarray([k0[i] for i in idx])
    return {"jac": np.asarray(jax.jacfwd(outputs)(th0)), "outputs": np.asarray(outputs(th0)), "names": np.array(names)}


_ = Predictor
