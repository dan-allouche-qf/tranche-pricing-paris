"""End-to-end Monte Carlo engine.

Single entry point :func:`run_simulation` composes the simulation
building blocks into one vectorised batch:

1. Property-price path (GBM) and short-rate path (Vasicek).
2. Default times under the chosen credit model (Gaussian copula, Student-t
   or Cox doubly-stochastic).
3. Stochastic LGDs (Beta distribution).
4. Cumulative-loss path -> ASB waterfall -> per-tranche cash flows.

The output is a tidy :class:`SimulationOutput` that the pricing and risk
layers consume directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from ..credit import cox_intensity, gaussian_copula, student_t_copula
from ..credit import lgd as lgd_mod
from ..credit._types import default_time_from_uniform
from ..credit.lgd import BetaLGDParams
from ..markets.price_gbm import GBMParams
from ..markets.price_gbm import simulate_paths as simulate_gbm
from ..markets.rates_vasicek import VasicekParams
from ..markets.rates_vasicek import simulate_paths as simulate_vasicek
from ..waterfall import andersen_sidenius, loss_paths
from ..waterfall.andersen_sidenius import WaterfallOutcome
from ..waterfall.tranches import Tranche
from . import qmc, seeds

CreditModelName = Literal["gaussian_copula", "student_t_copula", "cox_intensity"]


@dataclass(slots=True)
class SimulationConfig:
    """All numerical knobs of one Monte Carlo run."""

    # Portfolio structure
    n_obligors: int
    horizon_years: float
    steps_per_year: int
    par: float
    gross_yield: float
    maintenance_pct: float
    initial_rate: float

    # Credit
    pd_terminal: float
    credit_model: CreditModelName
    rho: float
    nu: float = 5.0  # used only for Student-t
    cox: cox_intensity.CoxIntensityParams | None = None

    # Variance reduction
    n_paths: int = 50_000
    master_seed: int = 20260519
    antithetic: bool = True
    use_qmc: bool = False

    # Discounting convention
    discount_convention: Literal["vasicek_simulated", "constant_at_initial_rate"] = (
        "vasicek_simulated"
    )


@dataclass(slots=True)
class SimulationOutput:
    """All paths produced by one Monte Carlo run."""

    price_paths: NDArray[np.float64]  # (n_paths, n_steps + 1)
    rate_paths: NDArray[np.float64]  # (n_paths, n_steps + 1)
    discount_factors: NDArray[np.float64]  # (n_paths, n_steps + 1)
    default_times: NDArray[np.float64]  # (n_paths, n_obligors)
    lgd_samples: NDArray[np.float64]  # (n_paths, n_obligors)
    cumulative_loss: NDArray[np.float64]  # (n_paths, n_steps + 1)
    net_rent: NDArray[np.float64]  # (n_paths, n_steps)
    terminal_value: NDArray[np.float64]  # (n_paths,)
    waterfall: WaterfallOutcome
    config: SimulationConfig

    @property
    def n_paths(self) -> int:
        return int(self.price_paths.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.price_paths.shape[1] - 1)

    @property
    def dt(self) -> float:
        return float(self.config.horizon_years / self.n_steps)


def run_simulation(
    sim: SimulationConfig,
    *,
    gbm: GBMParams,
    vasicek: VasicekParams,
    lgd: BetaLGDParams,
    tranches: list[Tranche],
    coupons: dict[str, float],
) -> SimulationOutput:
    """Run one full Monte Carlo batch and return the per-path outputs."""
    if sim.n_paths <= 0 or sim.n_obligors <= 0 or sim.steps_per_year <= 0:
        raise ValueError("n_paths, n_obligors and steps_per_year must be positive.")
    if sim.horizon_years <= 0 or sim.par <= 0:
        raise ValueError("horizon_years and par must be positive.")
    if not 0 < sim.pd_terminal < 1:
        raise ValueError("pd_terminal must be in (0, 1).")
    if sim.use_qmc and sim.antithetic:
        import warnings

        warnings.warn(
            "QMC and antithetic flags are both set; the credit-factor stream "
            "uses Sobol points and ignores antithetic pairing. Market and rate "
            "streams continue to apply antithetic.",
            UserWarning,
            stacklevel=2,
        )

    n_steps = round(sim.horizon_years * sim.steps_per_year)
    dt = sim.horizon_years / n_steps

    streams = seeds.make_streams(sim.master_seed)

    # --- 1. Market paths --------------------------------------------------- #
    price_paths = simulate_gbm(
        gbm,
        s0=sim.par,
        n_paths=sim.n_paths,
        n_steps=n_steps,
        dt=dt,
        rng=streams.price,
        antithetic=sim.antithetic,
    )
    rate_paths = simulate_vasicek(
        vasicek,
        r0=sim.initial_rate,
        n_paths=sim.n_paths,
        n_steps=n_steps,
        dt=dt,
        rng=streams.rate,
        antithetic=sim.antithetic,
    )

    # Discount factors. Two conventions are supported:
    #   * "vasicek_simulated" — integrate the simulated short rate path
    #     (trapezoidal rule), so every path discounts at its own realised
    #     short rate. This is the most natural choice when the short rate
    #     and the property are correlated.
    #   * "constant_at_initial_rate" — use the deterministic schedule
    #     D(t_k) = exp(-r0 * t_k). The rate paths are still simulated and
    #     returned (for downstream use), but discounting ignores them.
    if sim.discount_convention == "constant_at_initial_rate":
        t_grid = np.arange(n_steps + 1, dtype=np.float64) * dt
        deterministic = np.exp(-sim.initial_rate * t_grid)
        discount_factors = np.broadcast_to(deterministic, (sim.n_paths, n_steps + 1)).copy()
    else:
        rate_increments = 0.5 * dt * (rate_paths[:, :-1] + rate_paths[:, 1:])
        cum_rate = np.concatenate(
            [np.zeros((sim.n_paths, 1)), np.cumsum(rate_increments, axis=1)], axis=1
        )
        discount_factors = np.exp(-cum_rate)

    # --- 2. Credit layer --------------------------------------------------- #
    default_times = _draw_default_times(sim, streams=streams)

    # --- 3. LGD layer ------------------------------------------------------ #
    lgd_samples = lgd_mod.sample(
        lgd, n_samples=sim.n_paths * sim.n_obligors, rng=streams.lgd
    ).reshape(sim.n_paths, sim.n_obligors)

    # --- 4. Cumulative loss path ------------------------------------------- #
    cumulative_loss = loss_paths.cumulative_loss_path(
        default_times=default_times,
        lgd_samples=lgd_samples,
        n_periods=n_steps,
        dt=dt,
    )

    # --- 5. Cash-flow construction ----------------------------------------- #
    surviving_fraction = 1.0 - cumulative_loss
    # Rent paid by the SURVIVING portfolio fraction.
    rent_per_period = surviving_fraction[:, :-1] * sim.gross_yield * price_paths[:, :-1] * dt
    maintenance_per_period = sim.maintenance_pct * price_paths[:, :-1] * dt
    net_rent = np.maximum(rent_per_period - maintenance_per_period, 0.0)

    # Terminal proceeds: surviving fraction times market price at horizon.
    terminal_value = surviving_fraction[:, -1] * price_paths[:, -1]

    # --- 6. Waterfall ------------------------------------------------------ #
    waterfall_out = andersen_sidenius.run(
        cumulative_loss=cumulative_loss,
        net_rent=net_rent,
        terminal_value=terminal_value,
        tranches=tranches,
        coupons=coupons,
        par=sim.par,
        dt=dt,
    )

    return SimulationOutput(
        price_paths=price_paths,
        rate_paths=rate_paths,
        discount_factors=discount_factors,
        default_times=default_times,
        lgd_samples=lgd_samples,
        cumulative_loss=cumulative_loss,
        net_rent=net_rent,
        terminal_value=terminal_value,
        waterfall=waterfall_out,
        config=sim,
    )


def _draw_default_times(
    sim: SimulationConfig,
    *,
    streams: seeds.StreamRNGs,
) -> NDArray[np.float64]:
    """Dispatch on the configured credit model."""
    if sim.credit_model == "gaussian_copula":
        if sim.use_qmc:
            return _gaussian_copula_qmc(sim, qmc_seed=int(streams.qmc.integers(2**30)))
        return gaussian_copula.simulate_default_times(
            gaussian_copula.GaussianCopulaParams(rho=sim.rho),
            horizon_years=sim.horizon_years,
            pd_terminal=sim.pd_terminal,
            n_sims=sim.n_paths,
            n_obligors=sim.n_obligors,
            rng=streams.credit,
            antithetic=sim.antithetic,
        )
    if sim.credit_model == "student_t_copula":
        return student_t_copula.simulate_default_times(
            student_t_copula.StudentTCopulaParams(rho=sim.rho, nu=sim.nu),
            horizon_years=sim.horizon_years,
            pd_terminal=sim.pd_terminal,
            n_sims=sim.n_paths,
            n_obligors=sim.n_obligors,
            rng=streams.credit,
        )
    if sim.credit_model == "cox_intensity":
        if sim.cox is None:
            # Construct sensible defaults consistent with pd_terminal.
            theta = 0.03
            beta = 0.8
            alpha = cox_intensity.calibrate_alpha_for_pd(
                pd_terminal=sim.pd_terminal,
                horizon_years=sim.horizon_years,
                beta=beta,
                theta=theta,
            )
            cox_params = cox_intensity.CoxIntensityParams(
                alpha=alpha, beta=beta, kappa=0.5, theta=theta, xi=0.04
            )
        else:
            cox_params = sim.cox
        return cox_intensity.simulate_default_times(
            cox_params,
            horizon_years=sim.horizon_years,
            pd_terminal=sim.pd_terminal,
            n_sims=sim.n_paths,
            n_obligors=sim.n_obligors,
            rng=streams.credit,
            steps_per_year=sim.steps_per_year,
        )
    raise ValueError(f"Unknown credit model: {sim.credit_model!r}")


def _gaussian_copula_qmc(
    sim: SimulationConfig,
    *,
    qmc_seed: int,
) -> NDArray[np.float64]:
    """QMC-driven Gaussian copula: Sobol-mapped Gaussians for the factor + idios."""
    sample = qmc.sobol_gaussian(
        n_paths=sim.n_paths,
        n_dimensions=sim.n_obligors + 1,
        seed=qmc_seed,
        scramble=True,
    )
    m = sample[:, 0]
    z = sample[:, 1:]
    sqrt_rho = float(np.sqrt(sim.rho))
    sqrt_1m = float(np.sqrt(1.0 - sim.rho))
    x = sqrt_rho * m[:, None] + sqrt_1m * z
    u = norm.cdf(x)
    return default_time_from_uniform(
        u, horizon_years=sim.horizon_years, pd_terminal=sim.pd_terminal
    )


__all__ = ["CreditModelName", "SimulationConfig", "SimulationOutput", "run_simulation"]
