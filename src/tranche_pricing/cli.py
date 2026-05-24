"""Command-line entry point.

The library exposes one console script, ``tranche-cli``, that drives the
end-to-end pipeline through a small set of subcommands. Each subcommand is a
thin wrapper around library functions so behaviour is identical when invoked
from a notebook, a Makefile target, the dashboard or the CLI.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__
from .config import Config, load_config

logger = logging.getLogger("tranche_pricing")


# --------------------------------------------------------------------------- #
# Subcommand handlers — keep them as light as possible.                       #
# --------------------------------------------------------------------------- #


def cmd_data(cfg: Config, args: argparse.Namespace) -> int:
    """Download / refresh every external data series listed in the config."""
    from .data import SERIES
    from .data._http import UpstreamError

    start = cfg.data_window.start
    end = cfg.data_window.end
    refresh = bool(getattr(args, "refresh", True))

    errors: list[tuple[str, str]] = []
    for name, fetch_fn in SERIES.items():
        try:
            df = fetch_fn(start=start, end=end, refresh=refresh)
            logger.info("%-22s ok (%d rows, %s → %s)", name, len(df), start, end)
        except UpstreamError as exc:
            logger.warning("%-22s upstream error: %s", name, exc)
            errors.append((name, str(exc)))
        except FileNotFoundError as exc:
            logger.warning("%-22s no snapshot yet (%s)", name, exc)
            errors.append((name, "no snapshot — see data/DATA_SOURCES.md"))

    if errors:
        logger.warning("%d series could not be refreshed.", len(errors))
        return 1
    return 0


def cmd_calibrate(cfg: Config, args: argparse.Namespace) -> int:
    """Fit GBM / Merton / Vasicek / copulas / hazards from processed data."""
    from .calibration import runner

    del cfg, args
    payload = runner.run_all()
    runner.persist(payload)
    gbm = payload["gbm_paris"]["params"]
    vas = payload["vasicek_oat_10y"]["params"]
    mer = payload["merton_paris"]["params"]
    logger.info(
        "GBM Paris:    mu=%.4f sigma=%.4f  (n=%d, LL=%.2f, AIC=%.2f)",
        gbm["mu"],
        gbm["sigma"],
        payload["gbm_paris"]["n_obs"],
        payload["gbm_paris"]["log_likelihood"],
        payload["gbm_paris"]["aic"],
    )
    logger.info(
        "Merton Paris: mu=%.4f sigma=%.4f lam=%.3f mu_J=%.4f sig_J=%.4f  (AIC=%.2f)",
        mer["mu"],
        mer["sigma"],
        mer["lam"],
        mer["mu_jump"],
        mer["sigma_jump"],
        payload["merton_paris"]["aic"],
    )
    logger.info(
        "Vasicek OAT:  a=%.4f b=%.4f sigma_r=%.4f  (AIC=%.2f)",
        vas["a"],
        vas["b"],
        vas["sigma_r"],
        payload["vasicek_oat_10y"]["aic"],
    )
    return 0


def cmd_mc(cfg: Config, args: argparse.Namespace) -> int:
    """Run the Monte Carlo pricing pipeline and write ``artifacts/results.csv``."""
    del args
    from .pricing import runner as pricing_runner

    path = pricing_runner.run(cfg)
    logger.info("Pipeline complete; results at %s", path)
    return 0


def cmd_figures(cfg: Config, args: argparse.Namespace) -> int:
    """Regenerate every headline figure from the latest artifacts."""
    from pathlib import Path

    from .data import notaires
    from .viz import figures

    outdir = Path("artifacts/figures")
    outdir.mkdir(parents=True, exist_ok=True)

    notaires_df = notaires.fetch(start=cfg.data_window.start, end=cfg.data_window.end)
    fig = figures.fig_paris_price_index(notaires_df)
    out_pdf = outdir / "fig_paris_price_index.pdf"
    out_png = outdir / "fig_paris_price_index.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    logger.info("Wrote %s", out_pdf)
    return 0


HANDLERS = {
    "data": cmd_data,
    "calibrate": cmd_calibrate,
    "mc": cmd_mc,
    "figures": cmd_figures,
}


# --------------------------------------------------------------------------- #
# Parser                                                                      #
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tranche-cli",
        description="Tranche pricing on Paris residential rental cash flows.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (-v = INFO, -vv = DEBUG).",
    )

    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    for name in HANDLERS:
        sp = sub.add_parser(name, help=HANDLERS[name].__doc__)
        sp.add_argument(
            "-c",
            "--config",
            type=Path,
            required=True,
            help="Path to a scenario YAML (e.g. config/paris_intermediate.yaml).",
        )
        if name == "mc":
            sp.add_argument(
                "--n-sims",
                type=int,
                default=None,
                help="Override n_sims from the YAML (useful for smoke tests).",
            )
        if name == "data":
            sp.add_argument(
                "--no-refresh",
                dest="refresh",
                action="store_false",
                help="Use local snapshots only, do not hit upstream APIs.",
            )
            sp.set_defaults(refresh=True)

    return parser


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    cfg = load_config(args.config)
    if getattr(args, "n_sims", None):
        cfg = cfg.model_copy(
            update={"monte_carlo": cfg.monte_carlo.model_copy(update={"n_sims": args.n_sims})}
        )

    logger.info("Running %r on scenario %r", args.command, cfg.scenario.name)
    return HANDLERS[args.command](cfg, args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
