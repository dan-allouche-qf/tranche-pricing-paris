# Data sources

Every external series ingested by `make data` is listed below with its
upstream identifier, the project module that fetches it, and a short note on
its use in the pipeline. Retrieval dates are recorded in
`data/raw/_provenance.json` (populated by the `cache.py` wrapper) at every
run.

| Series | Frequency | Source | Module | Used by |
|---|---|---|---|---|
| Paris residential price index (Notaires de France / INSEE) | Quarterly | INSEE SDMX series `010567013` | `data.notaires` | GBM / Merton calibration; price simulation; recession-shaded plot. |
| Indice de Référence des Loyers (IRL) | Quarterly | INSEE SDMX series `001515333` | `data.insee_irl` | Rent indexation; vacancy / arrears proxy. |
| France ILO unemployment rate (CVS) | Quarterly | INSEE SDMX series `001688526` | `data.insee_unemployment` | Cox doubly-stochastic macro factor calibration (CIR MLE). |
| OAT 10Y nominal yield | Monthly | FRED `IRLTLT01FRM156N` (cross-checked vs Banque de France Webstat) | `data.oat` | Vasicek calibration; discounting. |
| ECB AAA euro-area yield curve (fallback) | Daily | ECB SDW | `data.ecb` | Sanity check vs OAT 10Y; secondary discounting. |
| Visale / ANIL rental-arrears report | Annual (manual snapshot) | Action Logement annual reports + ANIL barometer (PDF) | `data.visale` | Default-rate prior; LGD calibration (eviction duration). |
| Case-Shiller US national index | Monthly | FRED `CSUSHPISA` | `data.fred` | Cross-validation of GBM / Merton calibration. |
| Kenneth French Europe factors | Monthly | Ken French Library | `data.fama_french` | Risk-premia comparison plot. |

The cache layer (`data.cache`) hashes each request by `(URL, start, end)` and
stores the resulting payload under `data/.cache/`. Re-running `make data`
re-validates the cache and refreshes only the series whose upstream signature
has changed.

## Visale snapshot — manual extraction protocol

The Visale series is not exposed programmatically. The committed
`data/raw/visale.csv` contains the annual rental-arrears rate (decimal
fraction of insured contracts in default) for 2018–2024, sourced from:

1. The annual reports of **Action Logement** (operator of the Visale
   guarantee), available at
   [actionlogement.fr](https://www.actionlogement.fr/). Search for
   "rapport annuel" + the relevant year.
2. The **ANIL** (Agence nationale pour l'information sur le logement)
   private-rental arrears barometer, published yearly on
   [anil.org](https://www.anil.org/).
3. Banque de France household debt indicators for cross-validation
   (`data-explorer.banque-france.fr`).

To refresh the snapshot, download the most recent Action Logement annual
report (PDF), locate the section on default claims (`sinistralité`), and
update `data/raw/visale.csv` with the headline percentage divided by 100.
The schema is `date,value` with `date` set to the year-end. Commit the
updated CSV.

The values currently committed correspond to the public figures Action
Logement has communicated since the 2018 broadening of Visale to the
mainstream rental market. They land in the 2.9 %–3.7 % band typical of
the French private rental sector, with the 2020 spike attributable to
the COVID-19 moratorium effects.
