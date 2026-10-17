# Model Standard Comparison — Version 12 (2026-10-17)

Status: USER-AUTHORIZED 2026-10-17. Versioned, append-only report scientific contract.

- Contract ID: `model.standard_comparison@v12`.
- Supersedes: `model.standard_comparison@v11`; v11 remains historical.
- Model Evaluation Contract: `model.standard_sop@v8`.
- Sections: Learnability, Generalization, Upside / Downside Validation,
  Joint MFE / Adverse Quality, exactly in that order.
- Rows: model as comparison row; OOS / Breakout are separate scope tables.
  The standard single-model section schemas are the sole column owner; the
  comparison adds Model and excludes redundant Split / Comparison labels.
- Each model must satisfy the identical evaluation identity: matched OOS
  signal dates, mature target observation window, price truth, score-universe
  and expected seed / fold population. Otherwise cross-model comparison blocks.
- Joint Quality uses same-date true outcome percentage-ranks of MFE% and
  negative Adverse-to-MFE%; it is not realized PnL or a success probability.
- RUN, REUSE, Forward, Rolling, Robustness and human-readable output must use
  the same canonical scientific calculation and current report contract.
- A report schema or metric semantic change requires a new contract ID/version,
  preserving this file and its recorded SHA256 unchanged.
