"""Generate audience-specific model summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_stakeholder_summary(metrics: dict[str, Any], output_path: Path) -> None:
    test = metrics["autoencoder_test"]
    baseline = metrics["baseline_test"]
    selected = metrics["selected_candidate"]
    text = f"""# Manufacturing Anomaly Pilot - Stakeholder Brief

## Decision

Proceed with a limited shadow-mode pilot of `{selected}`. Do not automate maintenance shutdowns from this model.

## Business objective

Identify unusual equipment behavior early enough for an operator to investigate, while keeping false alerts low enough that the system remains usable.

## Test results

- Autoencoder precision: {test['precision']:.1%}
- Autoencoder recall: {test['recall']:.1%}
- Autoencoder F1: {test['f1']:.1%}
- False positives: {test['false_positives']}
- Missed anomalies: {test['false_negatives']}
- Isolation Forest baseline F1: {baseline['f1']:.1%}

## Controls and caveats

- The data are synthetic and demonstrate workflow quality, not production performance.
- The service accepts only five approved numeric sensor fields and rejects common PII fields.
- Alerts require human review; the score is not a safety decision.
- Production use would require site-specific validation, drift monitoring, access controls, and a rollback plan.

## Next experiment

Run a four-week shadow test on de-identified historical telemetry, review false alerts with an engineer, and set the threshold from the cost of misses versus unnecessary inspections.
"""
    output_path.write_text(text, encoding="utf-8")


def write_metrics(metrics: dict[str, Any], output_path: Path) -> None:
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

