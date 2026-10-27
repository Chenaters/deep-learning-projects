"""Input validation and privacy controls for scoring data."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS

FORBIDDEN_PII_FIELDS = {
    "address",
    "date_of_birth",
    "dob",
    "email",
    "full_name",
    "name",
    "phone",
    "social_security_number",
    "ssn",
}


class SchemaError(ValueError):
    """Raised when scoring data violates the approved input contract."""


def validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return an ordered numeric feature frame or raise a clear error."""
    normalized_columns = {str(column).strip().lower() for column in frame.columns}
    pii_found = sorted(normalized_columns & FORBIDDEN_PII_FIELDS)
    if pii_found:
        raise SchemaError(
            "Potential PII fields are not accepted by this service: "
            + ", ".join(pii_found)
        )

    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise SchemaError("Missing required features: " + ", ".join(missing))

    ordered = frame.loc[:, FEATURE_COLUMNS].copy()
    for column in FEATURE_COLUMNS:
        ordered[column] = pd.to_numeric(ordered[column], errors="coerce")

    values = ordered.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise SchemaError("All feature values must be finite numbers.")
    return ordered


def validate_record(record: Mapping[str, object]) -> pd.DataFrame:
    """Validate a single API record and reject unexpected fields."""
    keys = {str(key) for key in record}
    normalized_keys = {key.strip().lower() for key in keys}
    pii_found = sorted(normalized_keys & FORBIDDEN_PII_FIELDS)
    if pii_found:
        raise SchemaError(
            "Potential PII fields are not accepted by this service: "
            + ", ".join(pii_found)
        )
    unexpected = sorted(keys - set(FEATURE_COLUMNS))
    if unexpected:
        raise SchemaError(
            "Unexpected fields are not accepted: " + ", ".join(unexpected)
        )
    return validate_frame(pd.DataFrame([dict(record)]))
