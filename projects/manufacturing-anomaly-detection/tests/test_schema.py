import unittest

import numpy as np
import pandas as pd

from crowe_anomaly.config import FEATURE_COLUMNS
from crowe_anomaly.schema import SchemaError, validate_frame, validate_record


def valid_record() -> dict[str, float]:
    return dict(
        zip(FEATURE_COLUMNS, [72.0, 2.2, 35.0, 12.5, 102.0], strict=True)
    )


class SchemaTests(unittest.TestCase):
    def test_valid_record_uses_feature_order(self) -> None:
        frame = validate_record(valid_record())
        self.assertEqual(tuple(frame.columns), FEATURE_COLUMNS)

    def test_rejects_pii(self) -> None:
        record = valid_record()
        record["email"] = "person@example.com"
        with self.assertRaisesRegex(SchemaError, "Potential PII"):
            validate_record(record)

    def test_rejects_non_finite_values(self) -> None:
        frame = pd.DataFrame([valid_record()])
        frame.loc[0, FEATURE_COLUMNS[0]] = np.nan
        with self.assertRaisesRegex(SchemaError, "finite"):
            validate_frame(frame)


if __name__ == "__main__":
    unittest.main()
