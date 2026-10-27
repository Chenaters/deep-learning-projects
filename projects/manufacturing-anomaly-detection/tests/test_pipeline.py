import tempfile
import unittest
import warnings
from pathlib import Path

import joblib
from sklearn.exceptions import ConvergenceWarning

from crowe_anomaly.config import FEATURE_COLUMNS
from crowe_anomaly.service import AnomalyScorer
from crowe_anomaly.train import train_pipeline


class PipelineTests(unittest.TestCase):
    def test_train_package_and_score(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with warnings.catch_warnings():
                # The deliberately tiny smoke-test dataset may hit its iteration cap;
                # the full integration run uses 6,000 rows and converges normally.
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", DeprecationWarning)
                metrics = train_pipeline(
                    data_dir=root / "data",
                    output_dir=root / "artifacts",
                    rows=600,
                    seed=5,
                    max_iter=150,
                    architectures=((8, 3, 8),),
                )
            model_path = root / "artifacts" / "model.joblib"
            self.assertTrue(model_path.exists())
            self.assertGreaterEqual(metrics["autoencoder_test"]["recall"], 0.5)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                bundle = joblib.load(model_path)
                scorer = AnomalyScorer(model_path)
            self.assertEqual(tuple(bundle["features"]), FEATURE_COLUMNS)
            result = scorer.score(
                dict(
                    zip(
                        FEATURE_COLUMNS,
                        [72.0, 2.1, 35.0, 12.5, 102.0],
                        strict=True,
                    )
                )
            )
            self.assertIn("anomaly_score", result)
            self.assertIsInstance(result["predicted_anomaly"], bool)


if __name__ == "__main__":
    unittest.main()
