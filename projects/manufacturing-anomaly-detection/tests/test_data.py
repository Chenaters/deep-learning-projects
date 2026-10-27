import unittest

from crowe_anomaly.config import LABEL_COLUMN
from crowe_anomaly.data import generate_manufacturing_data, split_by_time


class DataTests(unittest.TestCase):
    def test_generation_is_reproducible(self) -> None:
        first = generate_manufacturing_data(n_rows=500, seed=9)
        second = generate_manufacturing_data(n_rows=500, seed=9)
        self.assertTrue(first.equals(second))

    def test_split_is_chronological_and_complete(self) -> None:
        frame = generate_manufacturing_data(n_rows=500, seed=2)
        train, validation, test = split_by_time(frame)
        self.assertEqual((len(train), len(validation), len(test)), (300, 100, 100))
        self.assertLess(train.index.max(), validation.index.min())
        self.assertLess(validation.index.max(), test.index.min())
        self.assertGreater(int(frame[LABEL_COLUMN].sum()), 0)


if __name__ == "__main__":
    unittest.main()

