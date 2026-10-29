import unittest

import pandas as pd

from backend.train_model import FEATURE_COLUMNS


class TrainingContractTests(unittest.TestCase):
    def test_features_do_not_leak_targets(self):
        self.assertNotIn("endterm_marks_45", FEATURE_COLUMNS)
        self.assertNotIn("total_marks_100", FEATURE_COLUMNS)
        self.assertNotIn("pass_status", FEATURE_COLUMNS)

    def test_existing_csv_schema(self):
        data = pd.read_csv("lpu_student_performance.csv", nrows=1)
        self.assertIn("endterm_marks_45", data.columns)
        self.assertIn("total_marks_100", data.columns)


if __name__ == "__main__":
    unittest.main()
