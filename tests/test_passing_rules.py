import unittest

from backend.passing_rules import is_passing, minimum_ete_required, projected_outcome


class PassingRuleTests(unittest.TestCase):
    def test_examples(self):
        self.assertEqual(minimum_ete_required(20, 15)["minimum_ete_marks"], 13.5)
        self.assertEqual(minimum_ete_required(10, 10)["minimum_ete_marks"], 20)
        self.assertEqual(minimum_ete_required(25, 10)["minimum_ete_marks"], 13.5)

    def test_whole_number_rounds_up_to_fourteen(self):
        self.assertEqual(minimum_ete_required(25, 10, whole_number=True)["minimum_ete_marks"], 14)

    def test_zero_marks_still_allows_maximum_ete(self):
        result = minimum_ete_required(0, 0)
        self.assertTrue(result["passing_possible"])
        self.assertEqual(result["maximum_possible_total"], 45)
        self.assertEqual(result["minimum_ete_marks"], 40)

    def test_exact_thresholds(self):
        self.assertTrue(is_passing(40, 13.5))
        self.assertFalse(is_passing(39.99, 13.5))
        self.assertFalse(is_passing(40, 13.49))

    def test_gap_is_reported_from_prediction(self):
        result = projected_outcome(10, 10, 15)
        self.assertEqual(result["projected_status"], "Fail")
        self.assertEqual(result["ete_gap"], -5)
        self.assertTrue(result["passing_possible"])
