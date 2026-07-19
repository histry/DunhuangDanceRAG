import unittest

from data_pipeline.split_sources import assign_sources, exact_split_counts


class ExactSourceSplitTests(unittest.TestCase):
    def test_nonempty_exact_counts(self):
        self.assertEqual(exact_split_counts(5, 0.8, 0.1, 0.1), {"train": 3, "val": 1, "test": 1})
        self.assertEqual(exact_split_counts(6, 0.67, 0.165, 0.165), {"train": 4, "val": 1, "test": 1})
        self.assertEqual(exact_split_counts(12, 0.67, 0.165, 0.165), {"train": 8, "val": 2, "test": 2})

    def test_assignment_is_disjoint_and_exact(self):
        labels = {f"source_{i}": ("pose" if i < 5 else "instrument") for i in range(12)}
        a = assign_sources(labels, seed=123, train_ratio=0.67, val_ratio=0.165, test_ratio=0.165)
        self.assertEqual(len(a), 12)
        self.assertEqual(sum(v == "train" for v in a.values()), 8)
        self.assertEqual(sum(v == "val" for v in a.values()), 2)
        self.assertEqual(sum(v == "test" for v in a.values()), 2)
        self.assertEqual(a, assign_sources(labels, seed=123, train_ratio=0.67, val_ratio=0.165, test_ratio=0.165))


if __name__ == "__main__":
    unittest.main()
