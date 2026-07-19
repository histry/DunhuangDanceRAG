import unittest

try:
    from tools.v46_51_split_retarget_cache import (
        exact_split_counts,
        performer_capacities,
    )
except ImportError:
    from data_pipeline.split_sources import (
        exact_split_counts,
        performer_capacities,
    )


class ResearchPerformerSplitTest(unittest.TestCase):
    def test_four_female_eight_male(self):
        records = []
        for index in range(4):
            records.append({
                "source_uid": "female_%d" % index,
                "performer_group": "female",
                "dance_key": "female_cat_%d" % (index % 2),
            })
        for index in range(8):
            records.append({
                "source_uid": "male_%d" % index,
                "performer_group": "male",
                "dance_key": "male_cat_%d" % (index % 4),
            })
        target = exact_split_counts(12, 0.67, 0.165, 0.165)
        self.assertEqual(target, {"train": 8, "val": 2, "test": 2})
        capacities = performer_capacities(records, target)
        self.assertEqual(capacities["female"], {"train": 2, "val": 1, "test": 1})
        self.assertEqual(capacities["male"], {"train": 6, "val": 1, "test": 1})


if __name__ == "__main__":
    unittest.main()
