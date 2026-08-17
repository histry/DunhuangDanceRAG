import unittest

from support.legacy_compatibility import (
    has_scheduler_provenance,
    normalize_motion_checkpoint_role,
    normalize_motion_checkpoint_version,
    normalize_transition_architecture,
)


class LegacyCompatibilityTests(unittest.TestCase):
    def test_motion_checkpoint_roles_are_translated_at_read_time(self):
        self.assertEqual(
            "boundary_refiner",
            normalize_motion_checkpoint_role("v45_refiner"),
        )
        self.assertEqual(
            "motion_diffusion",
            normalize_motion_checkpoint_role("motion_diffusion"),
        )

    def test_motion_checkpoint_version_suffix_is_preserved(self):
        self.assertEqual(
            "product_manifold_boundary_refiner_v1",
            normalize_motion_checkpoint_version("v45_product_manifold_79d_v1"),
        )

    def test_transition_architecture_is_translated(self):
        self.assertEqual(
            "boundary_transition_continuous_c3_contact_inr_latent_diffusion",
            normalize_transition_architecture(
                "v34_continuous_c3_contact_inr_latent_diffusion"
            ),
        )

    def test_scheduler_provenance_accepts_semantic_and_historical_assets(self):
        self.assertTrue(has_scheduler_provenance("music_router_whole_song_planner"))
        self.assertTrue(has_scheduler_provenance("v26_schedule"))
        self.assertFalse(has_scheduler_provenance("external_sidecar"))


if __name__ == "__main__":
    unittest.main()
