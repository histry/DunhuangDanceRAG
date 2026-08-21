#!/usr/bin/env python3
import unittest

from events.build_database import validate_retarget_contract


class SourceAwareEventContractTest(unittest.TestCase):
    def _report(self):
        return {
            "schema": "chang_e_official_smpl_source_aware_preprocess_v1",
            "version": "chang_e_official_smpl_source_aware_preprocess_event_geometry_2",
            "ok": True,
            "source_gate_ok": True,
            "source_preprocess_ok": True,
            "retargeting_applied": False,
            "source_preprocess_contract": {
                "schema": "chang_e_official_smpl_source_aware_preprocess_v1",
                "direct_official_smpl": True,
                "retargeting_applied": False,
                "event_quality_gate_deferred": True,
            },
            "preprocess_segment": {
                "clean": True,
                "segment_index": 2,
                "source_start_seconds": 10.0,
                "source_end_seconds": 15.0,
            },
        }

    def test_official_source_aware_contract_is_accepted(self):
        ok, reasons = validate_retarget_contract(None, self._report())
        self.assertTrue(ok, reasons)

    def test_failed_source_gate_is_rejected(self):
        report = self._report()
        report["source_gate_ok"] = False
        ok, reasons = validate_retarget_contract(None, report)
        self.assertFalse(ok)
        self.assertIn("source_aware_source_gate_not_ok", reasons)


if __name__ == "__main__":
    unittest.main()
