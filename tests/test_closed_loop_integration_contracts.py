#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def function_node(path: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(source(path), filename=path)
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise AssertionError(f"{path}: expected one {name}, found {len(matches)}")
    return matches[0]


def keyword_only_names(node: ast.FunctionDef) -> set[str]:
    return {argument.arg for argument in node.args.kwonlyargs}


def calls_named(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        value
        for value in ast.walk(node)
        if isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == name
    ]


def calls_attribute(node: ast.AST, attribute: str) -> list[ast.Call]:
    return [
        value
        for value in ast.walk(node)
        if isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == attribute
    ]


def keyword_names(call: ast.Call) -> set[str]:
    return {keyword.arg for keyword in call.keywords if keyword.arg is not None}


class ClosedLoopIntegrationContracts(unittest.TestCase):
    def test_research_entrypoint_installs_real_global_path_patch(self):
        text = source("routing/research_closed_loop.py")
        self.assertIn("latest._install_global_path_patches()", text)
        self.assertNotIn("latest._install_global_route_patches()", text)

    def test_sliding_support_is_keyword_only_across_wrapper_chain(self):
        targets = (
            ("routing/boundary_closed_loop.py", "apply_generators"),
            ("routing/heading_closed_loop.py", "apply_generators_with_heading_guard"),
            ("routing/anatomy_heading_closed_loop.py", "_apply_heading_generators"),
            ("routing/global_path.py", "apply_global_route_guard"),
        )
        for path, name in targets:
            with self.subTest(path=path, function=name):
                node = function_node(path, name)
                self.assertIn("sliding_support_eligible", keyword_only_names(node))

    def test_every_authoritative_wrapper_audit_preserves_sliding_support(self):
        targets = (
            ("routing/boundary_closed_loop.py", "apply_generators"),
            ("routing/heading_closed_loop.py", "apply_generators_with_heading_guard"),
            ("routing/anatomy_heading_closed_loop.py", "_apply_heading_generators"),
            ("routing/global_path.py", "apply_global_route_guard"),
        )
        for path, name in targets:
            with self.subTest(path=path, function=name):
                node = function_node(path, name)
                audits = calls_attribute(node, "audit_motion_np")
                self.assertTrue(audits, f"{path}:{name} has no audit_motion_np call")
                for call in audits:
                    self.assertIn(
                        "sliding_support_eligible",
                        keyword_names(call),
                        f"{path}:{name} has an audit that drops sliding semantics",
                    )

    def test_nested_wrappers_forward_sliding_support_to_previous_layer(self):
        anatomy = function_node(
            "routing/anatomy_heading_closed_loop.py", "_apply_heading_generators"
        )
        anatomy_calls = calls_named(anatomy, "_ORIG_APPLY")
        self.assertEqual(len(anatomy_calls), 1)
        self.assertIn("sliding_support_eligible", keyword_names(anatomy_calls[0]))

        global_guard = function_node("routing/global_path.py", "apply_global_route_guard")
        global_calls = calls_named(global_guard, "original_apply")
        self.assertEqual(len(global_calls), 1)
        self.assertIn("sliding_support_eligible", keyword_names(global_calls[0]))

    def test_closed_loop_conditioning_uses_authoritative_frame_local_builder(self):
        node = function_node("routing/boundary_closed_loop.py", "compute_condition")
        self.assertFalse(
            calls_attribute(node, "mean"),
            "compute_condition must not average slot descriptors",
        )
        builders = calls_attribute(node, "build_frame_local_conditioning")
        self.assertEqual(len(builders), 1)

    def test_conditioning_is_rebuilt_inside_each_reselection_round(self):
        node = function_node("routing/boundary_closed_loop.py", "generate_closed_loop")
        loops = [value for value in ast.walk(node) if isinstance(value, ast.For)]
        matching = []
        for loop in loops:
            assembly = calls_named(loop, "assemble_closed_loop_reference")
            conditioning = calls_named(loop, "compute_condition")
            if assembly and conditioning:
                matching.append((assembly, conditioning))
        self.assertEqual(len(matching), 1)
        assembly, conditioning = matching[0]
        self.assertLess(assembly[0].lineno, conditioning[0].lineno)
        self.assertEqual(len(conditioning), 1)

    def test_closed_loop_conditioning_report_disables_whole_song_mean(self):
        text = source("routing/boundary_closed_loop.py")
        self.assertIn('"mode": "frame_local_slot_conditioning"', text)
        self.assertIn('"whole_song_mean_conditioning": False', text)
        self.assertIn('"recomputed_after_each_reselection": True', text)


if __name__ == "__main__":
    unittest.main()
