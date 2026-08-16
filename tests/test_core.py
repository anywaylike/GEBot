"""Fast, offline checks for the refactored package."""

from __future__ import annotations

import unittest

from graphreason_bot.metrics import calculate_metrics_report
from graphreason_bot.prompt import build_prompt
from graphreason_bot.se_hci import SEHCI


class MetricsTests(unittest.TestCase):
    def test_unknown_predictions_are_counted_as_incorrect(self) -> None:
        report = calculate_metrics_report(
            ["bot", "human"], ["unknown", "human"]
        )
        self.assertEqual(report["metrics"]["accuracy"], 0.5)
        self.assertEqual(report["valid_only_metrics"]["accuracy"], 1.0)
        self.assertEqual(report["unknown_rate"], 0.5)


class PromptTests(unittest.TestCase):
    def test_disabled_modules_are_not_claimed_in_the_prompt(self) -> None:
        prompt = build_prompt(
            "target",
            {"description": "target description"},
            ["neighbor"],
            {"neighbor"},
            set(),
            {"neighbor": {"description": "neighbor description"}},
            use_wgse=False,
            use_sehci=False,
            use_prkns=False,
        )
        self.assertIn("raw one-hop context", prompt)
        self.assertNotIn("PageRank-based", prompt)
        self.assertNotIn("structural-entropy", prompt)


class SEHCITests(unittest.TestCase):
    def test_height_override_is_honored(self) -> None:
        model = SEHCI(h=5)
        edges = {(0, 1), (1, 2), (2, 3), (3, 4)}
        root, _ = model.build_coding_tree(5, edges, target_idx=0, h=2)
        self.assertLessEqual(model._tree_height(root), 2)


if __name__ == "__main__":
    unittest.main()
