"""Fast, offline checks for the refactored package."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from graphreason_bot import llm
from graphreason_bot import evaluation
from graphreason_bot import prepare_cohort
from graphreason_bot.baselines import seeded_data
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


class ProviderConfigurationTests(unittest.TestCase):
    @patch("openai.OpenAI")
    def test_deepseek_uses_its_own_credentials(self, client_class) -> None:
        with (
            patch.object(llm, "DEEPSEEK_API_KEY", "x"),
            patch.object(llm, "DEEPSEEK_BASE_URL", "https://example.invalid"),
            patch.object(llm, "DEEPSEEK_MODEL", "deepseek-test"),
        ):
            _, model = llm._get_client("DeepSeek")
        client_class.assert_called_once_with(
            api_key="x",
            base_url="https://example.invalid",
            timeout=llm.LLM_TIMEOUT,
        )
        self.assertEqual(model, "deepseek-test")

    def test_deepseek_validation_requires_deepseek_key(self) -> None:
        with (
            patch.object(evaluation.cfg, "DEEPSEEK_API_KEY", None),
            self.assertRaisesRegex(ValueError, "DEEPSEEK_API_KEY"),
        ):
            evaluation._validate_model_configuration("DeepSeek")

    def test_deepseek_requests_use_reasoning_helper(self) -> None:
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="bot"))]
        )
        client = object()
        with (
            patch.object(
                llm, "_get_client", return_value=(client, "deepseek-test")
            ),
            patch.object(llm, "_call_deepseek", return_value=response) as call,
        ):
            answer = llm._call_llm("prompt", "DeepSeek")
        call.assert_called_once_with(client, "deepseek-test", "prompt")
        self.assertEqual(answer, "bot")

    def test_deepseek_helper_applies_reasoning_settings(self) -> None:
        recorded = {}

        def create(
            model,
            messages,
            stream,
            reasoning_effort=None,
            extra_body=None,
        ):
            recorded.update(
                model=model,
                messages=messages,
                stream=stream,
                reasoning_effort=reasoning_effort,
                extra_body=extra_body,
            )
            return object()

        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)
            )
        )
        with (
            patch.object(llm, "DEEPSEEK_REASONING_EFFORT", "high"),
            patch.object(llm, "DEEPSEEK_THINKING_ENABLED", True),
        ):
            llm._call_deepseek(client, "deepseek-test", "prompt")
        self.assertEqual(recorded["reasoning_effort"], "high")
        self.assertEqual(
            recorded["extra_body"], {"thinking": {"type": "enabled"}}
        )


class CohortPreparationTests(unittest.TestCase):
    def test_private_cohort_is_balanced_and_written_below_data_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            processed = data_root / "BotSim-24" / "processed"
            processed.mkdir(parents=True)
            records = [
                {"ID": "b1", "label": "bot"},
                {"ID": "b2", "label": "bot"},
                {"ID": "h1", "label": "human"},
                {"ID": "h2", "label": "human"},
            ]
            (processed / "train.json").write_text(
                json.dumps(records), encoding="utf-8"
            )
            (processed / "test.json").write_text("[]", encoding="utf-8")
            with (
                patch.object(seeded_data, "DATA_ROOT", data_root),
                patch.object(prepare_cohort, "DATA_ROOT", data_root),
                redirect_stdout(io.StringIO()),
            ):
                output = prepare_cohort.write_cohort(
                    dataset="botsim-24",
                    sample_seed=2026,
                    sample_size=4,
                    balanced=True,
                    overwrite=False,
                )
            written = json.loads(output.read_text(encoding="utf-8"))
            labels = [item["label"] for item in written]
            self.assertEqual(output, data_root / "BotSim-24" / "selected_4.json")
            self.assertEqual(labels.count("bot"), 2)
            self.assertEqual(labels.count("human"), 2)

    def test_twibot_top_level_labels_follow_documented_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            community = data_root / "Twibot-22" / "community_format"
            community.mkdir(parents=True)
            records = [
                {"target_id": "b1", "label": "bot", "following": {}},
                {"target_id": "b2", "label": "bot", "following": {}},
                {"target_id": "h1", "label": "human", "following": {}},
                {"target_id": "h2", "label": "human", "following": {}},
            ]
            (community / "test.json").write_text(
                json.dumps(records), encoding="utf-8"
            )
            with (
                patch.object(seeded_data, "DATA_ROOT", data_root),
                patch.object(prepare_cohort, "DATA_ROOT", data_root),
                redirect_stdout(io.StringIO()),
            ):
                output = prepare_cohort.write_cohort(
                    dataset="twibot-22",
                    sample_seed=2026,
                    sample_size=4,
                    balanced=True,
                    overwrite=False,
                )
            written = json.loads(output.read_text(encoding="utf-8"))
            labels = [item["label"] for item in written]
            self.assertEqual(labels.count("bot"), 2)
            self.assertEqual(labels.count("human"), 2)


if __name__ == "__main__":
    unittest.main()
