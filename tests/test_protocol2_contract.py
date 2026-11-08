"""Contract tests for the conference-paper version of Protocol 2."""

from __future__ import annotations

from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _yaml(name: str) -> dict:
    with (ROOT / "configs" / name).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _combination_set(rule: dict) -> set[tuple[str, ...]]:
    return {tuple(sorted(map(str, value))) for value in rule["combinations"]}


class Protocol2ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = _yaml("data.yaml")
        cls.stage5 = _yaml("stage5.yaml")
        cls.protocol = cls.data["protocols"]["protocol_2"]

    def test_conference_matrix_is_exact(self) -> None:
        self.assertEqual(self.data["experiment"]["seeds"], [0, 1, 2, 3, 4])
        self.assertEqual(
            self.stage5["protocols"]["protocol_2"]["main_models"],
            ["hgt", "hgt_learnable_quality", "hgt_consistency", "qc_hgt"],
        )
        self.assertFalse(self.stage5["run"]["include_ablations"])
        self.assertFalse(self.stage5["run"]["include_sensitivity_in_protocol2"])

    def test_seen_degradation_matches_training_distribution(self) -> None:
        train = self.protocol["splits"]["train"]
        validation = self.protocol["splits"]["val"]
        seen = self.protocol["test_scenarios"]["known"]
        for rule in (train, validation):
            self.assertEqual(rule["mode"], "sample")
            self.assertEqual(float(rule["probability"]), 0.8)
            self.assertEqual(_combination_set(rule), _combination_set(train))
            self.assertEqual(rule["levels"], train["levels"])
        self.assertEqual(float(seen["probability"]), 1.0)
        self.assertEqual(_combination_set(seen), _combination_set(train))
        self.assertEqual(seen["levels"], train["levels"])

    def test_unseen_combinations_are_novel_but_strength_matched(self) -> None:
        train = self.protocol["splits"]["train"]
        unseen = self.protocol["test_scenarios"]["unseen_combination"]
        self.assertTrue(_combination_set(unseen))
        self.assertTrue(_combination_set(train).isdisjoint(_combination_set(unseen)))
        self.assertEqual(unseen["levels"], train["levels"])
        self.assertEqual(float(unseen["probability"]), 1.0)

    def test_run_first_data_and_inference_quality_contract(self) -> None:
        dataset = self.data["dataset"]
        variant = dataset["variants"][dataset["variant"]]
        self.assertEqual(dataset["variant"], dataset["paper_main_variant"])
        self.assertEqual(variant["num_classes"], 21)
        self.assertEqual(dataset["num_variables"], 52)
        self.assertTrue(dataset["split"]["fail_on_leakage"])
        self.assertEqual(self.data["quality"]["attach_target"], {"train": True, "val": False, "test": False})
        self.assertEqual(self.protocol["splits"]["test"]["mode"], "clean")

    def test_runtime_and_table_contract(self) -> None:
        overrides = self.stage5["formal_overrides"]
        self.assertEqual(overrides["runtime"]["device"], "cuda")
        self.assertTrue(overrides["runtime"]["deterministic"])
        self.assertFalse(overrides["runtime"]["deterministic_warn_only"])
        self.assertEqual(
            [row[1] for row in self.stage5["summary"]["ablation_rows"]],
            ["qc_hgt", "hgt", "hgt_learnable_quality", "hgt_consistency"],
        )


if __name__ == "__main__":
    unittest.main()


# Design note (paper Sec. 5.1, 5.4, Table II): this locks five-seed mixed
# training, matched-strength novel combinations, run-safe data, and hidden
# quality targets at evaluation so future configuration edits cannot drift.
