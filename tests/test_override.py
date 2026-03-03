# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for config override (base + override_*) functionality."""

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from srtctl.cli.submit import is_override_config, parse_config_arg, submit_override
from srtctl.core.config import deep_merge, generate_override_configs


# =============================================================================
# TestDeepMerge
# =============================================================================


class TestDeepMerge:
    def test_scalar_override(self):
        """Scalar fields are overridden."""
        base = {"name": "old", "resources": {"decode_nodes": 8}}
        override = {"resources": {"decode_nodes": 4}}
        result = deep_merge(base, override)
        assert result["resources"]["decode_nodes"] == 4
        assert result["name"] == "old"

    def test_list_replace(self):
        """Lists are fully replaced, not appended."""
        base = {"benchmark": {"concurrencies": [8192, 10240]}}
        override = {"benchmark": {"concurrencies": [4096]}}
        result = deep_merge(base, override)
        assert result["benchmark"]["concurrencies"] == [4096]

    def test_nested_dict_merge(self):
        """Nested dicts are recursively merged, preserving untouched keys."""
        base = {"backend": {"sglang_config": {"prefill": {"tp-size": 32, "trust-remote-code": True}}}}
        override = {"backend": {"sglang_config": {"prefill": {"tp-size": 64}}}}
        result = deep_merge(base, override)
        assert result["backend"]["sglang_config"]["prefill"]["tp-size"] == 64
        assert result["backend"]["sglang_config"]["prefill"]["trust-remote-code"] is True

    def test_null_deletes_key(self):
        """Setting a value to None deletes the key."""
        base = {"extra_mount": ["/data:/data"], "name": "test"}
        override = {"extra_mount": None}
        result = deep_merge(base, override)
        assert "extra_mount" not in result
        assert result["name"] == "test"

    def test_null_delete_missing_key_is_noop(self):
        """Deleting a non-existent key is a no-op."""
        base = {"name": "test"}
        override = {"nonexistent": None}
        result = deep_merge(base, override)
        assert result == {"name": "test"}

    def test_add_new_key(self):
        """Override can add keys not present in base."""
        base = {"name": "test"}
        override = {"environment": {"NEW_VAR": "value"}}
        result = deep_merge(base, override)
        assert result["environment"]["NEW_VAR"] == "value"
        assert result["name"] == "test"

    def test_base_not_mutated(self):
        """Deep merge does not mutate the original base dict."""
        base = {"resources": {"decode_nodes": 8}}
        override = {"resources": {"decode_nodes": 4}}
        deep_merge(base, override)
        assert base["resources"]["decode_nodes"] == 8

    def test_override_not_mutated(self):
        """Deep merge does not mutate the override dict."""
        base = {"resources": {"decode_nodes": 8}}
        override = {"resources": {"decode_nodes": 4, "extra": [1, 2, 3]}}
        result = deep_merge(base, override)
        result["resources"]["extra"].append(4)
        assert override["resources"]["extra"] == [1, 2, 3]


# =============================================================================
# TestParseConfigArg
# =============================================================================


class TestParseConfigArg:
    def test_plain_path(self):
        """Plain path without selector."""
        path, selector = parse_config_arg("config.yaml")
        assert path == Path("config.yaml")
        assert selector is None

    def test_path_with_override_selector(self):
        """Path with override selector."""
        path, selector = parse_config_arg("config.yaml:override_tp64")
        assert path == Path("config.yaml")
        assert selector == "override_tp64"

    def test_path_with_base_selector(self):
        """Path with base selector."""
        path, selector = parse_config_arg("recipes/test.yaml:base")
        assert path == Path("recipes/test.yaml")
        assert selector == "base"

    def test_invalid_selector_raises(self):
        """Invalid selector raises ValueError."""
        with pytest.raises(ValueError, match="Invalid selector"):
            parse_config_arg("config.yaml:foobar")

    def test_directory_path_no_selector(self):
        """Directory path without selector."""
        path, selector = parse_config_arg("./configs/")
        assert path == Path("./configs/")
        assert selector is None


# =============================================================================
# TestGenerateOverrideConfigs
# =============================================================================


class TestGenerateOverrideConfigs:
    def test_base_only(self):
        """Config with only base returns one config."""
        raw = {"base": {"name": "test", "resources": {"decode_nodes": 8}}}
        configs = generate_override_configs(raw)
        assert len(configs) == 1
        assert configs[0][0] == "base"
        assert configs[0][1]["name"] == "test"

    def test_with_overrides(self):
        """Base + 2 overrides returns 3 configs with correct names."""
        raw = {
            "base": {"name": "test", "resources": {"decode_nodes": 8}},
            "override_small": {"resources": {"decode_nodes": 4}},
            "override_large": {"resources": {"decode_nodes": 16}},
        }
        configs = generate_override_configs(raw)
        assert len(configs) == 3

        # base comes first, overrides sorted alphabetically
        assert configs[0][0] == "base"
        assert configs[0][1]["name"] == "test"

        assert configs[1][0] == "large"
        assert configs[1][1]["name"] == "test_large"
        assert configs[1][1]["resources"]["decode_nodes"] == 16

        assert configs[2][0] == "small"
        assert configs[2][1]["name"] == "test_small"
        assert configs[2][1]["resources"]["decode_nodes"] == 4

    def test_override_name_generation(self):
        """Override auto-generates name = base_name + suffix."""
        raw = {
            "base": {"name": "gb300-fp8"},
            "override_tp64": {"backend": {"sglang_config": {"prefill": {"tp-size": 64}}}},
        }
        configs = generate_override_configs(raw)
        assert configs[1][1]["name"] == "gb300-fp8_tp64"

    def test_deep_merge_applied(self):
        """Override fields are deep-merged with base."""
        raw = {
            "base": {
                "name": "test",
                "backend": {"sglang_config": {"prefill": {"tp-size": 32, "trust-remote-code": True}}},
            },
            "override_tp64": {"backend": {"sglang_config": {"prefill": {"tp-size": 64}}}},
        }
        configs = generate_override_configs(raw)
        merged = configs[1][1]
        assert merged["backend"]["sglang_config"]["prefill"]["tp-size"] == 64
        assert merged["backend"]["sglang_config"]["prefill"]["trust-remote-code"] is True

    def test_selector_base_only(self):
        """selector='base' returns only the base config."""
        raw = {
            "base": {"name": "test"},
            "override_a": {"name": "a"},
        }
        configs = generate_override_configs(raw, selector="base")
        assert len(configs) == 1
        assert configs[0][0] == "base"

    def test_selector_specific_override(self):
        """selector='override_a' returns only that variant."""
        raw = {
            "base": {"name": "test"},
            "override_a": {"resources": {"decode_nodes": 4}},
            "override_b": {"resources": {"decode_nodes": 16}},
        }
        configs = generate_override_configs(raw, selector="override_a")
        assert len(configs) == 1
        assert configs[0][0] == "a"
        assert configs[0][1]["name"] == "test_a"
        assert configs[0][1]["resources"]["decode_nodes"] == 4

    def test_selector_not_found_raises(self):
        """Selector for non-existent override raises ValueError."""
        raw = {"base": {"name": "test"}}
        with pytest.raises(ValueError, match="not found"):
            generate_override_configs(raw, selector="override_nope")

    def test_non_override_keys_ignored(self):
        """Keys that don't start with 'override_' (other than 'base') are ignored."""
        raw = {
            "base": {"name": "test"},
            "override_a": {"resources": {"decode_nodes": 4}},
            "some_other_key": {"foo": "bar"},
        }
        configs = generate_override_configs(raw)
        assert len(configs) == 2  # base + override_a, some_other_key ignored


# =============================================================================
# TestIsOverrideConfig
# =============================================================================


class TestIsOverrideConfig:
    def test_normal_config_not_detected(self, tmp_path):
        """Normal config (no 'base' key) is not detected as override."""
        config_file = tmp_path / "normal.yaml"
        config_file.write_text(yaml.dump({"name": "test", "resources": {"decode_nodes": 8}}))
        assert is_override_config(config_file) is False

    def test_override_config_detected(self, tmp_path):
        """Config with 'base' key is detected as override."""
        config_file = tmp_path / "override.yaml"
        config_file.write_text(
            yaml.dump({
                "base": {"name": "test", "resources": {"decode_nodes": 8}},
                "override_small": {"resources": {"decode_nodes": 4}},
            })
        )
        assert is_override_config(config_file) is True

    def test_base_only_config_detected(self, tmp_path):
        """Config with only 'base' key (no overrides) is still detected."""
        config_file = tmp_path / "base_only.yaml"
        config_file.write_text(yaml.dump({"base": {"name": "test"}}))
        assert is_override_config(config_file) is True

    def test_empty_file_not_detected(self, tmp_path):
        """Empty YAML file is not detected as override."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        assert is_override_config(config_file) is False


# =============================================================================
# TestSubmitOverrideE2E
# =============================================================================

# Minimal valid SrtConfig for testing
MINIMAL_CONFIG = {
    "name": "test-job",
    "model": {
        "path": "/models/test-model",
        "container": "test-container.sqsh",
        "precision": "fp8",
    },
    "resources": {
        "gpu_type": "h100",
        "gpus_per_node": 8,
        "prefill_nodes": 1,
        "decode_nodes": 1,
        "prefill_workers": 1,
        "decode_workers": 1,
    },
    "benchmark": {"type": "manual"},
}


class TestSubmitOverrideE2E:
    """Integration tests for submit_override with mocked SLURM."""

    def _write_override_config(self, tmp_path, overrides=None):
        """Write an override config YAML to tmp_path and return the path."""
        raw = {"base": MINIMAL_CONFIG.copy()}
        raw["base"] = {**MINIMAL_CONFIG}
        if overrides:
            raw.update(overrides)
        config_file = tmp_path / "override_test.yaml"
        config_file.write_text(yaml.dump(raw, default_flow_style=False))
        return config_file

    def test_dry_run_base_only(self, tmp_path, capsys):
        """Dry-run with base-only override config shows one variant."""
        config_file = self._write_override_config(tmp_path)

        with patch("srtctl.cli.submit.load_cluster_config", return_value=None):
            submit_override(config_file, dry_run=True)

        output = capsys.readouterr().out
        assert "1 variant" in output
        assert "test-job" in output

    def test_dry_run_with_overrides(self, tmp_path, capsys):
        """Dry-run with overrides shows correct number of variants."""
        config_file = self._write_override_config(
            tmp_path,
            overrides={
                "override_small": {"resources": {"decode_nodes": 2}},
                "override_large": {"resources": {"decode_nodes": 4}},
            },
        )

        with patch("srtctl.cli.submit.load_cluster_config", return_value=None):
            submit_override(config_file, dry_run=True)

        output = capsys.readouterr().out
        assert "3 variants" in output
        assert "test-job" in output
        assert "test-job_small" in output
        assert "test-job_large" in output

    def test_dry_run_with_selector(self, tmp_path, capsys):
        """Dry-run with selector shows only selected variant."""
        config_file = self._write_override_config(
            tmp_path,
            overrides={
                "override_small": {"resources": {"decode_nodes": 2}},
                "override_large": {"resources": {"decode_nodes": 4}},
            },
        )

        with patch("srtctl.cli.submit.load_cluster_config", return_value=None):
            submit_override(config_file, selector="override_small", dry_run=True)

        output = capsys.readouterr().out
        assert "1 variant" in output
        assert "override_small" in output
        assert "test-job_small" in output
        # Should NOT mention the large variant
        assert "test-job_large" not in output

    def test_submit_calls_sbatch_for_each_variant(self, tmp_path):
        """Real submit (non-dry-run) calls sbatch for each variant."""
        config_file = self._write_override_config(
            tmp_path,
            overrides={"override_small": {"resources": {"decode_nodes": 2}}},
        )

        mock_result = MagicMock()
        mock_result.stdout = "Submitted batch job 99999"
        mock_result.returncode = 0

        with (
            patch("srtctl.cli.submit.load_cluster_config", return_value=None),
            patch("subprocess.run", return_value=mock_result) as mock_sbatch,
            patch("srtctl.cli.submit.get_srtslurm_setting", return_value=None),
            patch("srtctl.cli.submit.create_job_record"),
        ):
            submit_override(config_file, output_dir=tmp_path)

        # Should have called sbatch twice (base + override_small)
        sbatch_calls = [c for c in mock_sbatch.call_args_list if c[0][0][0] == "sbatch"]
        assert len(sbatch_calls) == 2

    def test_selector_base_submits_one_job(self, tmp_path):
        """Selector 'base' submits exactly one job."""
        config_file = self._write_override_config(
            tmp_path,
            overrides={"override_small": {"resources": {"decode_nodes": 2}}},
        )

        mock_result = MagicMock()
        mock_result.stdout = "Submitted batch job 99999"
        mock_result.returncode = 0

        with (
            patch("srtctl.cli.submit.load_cluster_config", return_value=None),
            patch("subprocess.run", return_value=mock_result) as mock_sbatch,
            patch("srtctl.cli.submit.get_srtslurm_setting", return_value=None),
            patch("srtctl.cli.submit.create_job_record"),
        ):
            submit_override(config_file, selector="base", output_dir=tmp_path)

        sbatch_calls = [c for c in mock_sbatch.call_args_list if c[0][0][0] == "sbatch"]
        assert len(sbatch_calls) == 1
