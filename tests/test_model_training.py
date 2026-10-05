from pathlib import Path

import pytest
import torch

from roofseg.model import EfficientNetB0UNet, build_model, state_digest
from roofseg.optimization import (
    build_late_lr_drop_config,
    build_learning_rate_config,
    build_training_horizon_config,
    late_lr_drop_run_directory,
    learning_rate_run_directory,
    optimization_run_root,
    training_horizon_run_directory,
)
from roofseg.pilot import build_config, pilot_run_root
from roofseg.run_artifacts import prepare_run
from roofseg.training import (
    adamw, apply_learning_rate, learning_rate_for_step, restore_checkpoint,
)


def test_model_output_and_batchnorm_modes():
    model = EfficientNetB0UNet()
    batch_norms = [module for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d)]
    assert batch_norms and all(module.weight.requires_grad and module.bias.requires_grad for module in batch_norms)
    model.train()
    before = batch_norms[0].running_mean.clone()
    output = model(torch.randn(2, 3, 64, 64))
    assert output.shape == (2, 1, 64, 64)
    assert not torch.equal(before, batch_norms[0].running_mean)
    model.eval()
    before_eval = batch_norms[0].running_mean.clone()
    with torch.inference_mode():
        model(torch.randn(2, 3, 64, 64))
    torch.testing.assert_close(before_eval, batch_norms[0].running_mean)


def test_random_models_pair_decoder_initialization():
    first, first_report = build_model("random", 42)
    second, second_report = build_model("random", 42)
    assert first_report["decoder_sha256"] == second_report["decoder_sha256"]
    assert state_digest(first) == state_digest(second)


def test_optimizer_excludes_bias_and_normalization_from_decay():
    model = EfficientNetB0UNet()
    optimizer, record = adamw(model, 3e-4, 1e-4)
    assert [group["weight_decay"] for group in optimizer.param_groups] == [1e-4, 0.0]
    assert [group["lr"] for group in optimizer.param_groups] == [3e-4, 3e-4]
    assert record["learning_rates"] == [3e-4, 3e-4]
    assert record["decayed_parameter_tensors"] > 0
    assert record["no_decay_parameter_tensors"] > 0


def test_split_config_uses_only_saved_training_and_validation_ids():
    splits = {
        "split_name": "fixture",
        "roles": {"validation": ["108", "109"], "test": ["110"]},
        "training_subsets": {"repetitions": {"1": {"seed": 17, "subsets": {"25": [str(i) for i in range(25)]}}}},
    }
    config = build_config(splits, 25, "random")
    assert config["training_ids"] == [str(i) for i in range(25)]
    assert config["validation_ids"] == ["108", "109"]
    assert not set(config["training_ids"]) & set(splits["roles"]["test"])


def test_pilot_run_root_is_split_versioned(tmp_path: Path):
    splits = {"split_name": "geographic_v2"}
    assert pilot_run_root(tmp_path, splits) == tmp_path / "runs/training_pilot/geographic_v2"


def test_learning_rate_configs_change_only_identity_and_learning_rate(tmp_path: Path):
    splits = {
        "split_name": "geographic_v2",
        "roles": {"validation": ["108"], "test": ["110"]},
        "training_subsets": {
            "repetitions": {"1": {"seed": 17, "subsets": {"100": [str(i) for i in range(100)]}}}
        },
    }
    reference = build_learning_rate_config(splits, "random", 3e-4)
    lower = build_learning_rate_config(splits, "random", 1e-4)
    differences = {
        key for key in reference | lower
        if reference.get(key) != lower.get(key)
    }
    assert differences == {"run_name", "experiment", "learning_rate"}
    assert learning_rate_run_directory(tmp_path, splits, "random", 3e-4) == (
        tmp_path / "runs/training_pilot/geographic_v2/n100_random"
    )
    assert learning_rate_run_directory(tmp_path, splits, "random", 1e-4) == (
        optimization_run_root(tmp_path, splits) / "n100_random_lr1e-4"
    )
    assert learning_rate_run_directory(tmp_path, splits, "random", 1e-3) != (
        learning_rate_run_directory(tmp_path, splits, "random", 1e-4)
    )


def test_training_horizon_changes_only_identity_and_budget(tmp_path: Path):
    splits = {
        "split_name": "geographic_v2",
        "roles": {"validation": ["108"], "test": ["110"]},
        "training_subsets": {
            "repetitions": {"1": {"seed": 17, "subsets": {"100": [str(i) for i in range(100)]}}}
        },
    }
    historical = build_learning_rate_config(splits, "imagenet", 1e-4)
    extended = build_training_horizon_config(splits, "imagenet", 1e-4)
    differences = {
        key for key in historical | extended
        if historical.get(key) != extended.get(key)
    }
    assert differences == {"run_name", "experiment", "max_steps", "evaluation_steps"}
    assert extended["max_steps"] == 4000
    assert extended["evaluation_steps"] == [0] + list(range(100, 4001, 100))
    assert training_horizon_run_directory(tmp_path, splits, "imagenet", 1e-4) != (
        learning_rate_run_directory(tmp_path, splits, "imagenet", 1e-4)
    )


def test_training_horizon_rejects_unapproved_rate():
    splits = {
        "split_name": "geographic_v2",
        "roles": {"validation": ["108"], "test": ["110"]},
        "training_subsets": {
            "repetitions": {"1": {"seed": 17, "subsets": {"100": [str(i) for i in range(100)]}}}
        },
    }
    with pytest.raises(ValueError, match="outside"):
        build_training_horizon_config(splits, "random", 3e-4)


def test_checkpoint_restore_rejects_mismatch(tmp_path: Path):
    model = torch.nn.Conv2d(1, 1, 1)
    path = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "step": 3,
                "config_sha256": "config", "input_sha256": "input"}, path)
    assert restore_checkpoint(model, path, "config", "input") == 3
    with pytest.raises(ValueError, match="identity mismatch"):
        restore_checkpoint(model, path, "different", "input")


def test_incomplete_or_conflicting_run_is_not_reused(tmp_path: Path):
    config, metadata = {"a": 1}, {"compatibility_identity": {"x": 1}}
    assert prepare_run(tmp_path, config, metadata) is False
    with pytest.raises(ValueError, match="incomplete"):
        prepare_run(tmp_path, config, metadata)
    with pytest.raises(ValueError, match="conflicts"):
        prepare_run(tmp_path, {"a": 2}, metadata)


def test_fixed_late_drop_changes_only_identity_and_scheduler(tmp_path: Path):
    splits = {
        "split_name": "geographic_v2",
        "roles": {"validation": ["108"], "test": ["110"]},
        "training_subsets": {
            "repetitions": {"1": {"seed": 17, "subsets": {"100": [str(i) for i in range(100)]}}}
        },
    }
    reference = build_training_horizon_config(splits, "random", 1e-3)
    scheduled = build_late_lr_drop_config(splits, "random")
    differences = {
        key for key in reference | scheduled
        if reference.get(key) != scheduled.get(key)
    }
    assert differences == {"run_name", "experiment", "scheduler"}
    assert late_lr_drop_run_directory(tmp_path, splits, "random") != (
        training_horizon_run_directory(tmp_path, splits, "random", 1e-3)
    )


def test_fixed_late_drop_boundary_updates_all_groups_without_changing_decay():
    config = {"max_steps": 4000, "learning_rate": 1e-3, "scheduler": {
        "type": "piecewise_constant", "adaptive": False, "phases": [
            {"start_step": 1, "end_step": 2000, "learning_rate": 1e-3},
            {"start_step": 2001, "end_step": 4000, "learning_rate": 1e-4},
        ],
    }}
    model = torch.nn.Linear(2, 1)
    optimizer, _ = adamw(model, 1e-3, 1e-4)
    original_decay = [group["weight_decay"] for group in optimizer.param_groups]
    assert learning_rate_for_step(config, 2000) == 1e-3
    assert learning_rate_for_step(config, 2001) == 1e-4
    apply_learning_rate(optimizer, learning_rate_for_step(config, 2001))
    assert [group["lr"] for group in optimizer.param_groups] == [1e-4, 1e-4]
    assert [group["weight_decay"] for group in optimizer.param_groups] == original_decay
    constant = {"max_steps": 4000, "learning_rate": 1e-3, "scheduler": None}
    assert learning_rate_for_step(constant, 4000) == 1e-3


def test_fixed_drop_selects_saved_seed29_subset_and_unique_directory(tmp_path: Path):
    seed17_ids = [str(i) for i in range(100)]
    seed29_ids = [str(i) for i in range(100, 200)]
    splits = {
        "split_name": "geographic_v2",
        "roles": {"training": seed17_ids + seed29_ids,
                  "validation": ["300"], "test": ["400"]},
        "training_subsets": {"repetitions": {
            "1": {"seed": 17, "subsets": {"100": seed17_ids}},
            "2": {"seed": 29, "subsets": {"100": seed29_ids}},
        }},
    }
    seed17 = build_late_lr_drop_config(splits, "random")
    seed29_random = build_late_lr_drop_config(splits, "random", subset_repetition=2)
    seed29_imagenet = build_late_lr_drop_config(splits, "imagenet", subset_repetition=2)

    assert seed17["seeds"]["subset"] == 17
    assert seed29_random["training_ids"] == seed29_ids
    assert seed29_random["data"]["training_subset"] == (
        "training_subsets.repetitions.2.subsets.100"
    )
    assert seed29_random["seeds"] == {
        "subset": 29, "model": 20261004, "data_order": 1701, "augmentation": 1702
    }
    assert seed29_random["training_ids"] == seed29_imagenet["training_ids"]
    assert not set(seed29_random["training_ids"]) & {"300", "400"}
    differences = {
        key for key in seed17 | seed29_random if seed17.get(key) != seed29_random.get(key)
    }
    assert differences == {"run_name", "data", "training_ids", "seeds"}
    assert late_lr_drop_run_directory(tmp_path, splits, "random") != (
        late_lr_drop_run_directory(tmp_path, splits, "random", subset_repetition=2)
    )
    assert late_lr_drop_run_directory(tmp_path, splits, "random", 2) != (
        late_lr_drop_run_directory(tmp_path, splits, "imagenet", 2)
    )


def test_fixed_drop_data_efficiency_sizes_are_nested_paired_and_distinct(tmp_path: Path):
    ids25 = [str(i) for i in range(25)]
    ids100 = [str(i) for i in range(100)]
    ids500 = [str(i) for i in range(500)]
    splits = {
        "split_name": "geographic_v2",
        "roles": {"training": ids500, "validation": ["600"], "test": ["700"]},
        "training_subsets": {"repetitions": {
            "1": {"seed": 17, "subsets": {
                "25": ids25, "100": ids100, "500": ids500,
            }},
        }},
    }
    configs = {
        size: {
            initialization: build_late_lr_drop_config(
                splits, initialization, training_size=size
            )
            for initialization in ("random", "imagenet")
        }
        for size in (25, 100, 500)
    }
    assert set(configs[25]["random"]["training_ids"]) < set(
        configs[100]["random"]["training_ids"]
    ) < set(configs[500]["random"]["training_ids"])
    for size, pair in configs.items():
        assert len(pair["random"]["training_ids"]) == size
        assert pair["random"]["training_ids"] == pair["imagenet"]["training_ids"]
        assert not set(pair["random"]["training_ids"]) & {"600", "700"}
        assert pair["random"]["max_steps"] == 4000
        assert pair["random"]["scheduler"] == configs[100]["random"]["scheduler"]
    directories = {
        late_lr_drop_run_directory(
            tmp_path, splits, initialization, training_size=size
        )
        for size in (25, 100, 500)
        for initialization in ("random", "imagenet")
    }
    assert len(directories) == 6
    assert build_late_lr_drop_config(splits, "random") == configs[100]["random"]
