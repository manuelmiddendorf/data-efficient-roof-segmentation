from pathlib import Path

import pytest
import torch

from roofseg.model import EfficientNetB0UNet, build_model, state_digest
from roofseg.pilot import build_config, pilot_run_root
from roofseg.run_artifacts import prepare_run
from roofseg.training import adamw, restore_checkpoint


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
    assert record["decayed_parameter_tensors"] > 0
    assert record["no_decay_parameter_tensors"] > 0


def test_split_config_uses_only_saved_training_and_validation_ids():
    splits = {
        "split_name": "fixture",
        "roles": {"validation": ["108", "109"], "test": ["110"]},
        "training_subsets": {"repetitions": {"1": {"subsets": {"25": [str(i) for i in range(25)]}}}},
    }
    config = build_config(splits, 25, "random")
    assert config["training_ids"] == [str(i) for i in range(25)]
    assert config["validation_ids"] == ["108", "109"]
    assert not set(config["training_ids"]) & set(splits["roles"]["test"])


def test_pilot_run_root_is_split_versioned(tmp_path: Path):
    splits = {"split_name": "geographic_v2"}
    assert pilot_run_root(tmp_path, splits) == tmp_path / "runs/training_pilot/geographic_v2"


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
