"""Observer-only algebra and decision tests; no TextRelationModel forwards."""
from pathlib import Path
import importlib.util
import sys
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def observer():
    path = ROOT / "tools/p08_signal_observer.py"
    assert path.exists(), "signal observer not implemented"
    spec = importlib.util.spec_from_file_location("p08_signal_observer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture():
    weight = torch.zeros(2, 64)
    weight[0, 0], weight[1, 0] = 2, -1
    bias = torch.tensor([0.75, -0.25])
    sequence = torch.zeros(3, 64)
    sequence[:, 0] = torch.tensor([1.0, 3.0, 999.0])
    mask = torch.tensor([True, True, False])
    return weight, bias, sequence, mask


def test_mean_margin_excludes_pad_and_reconstructs_time_contributions():
    obs = observer()
    weight, bias, sequence, mask = fixture()
    readout = sequence[:2].mean(0)
    with torch.no_grad():
        logits = torch.nn.functional.linear(readout, weight, bias)
        value = obs.decompose_allowed(weight, bias, readout, logits, sequence, mask, "valid_token_mean")
    assert value["original_margin"] == 7.0
    assert value["bias_difference"] == 1.0
    assert value["component_contributions"] == [6.0] + [0.0] * 63
    assert value["time_contributions"] == [1.5, 4.5]
    assert value["time_reconstructed_margin"] == 7.0
    assert value["reconstruction_abs_error"] == 0.0


def test_last_margin_uses_final_valid_state_and_no_mean_time_claim():
    obs = observer()
    weight, bias, sequence, mask = fixture()
    with torch.no_grad():
        logits = torch.nn.functional.linear(sequence[1], weight, bias)
        value = obs.decompose_allowed(weight, bias, sequence[1], logits, sequence, mask, "last")
    assert value["original_margin"] == 10.0
    assert value["component_contributions"][0] == 9.0
    assert value["time_contributions"] is None
    assert value["time_reconstructed_margin"] is None


def test_reconstruction_rejects_wrong_readout_and_wrong_margin():
    obs = observer()
    weight, bias, sequence, mask = fixture()
    with torch.no_grad():
        readout = sequence[:2].mean(0)
        logits = torch.nn.functional.linear(readout, weight, bias)
        with pytest.raises(AssertionError, match="readout"):
            obs.decompose_allowed(weight, bias, sequence[0], logits, sequence, mask, "valid_token_mean")
        with pytest.raises(AssertionError, match="margin"):
            obs.decompose_allowed(weight, bias, readout, logits + torch.tensor([0.1, 0.0]), sequence, mask, "valid_token_mean")


def test_classification_respects_repeat_noise_and_distinguishes_argmax():
    obs = observer()
    assert obs.classify_spreads(1e-8, 1e-8, 0, 0, True)["classification"] == "readout_difference_lost"
    assert obs.classify_spreads(0.1, 1e-8, 0, 0, True)["classification"] == "readout_difference_not_in_logits"
    assert obs.classify_spreads(0.1, 0.01, 0, 0, True)["classification"] == "argmax_only_same"
    assert obs.classify_spreads(1e-4, 1e-4, 1e-3, 1e-3, True)["classification"] == "readout_difference_lost"
