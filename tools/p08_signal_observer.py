"""Passive fixed8 observer. No training, labels, model edits, or causal attribution."""
from pathlib import Path
import hashlib
import json
import os
import time
import threading
import torch
from torch.nn.utils.rnn import pad_packed_sequence

HEADS = ("box", "tool", "from", "to", "allowed")
ATOL, RTOL = 2e-6, 1e-5


def metadata(path):
    raw = Path(path).read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def decompose_allowed(weight, bias, readout, logits, sequence, mask, policy):
    assert not torch.is_grad_enabled(), "observer requires no-grad"
    assert weight.shape == (2, 64) and bias.shape == (2,) and readout.shape == (64,)
    assert sequence.ndim == 2 and sequence.shape[1] == 64 and mask.dtype == torch.bool
    assert mask.shape == (sequence.shape[0],) and bool(mask.any())
    valid = sequence[mask]
    assert policy in ("last", "valid_token_mean")
    expected = valid[-1] if policy == "last" else valid.sum(0) / len(valid)
    assert torch.allclose(readout, expected, atol=1e-6, rtol=1e-6), "actual readout mismatch"
    delta_w = weight[0].double() - weight[1].double()
    delta_b = float(bias[0].double() - bias[1].double())
    components = readout.double() * delta_w
    margin = float(logits[0].double() - logits[1].double())
    reconstructed = delta_b + float(components.sum())
    assert abs(margin - reconstructed) <= ATOL + RTOL * abs(margin), "allowed margin mismatch"
    times = (valid.double() @ delta_w) / len(valid) if policy == "valid_token_mean" else None
    time_margin = delta_b + float(times.sum()) if times is not None else None
    assert time_margin is None or abs(margin - time_margin) <= ATOL + RTOL * abs(margin), "time margin mismatch"
    return {"sign": "false-minus-true;positive favors false",
            "original_margin": margin, "bias_difference": delta_b,
            "weight_difference": delta_w.tolist(), "component_contributions": components.tolist(),
            "reconstructed_margin": reconstructed, "reconstruction_abs_error": abs(margin-reconstructed),
            "time_contributions": times.tolist() if times is not None else None,
            "time_reconstructed_margin": time_margin,
            "time_reconstruction_abs_error": abs(margin-time_margin) if times is not None else None,
            "valid_token_count": len(valid), "readout_max_abs_error": float((readout-expected).abs().max()),
            "tolerance": {"atol": ATOL, "rtol": RTOL},
            "interpretation": "Exact linear arithmetic on contextual hidden states; not word causality or self-explanation."}


def classify_spreads(readout_spread, logit_spread, repeat_readout, repeat_logit, argmax_equal):
    read_floor, logit_floor = max(1e-6, 5*repeat_readout), max(1e-6, 5*repeat_logit)
    if readout_spread <= read_floor:
        kind = "readout_difference_lost"
    elif logit_spread <= logit_floor:
        kind = "readout_difference_not_in_logits"
    elif argmax_equal:
        kind = "argmax_only_same"
    else:
        kind = "argmax_also_differs"
    return {"classification": kind, "readout_threshold": read_floor, "logit_threshold": logit_floor,
            "numerical_floor": 1e-6, "repeat_noise_multiplier": 5}


def observe_single(model, text, ident, owner, seed, policy):
    from idea_model.codec import encode_text
    from idea_model.contracts import RunMode, new_state
    from p08_encoder import make_fixed_batch, to_tensor_batch
    assert not torch.is_grad_enabled() and not model.training
    batch = to_tensor_batch(make_fixed_batch((encode_text(text),), item_ids=(ident,)))
    # A masked PAD tail gives explicit actual input evidence; packed GRU never processes it.
    batch = type(batch)(torch.cat((batch.ids, torch.zeros(1, 7, dtype=torch.int64)), 1),
                        batch.lengths, torch.cat((batch.valid_mask, torch.zeros(1, 7, dtype=torch.bool)), 1),
                        batch.item_ids)
    assert batch.ids.shape[1] == int(batch.lengths[0])+7
    assert not bool(batch.valid_mask[0, int(batch.lengths[0]):].any())
    captured = {"readouts": [], "head_outputs": [], "gru": []}
    handles = []
    def guard(value):
        assert not torch.is_grad_enabled() and not value.requires_grad and value.grad_fn is None
        assert torch.isfinite(value).all()
    def gru_hook(module, args, outputs):
        sequence, lengths = pad_packed_sequence(outputs[0], batch_first=True)
        guard(sequence); guard(outputs[1])
        assert lengths.tolist() == batch.lengths.tolist()
        captured["gru"].append((sequence[0].clone(), outputs[1][0, 0].clone()))
    def pre_hook(module, args):
        guard(args[0]); captured["readouts"].append(args[0][0].clone())
    def head_hook(module, args, output):
        guard(output); captured["head_outputs"].append(output[0].clone())
    heads = (*model.relation_heads, model.allowed_head)
    assert all(type(head) is torch.nn.Linear and head.in_features == 64 and head.bias is not None
               and head.out_features == size for head, size in zip(heads, (4, 4, 4, 4, 2)))
    handles.append(model.gru.register_forward_hook(gru_hook))
    for head in heads:
        # Append after the mean adapter's existing hook to capture actual head input.
        handles.extend((head.register_forward_pre_hook(pre_hook), head.register_forward_hook(head_hook)))
    try:
        prediction = model(batch, (new_state(owner, ident, seed=seed),),
                           owner_id=owner, episode_ids=(ident,), seed=seed, mode=RunMode.SELECTION)
    finally:
        for handle in handles:
            handle.remove()
    assert len(captured["gru"]) == 1 and len(captured["readouts"]) == 5
    sequence, final = captured["gru"][0]
    assert torch.equal(final, sequence[-1])
    assert torch.equal(final, torch.tensor(prediction.new_states[0].hidden, dtype=torch.float32))
    readout = captured["readouts"][0]
    assert all(torch.equal(readout, value) for value in captured["readouts"])
    logits = [prediction.relation_logits[0, i] for i in range(4)] + [prediction.allowed_logits[0]]
    assert all(torch.equal(a, b) for a, b in zip(logits, captured["head_outputs"]))
    for value in logits:
        guard(value)
    probabilities = [torch.softmax(value, -1) for value in logits]
    decomposition = decompose_allowed(model.allowed_head.weight, model.allowed_head.bias,
                                      readout, logits[-1], sequence,
                                      batch.valid_mask[0, :len(sequence)], policy)
    return {"item_id": ident, "input_text": text, "ids": batch.ids[0].tolist(),
            "length": int(batch.lengths[0]), "valid_mask": batch.valid_mask[0].tolist(),
            "hidden_valid_steps": sequence.tolist(), "pad_hidden_steps": None,
            "final_recurrent_hidden": final.tolist(), "actual_readout": readout.tolist(),
            "head_readouts_identical": True,
            "logits18": torch.cat(logits).tolist(), "probabilities18": torch.cat(probabilities).tolist(),
            "argmax5": [int(value.argmax()) for value in logits],
            "allowed_decomposition": decomposition}


def summarize(records):
    values = records[:8]
    repeat = {}
    for key in ("hidden_valid_steps", "final_recurrent_hidden", "actual_readout", "logits18", "probabilities18"):
        a, b = torch.tensor(records[0][key], dtype=torch.float64), torch.tensor(records[8][key], dtype=torch.float64)
        repeat[key] = {"max_abs_delta": float((a-b).abs().max()), "l2_delta": float(torch.linalg.vector_norm(a-b)),
                       "bitwise_equal": torch.equal(a, b)}
        assert repeat[key]["max_abs_delta"] <= 1e-6, "repeat/order drift"
    assert records[0]["ids"] == records[8]["ids"] and records[0]["valid_mask"] == records[8]["valid_mask"]
    assert records[0]["argmax5"] == records[8]["argmax5"]
    spreads = {}
    for key in ("actual_readout", "final_recurrent_hidden", "logits18", "probabilities18"):
        tensor = torch.tensor([value[key] for value in values], dtype=torch.float64)
        distances = torch.linalg.vector_norm(tensor[:, None]-tensor[None, :], dim=-1)
        pairs = distances[torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1)]
        spreads[key] = {"max_component_spread": float((tensor.max(0).values-tensor.min(0).values).max()),
                        "pairwise_l2_min": float(pairs.min()), "pairwise_l2_max": float(pairs.max()),
                        "pairwise_l2_mean": float(pairs.mean()), "pairwise_l2": distances.tolist(),
                        "component_spreads": (tensor.max(0).values-tensor.min(0).values).tolist()}
    token_rows = [value["ids"][:value["length"]] for value in values]
    suffix = 0
    while suffix < min(map(len, token_rows)) and len({row[-suffix-1] for row in token_rows}) == 1:
        suffix += 1
    assert suffix > 0 and suffix < min(map(len, token_rows))
    profile = []
    for count in range(suffix+1):
        h = torch.tensor([value["hidden_valid_steps"][value["length"]-suffix-1+count] for value in values], dtype=torch.float64)
        d = torch.linalg.vector_norm(h[:, None]-h[None, :], dim=-1)
        pairs = d[torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1)]
        profile.append({"common_suffix_processed": count, "pair_l2_mean": float(pairs.mean()),
                        "pair_l2_max": float(pairs.max()),
                        "max_component_spread": float((h.max(0).values-h.min(0).values).max())})
    same_argmax = all(value["argmax5"] == values[0]["argmax5"] for value in values)
    classification = classify_spreads(spreads["actual_readout"]["max_component_spread"],
                                      spreads["logits18"]["max_component_spread"],
                                      repeat["actual_readout"]["max_abs_delta"], repeat["logits18"]["max_abs_delta"], same_argmax)
    return {"repeat_order_noise": repeat, "input_spreads": spreads, "common_suffix_tokens": suffix,
            "suffix_hidden_profile": profile, "all_input_argmax_identical": same_argmax,
            "argmax5": [value["argmax5"] for value in values],
            "allowed_margin_range": [min(v["allowed_decomposition"]["original_margin"] for v in values),
                                     max(v["allowed_decomposition"]["original_margin"] for v in values)],
            **classification}


