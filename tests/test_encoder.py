"""Tiny untrained model checks only; forward/backward requires execution slot."""
import dataclasses
import importlib
import importlib.util
import inspect
import pytest
from idea_model.codec import encode_text
from idea_model.contracts import ContractError, RunMode, new_state, reset, ResetScope


def api():
    assert importlib.util.find_spec("p08_encoder") is not None, "P08 component absent: GRU/head implementation required"
    p = importlib.import_module("p08_encoder")
    assert hasattr(p, "TextRelationModel"), "P08 GRU/head implementation absent"
    return p


def setup(texts=("각",), owner="owner-a", seed=31101, mode=RunMode.TEST):
    p = api()
    model = p.TextRelationModel(owner, seed=seed)
    episodes = tuple(f"episode-{i}" for i in range(len(texts)))
    batch = p.to_tensor_batch(p.make_fixed_batch(tuple(map(encode_text, texts)),
                                               item_ids=tuple(f"input-{i}" for i in range(len(texts)))))
    factory = p.new_train_state if mode is RunMode.TRAIN else new_state
    states = tuple(factory(owner, ep, seed=seed) for ep in episodes)
    return p, model, batch, states, dict(owner_id=owner, episode_ids=episodes, seed=seed, mode=mode)


def test_target_free_learned_heads_and_parameter_count():
    p, model, batch, states, kwargs = setup()
    import torch
    assert "targets" not in inspect.signature(model.forward).parameters
    result = model(batch, states, **kwargs)
    assert result.relation_logits.shape == (1, 4, 4) and result.allowed_logits.shape == (1, 2)
    assert result.relation_logits.dtype is torch.float32
    assert result.relation_names == ("box", "tool", "from", "to")
    assert result.observed_mask.tolist() == [True]
    assert model.embedding.weight.shape == (327, 32)
    assert model.gru.input_size == 32 and model.gru.hidden_size == 64
    assert sum(param.numel() for param in model.parameters()) == 30450
    # A real head-weight intervention must change predicted logits.
    with torch.no_grad():
        model.allowed_head.bias.add_(torch.tensor([1.0, -1.0]))
    changed = model(batch, states, **kwargs)
    assert not torch.equal(result.allowed_logits, changed.allowed_logits)
    assert result.family == "idea-relational-gru-v1" and result.port == "text-relations-327-v1"


def test_padding_and_batch_order_do_not_change_short_prediction_or_state():
    _, model, short, states, kw = setup()
    p = api()
    alone = model(short, states, **kw)
    mixed = p.to_tensor_batch(p.make_fixed_batch((encode_text("각"), encode_text("A" * 9)),
                                                item_ids=("input-0", "long")))
    mixed_kw = dict(kw, episode_ids=("episode-0", "episode-long"))
    together = model(mixed, states + (new_state("owner-a", "episode-long", seed=31101),), **mixed_kw)
    import torch
    torch.testing.assert_close(alone.relation_logits[0], together.relation_logits[0], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(alone.allowed_logits[0], together.allowed_logits[0], rtol=1e-6, atol=1e-7)
    assert alone.new_states[0].hidden == pytest.approx(together.new_states[0].hidden, abs=1e-7)
    assert states[0].hidden == () and states[0].mutable_trace == {}


def test_empty_observation_and_empty_batch_make_no_state_or_memory_write():
    _, model, batch, states, kw = setup(texts=("", "각"))
    states[0].hidden = (0.125,) * 64
    states[0].mutable_trace["public"] = [1]
    before = states[0].memory_view
    out = model(batch, states, **kw)
    assert out.observed_mask.tolist() == [False, True]
    assert out.new_states[0] is states[0]
    assert out.new_states[0].hidden == (0.125,) * 64
    assert out.new_states[0].mutable_trace == {"public": [1]}
    assert out.new_states[0].memory_view is before
    p = api()
    empty = p.to_tensor_batch(p.make_fixed_batch((), item_ids=()))
    result = model(empty, (), **dict(kw, episode_ids=()))
    assert result.relation_logits.shape == (0, 4, 4) and result.allowed_logits.shape == (0, 2)
    assert result.new_states == () and result.observed_mask.shape == (0,)


@pytest.mark.parametrize("fault", ["ids_dtype", "lengths_dtype", "mask_dtype", "negative_id", "unsupported_id",
                                   "active_pad", "masked_nonpad", "mask_not_prefix", "too_long", "duplicate_item",
                                   "schema", "port", "token_bound"])
def test_malformed_inputs_fail_before_any_state_update(fault):
    p, model, batch, states, kw = setup(texts=("각", ""))
    import torch
    fields = {}
    if fault == "ids_dtype": fields["ids"] = batch.ids.to(torch.float32)
    elif fault == "lengths_dtype": fields["lengths"] = batch.lengths.to(torch.int32)
    elif fault == "mask_dtype": fields["valid_mask"] = batch.valid_mask.to(torch.int64)
    elif fault in ("negative_id", "unsupported_id", "active_pad", "masked_nonpad"):
        fields["ids"] = batch.ids.clone()
        row, col, value = (1, 0, 1) if fault == "masked_nonpad" else (0, 0, {"negative_id": -1, "unsupported_id": 327, "active_pad": 0}[fault])
        fields["ids"][row, col] = value
    elif fault == "mask_not_prefix":
        fields["valid_mask"] = batch.valid_mask.clone(); fields["valid_mask"][0, 0] = False
    elif fault == "too_long": fields["lengths"] = torch.tensor([4, 0], dtype=torch.int64)
    elif fault == "duplicate_item": fields["item_ids"] = ("same", "same")
    elif fault == "schema": fields["schema_version"] = 2
    elif fault == "port": fields["port"] = "native77"
    elif fault == "token_bound":
        fields["ids"] = torch.zeros((2, 1537), dtype=torch.int64); fields["ids"][0] = 1
        fields["lengths"] = torch.tensor([1537, 0], dtype=torch.int64)
        fields["valid_mask"] = fields["ids"].bool()
    invalid = dataclasses.replace(batch, **fields)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    with pytest.raises(ContractError):
        model(invalid, states, **kw)
    assert all(state.hidden == () and state.mutable_trace == {} for state in states)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert all(param.grad is None for param in model.parameters())


@pytest.mark.parametrize("fault", ["owner", "seed", "episode", "duplicate_episode", "mode"])
def test_context_identity_is_explicit_and_fail_closed(fault):
    _, model, batch, states, kw = setup(texts=("각", "A"))
    if fault == "owner": kw["owner_id"] = "owner-b"
    elif fault == "seed": kw["seed"] = 31109
    elif fault == "episode": kw["episode_ids"] = ("wrong", "episode-1")
    elif fault == "duplicate_episode": kw["episode_ids"] = ("episode-0", "episode-0")
    elif fault == "mode": kw["mode"] = "unknown"
    with pytest.raises(ContractError): model(batch, states, **kw)
    assert all(state.hidden == () for state in states)


def test_eval_does_not_change_weights_or_gradients_and_uses_tuple_snapshots():
    _, model, batch, states, kw = setup()
    import torch
    before = {key: value.clone() for key, value in model.state_dict().items()}
    for mode in (RunMode.SELECTION, RunMode.TEST):
        out = model(batch, states, **dict(kw, mode=mode))
        assert not out.relation_logits.requires_grad and not out.allowed_logits.requires_grad
        assert type(out.new_states[0].hidden) is tuple and len(out.new_states[0].hidden) == 64
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert all(param.grad is None for param in model.parameters())


def test_seed_reproducibility_and_individual_weight_storage_isolation():
    _, a, batch, states, kw = setup()
    p = api()
    import torch
    rng_before = torch.random.get_rng_state().clone()
    b = p.TextRelationModel("owner-b", seed=31101)
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert all(torch.equal(x, y) and x.data_ptr() != y.data_ptr() for x, y in zip(a.parameters(), b.parameters()))
    before = tuple(x.clone() for x in b.parameters())
    with torch.no_grad(): next(a.parameters()).add_(1.0)
    assert all(torch.equal(x, y) for x, y in zip(before, b.parameters()))


def test_episode_reset_and_individual_reset_are_local():
    _, model, batch, states, kw = setup()
    out = model(batch, states, **kw)
    original = out.new_states[0]
    other = new_state("other", "other-episode", seed=3)
    other.mutable_trace["kept"] = [9]
    for scope in (ResetScope.EPISODE, ResetScope.INDIVIDUAL):
        clean = reset(original, scope=scope)
        assert clean.hidden == () and clean.memory_view.records == () and clean.mutable_trace == {}
    assert len(original.hidden) == 64 and other.mutable_trace == {"kept": [9]}


@pytest.mark.parametrize("mode", [RunMode.TEST, RunMode.TRAIN])
def test_external_cpu_autocast_cannot_change_float32_output_contract(mode):
    _, model, batch, states, kw = setup(texts=("",), mode=mode)
    import torch
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(batch, states, **kw)
    assert out.relation_logits.dtype == torch.float32
    assert out.allowed_logits.dtype == torch.float32
    assert out.new_states[0] is states[0]
