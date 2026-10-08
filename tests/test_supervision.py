"""Handwritten train-only labels; no formal packs, environment solve/render, or sealed test."""
import dataclasses
import importlib
import importlib.util
import pytest
from idea_model.codec import encode_text
from idea_model.contracts import ContractError, RunMode, new_state


def setup(texts=("각",), seed=31101):
    assert importlib.util.find_spec("p08_encoder") is not None, "P08 component absent: train graph/supervision required"
    p = importlib.import_module("p08_encoder")
    assert hasattr(p, "TrainOnlySupervisor"), "P08 train-only supervisor absent"
    import torch
    model = p.TextRelationModel("owner-a", seed=seed)
    episodes = tuple(f"episode-{i}" for i in range(len(texts)))
    items = tuple(f"input-{i}" for i in range(len(texts)))
    batch = p.to_tensor_batch(p.make_fixed_batch(tuple(map(encode_text, texts)), item_ids=items))
    states = tuple(p.new_train_state("owner-a", ep, seed=seed) for ep in episodes)
    kw = dict(owner_id="owner-a", episode_ids=episodes, seed=seed, mode=RunMode.TRAIN)
    supervisor = p.TrainOnlySupervisor(model, owner_id="owner-a", seed=seed, mode=RunMode.TRAIN, lr=0.001)
    targets = p.TrainTargets(items, episodes, torch.zeros((len(texts), 4), dtype=torch.int64),
                            torch.zeros(len(texts), dtype=torch.int64))
    return p, torch, model, supervisor, batch, states, kw, targets


def test_full_episode_bptt_reaches_earlier_observation_across_empty_input():
    p, torch, model, trainer, batch, states, kw, targets = setup()
    first = model(batch, states, **kw)
    early = first.new_states[0].hidden
    early.retain_grad()
    empty_batch = p.to_tensor_batch(p.make_fixed_batch((encode_text(""),), item_ids=("input-0",)))
    skipped = model(empty_batch, first.new_states, **kw)
    assert skipped.new_states[0] is first.new_states[0]
    second_batch = p.to_tensor_batch(p.make_fixed_batch((encode_text("A"),), item_ids=("input-0",)))
    second = model(second_batch, skipped.new_states, **kw)
    loss = trainer.relation_loss(second, targets)
    loss.backward()
    assert early.grad is not None and torch.count_nonzero(early.grad).item() > 0
    assert model.embedding.weight.grad is not None
    assert torch.count_nonzero(model.embedding.weight.grad[1:68]).item() > 0
    assert torch.count_nonzero(model.embedding.weight.grad[0]).item() == 0


def test_relation_loss_uses_five_valid_heads_and_excludes_empty_row_targets():
    p, torch, model, trainer, batch, states, kw, targets = setup(texts=("각", ""))
    out = model(batch, states, **kw)
    loss = trainer.relation_loss(out, targets)
    import torch.nn.functional as F
    manual = (F.cross_entropy(out.relation_logits[:1].reshape(4, 4), targets.relations[:1].reshape(4), reduction="sum") +
              F.cross_entropy(out.allowed_logits[:1], targets.allowed[:1], reduction="sum")) / 5
    torch.testing.assert_close(loss, manual)
    other_labels = dataclasses.replace(targets, relations=torch.tensor([[0, 0, 0, 0], [3, 2, 1, 3]], dtype=torch.int64),
                                      allowed=torch.tensor([0, 1], dtype=torch.int64))
    torch.testing.assert_close(loss, trainer.relation_loss(out, other_labels))


def test_all_empty_loss_is_zero_and_does_not_update_weights():
    _, torch, model, trainer, batch, states, kw, targets = setup(texts=("",))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    out = model(batch, states, **kw)
    loss = trainer.relation_loss(out, targets)
    assert loss.item() == 0.0
    loss.backward()
    assert all(param.grad is None or torch.count_nonzero(param.grad).item() == 0 for param in model.parameters())
    assert out.new_states[0] is states[0]
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


@pytest.mark.parametrize("mode", [RunMode.SELECTION, RunMode.TEST])
def test_eval_cannot_acquire_supervision_or_consume_train_labels(mode):
    p, _, model, trainer, batch, _, kw, targets = setup()
    with pytest.raises(ContractError):
        p.TrainOnlySupervisor(model, owner_id="owner-a", seed=31101, mode=mode, lr=0.001)
    eval_states = (new_state("owner-a", "episode-0", seed=31101),)
    eval_out = model(batch, eval_states, **dict(kw, mode=mode))
    with pytest.raises(ContractError):
        trainer.relation_loss(eval_out, targets)
    assert all(param.grad is None for param in model.parameters())


@pytest.mark.parametrize("fault", ["item", "episode", "relation_dtype", "allowed_dtype", "relation_shape",
                                   "relation_range", "allowed_range"])
def test_invalid_train_targets_are_rejected_atomically(fault):
    p, torch, model, trainer, batch, states, kw, targets = setup()
    out = model(batch, states, **kw)
    fields = {}
    if fault == "item": fields["item_ids"] = ("wrong",)
    elif fault == "episode": fields["episode_ids"] = ("wrong",)
    elif fault == "relation_dtype": fields["relations"] = targets.relations.to(torch.float32)
    elif fault == "allowed_dtype": fields["allowed"] = targets.allowed.to(torch.bool)
    elif fault == "relation_shape": fields["relations"] = targets.relations[:, :3]
    elif fault == "relation_range": fields["relations"] = torch.tensor([[0, 0, 0, 4]], dtype=torch.int64)
    elif fault == "allowed_range": fields["allowed"] = torch.tensor([2], dtype=torch.int64)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    old_hidden = out.new_states[0].hidden.clone()
    with pytest.raises(ContractError):
        trainer.relation_loss(out, dataclasses.replace(targets, **fields))
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert torch.equal(old_hidden, out.new_states[0].hidden)
    assert all(param.grad is None for param in model.parameters())


def test_train_eval_state_mix_and_foreign_model_graph_are_rejected():
    p, _, model, _, batch, states, kw, _ = setup()
    out = model(batch, states, **kw)
    with pytest.raises(ContractError): model(batch, out.new_states, **dict(kw, mode=RunMode.TEST))
    with pytest.raises(ContractError):
        model(batch, (new_state("owner-a", "episode-0", seed=31101),), **kw)
    other = p.TextRelationModel("owner-a", seed=31101)
    with pytest.raises(ContractError): other(batch, out.new_states, **kw)


def test_reset_train_state_cuts_only_that_episode_graph():
    p, torch, model, _, batch, states, kw, _ = setup()
    out = model(batch, states, **kw)
    old = out.new_states[0]
    clean = p.reset_train_state(old)
    assert clean.owner_id == old.owner_id and clean.episode_id == old.episode_id and clean.seed == old.seed
    assert clean.hidden.shape == (64,) and clean.hidden.grad_fn is None
    assert torch.count_nonzero(clean.hidden).item() == 0
    assert clean.hidden.data_ptr() != old.hidden.data_ptr()
    assert old.hidden.grad_fn is not None


def test_independent_optimizers_and_train_step_do_not_touch_other_individual():
    p, torch, a, trainer_a, batch, states, kw, targets = setup()
    b = p.TextRelationModel("owner-b", seed=31101)
    trainer_b = p.TrainOnlySupervisor(b, owner_id="owner-b", seed=31101, mode=RunMode.TRAIN, lr=0.001)
    ptrs_a = {q.data_ptr() for group in trainer_a.optimizer.param_groups for q in group["params"]}
    ptrs_b = {q.data_ptr() for group in trainer_b.optimizer.param_groups for q in group["params"]}
    assert ptrs_a.isdisjoint(ptrs_b)
    before_b = {k: v.clone() for k, v in b.state_dict().items()}
    before_a = {k: v.clone() for k, v in a.state_dict().items()}
    trainer_a.relation_loss(a(batch, states, **kw), targets).backward()
    trainer_a.optimizer.step()
    assert any(not torch.equal(before_a[k], v) for k, v in a.state_dict().items())
    assert all(torch.equal(before_b[k], v) for k, v in b.state_dict().items())
    assert trainer_b.optimizer.state == {} and all(param.grad is None for param in b.parameters())


def test_empty_train_batch_is_cpu_even_under_other_default_device():
    # A device-default setting must not route this CPU-only port elsewhere.
    _, torch, model, _, batch, states, kw, _ = setup(texts=())
    with torch.device("meta"):
        out = model(batch, states, **kw)
    assert out.relation_logits.device.type == "cpu" and out.allowed_logits.device.type == "cpu"
    assert out.relation_logits.shape == (0, 4, 4) and out.new_states == ()
