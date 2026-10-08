"""I-LANG-01 RED preparation. Unexecuted until the parent assigns the compute slot."""
import pytest


def fresh():
    # Imports and all model work occur only when pytest executes a test.
    import torch
    import p08_encoder as p
    from idea_model.codec import encode_text
    from idea_model.contracts import RunMode
    owner, episode, seed = "pilot-boundary-owner", "pilot-boundary-episode", 31101
    model = p.TextRelationModel(owner, seed=seed)
    batch = p.to_tensor_batch(p.make_fixed_batch(
        (encode_text("가A"),), item_ids=("pilot-boundary-item",)))
    state = p.new_train_state(owner, episode, seed=seed)
    kwargs = dict(owner_id=owner, episode_ids=(episode,), seed=seed, mode=RunMode.TRAIN)
    return torch, p, model, batch, state, kwargs


def snapshot(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def test_train_forward_rejects_external_inference_mode_before_state_return():
    # Regression mutation: remove the early TRAIN/inference-mode rejection.
    from idea_model.contracts import ContractError
    torch, _, model, batch, state, kwargs = fresh()
    before, hidden = snapshot(model), state.hidden.clone()
    visited = []
    handles = [module.register_forward_pre_hook(lambda module, args: visited.append(type(module).__name__))
               for module in (model.embedding, model.gru, *model.relation_heads, model.allowed_head)]
    try:
        with torch.inference_mode():
            with pytest.raises(ContractError, match="inference"):
                model(batch, (state,), **kwargs)
    finally:
        for handle in handles:
            handle.remove()
    assert visited == [], "rejection must precede all embedding/GRU/head computations"
    assert torch.equal(hidden, state.hidden) and state.observations == 0
    assert state.hidden.grad_fn is None and not state.hidden.requires_grad
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    assert all(param.grad is None for param in model.parameters())


def test_train_loss_rejects_external_inference_mode_before_detached_loss_return():
    # Regression mutation: remove the early relation_loss/inference-mode rejection.
    from idea_model.contracts import ContractError
    torch, p, model, batch, state, kwargs = fresh()
    prediction = model(batch, (state,), **kwargs)
    assert prediction.relation_logits.requires_grad
    assert prediction.new_states[0].hidden.grad_fn is not None
    supervisor = p.TrainOnlySupervisor(model, owner_id=kwargs["owner_id"],
                                      seed=kwargs["seed"], mode=kwargs["mode"], lr=0.001)
    targets = p.TrainTargets(batch.item_ids, kwargs["episode_ids"],
                            torch.tensor([[0, 1, 2, 3]], dtype=torch.int64, device="cpu"),
                            torch.tensor([1], dtype=torch.int64, device="cpu"))
    before, hidden = snapshot(model), prediction.new_states[0].hidden.detach().clone()
    with torch.inference_mode():
        with pytest.raises(ContractError, match="inference"):
            supervisor.relation_loss(prediction, targets)
    assert torch.equal(hidden, prediction.new_states[0].hidden)
    assert prediction.new_states[0].hidden.grad_fn is not None
    assert state.observations == 0 and supervisor.optimizer.state == {}
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    assert all(param.grad is None for param in model.parameters())
