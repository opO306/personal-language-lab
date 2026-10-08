"""New CPU float32 GRU/head family; no labels, parser, shared memory or native replacement."""
from copy import deepcopy
from dataclasses import dataclass
import math
import uuid
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from idea_model.contracts import (
    BatchedState, CoreState, ContractError, RunMode, require, validate_contract,
)
from .vocabulary import TensorBatch, VOCAB_SIZE, MAX_TOKENS, FAMILY, PORT, verify_p02_sources

RELATION_NAMES = ("box", "tool", "from", "to")


def _identity(owner_id, episode_id, seed):
    require(type(owner_id) is str and bool(owner_id), "owner identity")
    require(type(episode_id) is str and bool(episode_id), "episode identity")
    require(type(seed) is int and 0 <= seed < 2**63, "model seed")


@dataclass(frozen=True)
class TrainEpisodeState:
    owner_id: str
    episode_id: str
    hidden: torch.Tensor
    seed: int
    model_key: str | None = None
    observations: int = 0
    schema_version: int = 1


def _validate_train_storage(state):
    require(type(state) is TrainEpisodeState and type(state.schema_version) is int and state.schema_version == 1,
            "train state schema/type")
    _identity(state.owner_id, state.episode_id, state.seed)
    h = state.hidden
    require(type(h) is torch.Tensor and h.device.type == "cpu" and h.dtype == torch.float32 and h.shape == (64,),
            "train hidden dtype/shape/device")
    require(bool(torch.isfinite(h).all()), "nonfinite train hidden")
    require(type(state.observations) is int and state.observations >= 0, "train observations")
    require(state.model_key is None or (type(state.model_key) is str and bool(state.model_key)), "train model identity")
    if state.observations == 0:
        require(h.grad_fn is None and not h.requires_grad and not bool(torch.count_nonzero(h)), "initial train state must be empty")
    else:
        require(state.model_key is not None and h.requires_grad and h.grad_fn is not None, "detached episode graph denied")


def new_train_state(owner_id: str, episode_id: str, *, seed: int) -> TrainEpisodeState:
    _identity(owner_id, episode_id, seed)
    return TrainEpisodeState(owner_id, episode_id, torch.zeros(64, dtype=torch.float32, device="cpu"), seed)


def reset_train_state(state: TrainEpisodeState) -> TrainEpisodeState:
    _validate_train_storage(state)
    return TrainEpisodeState(state.owner_id, state.episode_id,
                             torch.zeros(64, dtype=torch.float32, device="cpu"), state.seed, state.model_key)


@dataclass(frozen=True)
class PredictionBatch:
    relation_logits: torch.Tensor
    allowed_logits: torch.Tensor
    observed_mask: torch.Tensor
    new_states: tuple[CoreState | TrainEpisodeState, ...]
    item_ids: tuple[str, ...]
    episode_ids: tuple[str, ...]
    owner_id: str
    seed: int
    mode: RunMode
    model_key: str
    relation_names: tuple[str, ...] = RELATION_NAMES
    family: str = FAMILY
    port: str = PORT
    schema_version: int = 1


def validate_tensor_batch(batch: TensorBatch) -> None:
    require(type(batch) is TensorBatch, "TensorBatch required")
    require(type(batch.schema_version) is int and batch.schema_version == 1 and batch.port == PORT, "tensor schema/port")
    require(type(batch.item_ids) is tuple and all(type(x) is str and x for x in batch.item_ids) and
            len(set(batch.item_ids)) == len(batch.item_ids), "tensor item identities")
    for tensor in (batch.ids, batch.lengths, batch.valid_mask):
        require(type(tensor) is torch.Tensor and tensor.device.type == "cpu", "CPU tensors required")
    require(batch.ids.dtype == torch.int64 and batch.lengths.dtype == torch.int64 and
            batch.valid_mask.dtype == torch.bool, "input dtype")
    require(batch.ids.ndim == 2 and batch.lengths.ndim == 1 and batch.valid_mask.ndim == 2, "input dimensions")
    rows, width = batch.ids.shape
    require(rows == len(batch.item_ids) and batch.lengths.shape == (rows,) and batch.valid_mask.shape == (rows, width),
            "input shapes")
    require(width <= MAX_TOKENS, "model token length bound")
    require(bool(((batch.lengths >= 0) & (batch.lengths <= width)).all()), "input lengths")
    expected = torch.arange(width, device="cpu").unsqueeze(0) < batch.lengths.unsqueeze(1)
    require(torch.equal(batch.valid_mask, expected), "prefix valid mask")
    require(bool(((batch.ids >= 0) & (batch.ids < VOCAB_SIZE)).all()), "unsupported fixed327 ID")
    require(bool(torch.where(expected, batch.ids > 0, batch.ids == 0).all()), "PAD placement")


class TextRelationModel(nn.Module):
    """One model owns one individual, weights and seed; each call has explicit episode states."""
    def __init__(self, owner_id: str, *, seed: int):
        super().__init__()
        _identity(owner_id, "constructor", seed)
        verify_p02_sources()
        self.owner_id, self.seed = owner_id, seed
        self._instance_key = uuid.uuid4().hex
        # CPU generator only; constructing an individual leaves global CPU/GPU RNG untouched.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            options = dict(dtype=torch.float32, device="cpu")
            self.embedding = nn.Embedding(VOCAB_SIZE, 32, padding_idx=0, **options)
            self.gru = nn.GRU(32, 64, batch_first=True, **options)
            self.relation_heads = nn.ModuleList(nn.Linear(64, 4, **options) for _ in RELATION_NAMES)
            self.allowed_head = nn.Linear(64, 2, **options)

    def _validate_context(self, batch, states, *, owner_id, episode_ids, seed, mode):
        require(owner_id == self.owner_id and type(owner_id) is str, "model owner mismatch")
        require(type(seed) is int and seed == self.seed, "model seed mismatch")
        try:
            mode = RunMode(mode)
        except (TypeError, ValueError) as exc:
            raise ContractError("model mode") from exc
        require(mode is not RunMode.TRAIN or not torch.is_inference_mode_enabled(),
                "TRAIN forward denied inside inference_mode")
        validate_tensor_batch(batch)
        require(type(states) is tuple and type(episode_ids) is tuple and
                len(states) == len(episode_ids) == len(batch.item_ids), "state/episode rows")
        require(all(type(x) is str and x for x in episode_ids) and len(set(episode_ids)) == len(episode_ids),
                "episode identities/duplicates")
        require(len({id(state) for state in states}) == len(states), "shared state object")
        for param in self.parameters():
            require(param.dtype == torch.float32 and param.device.type == "cpu" and bool(torch.isfinite(param).all()),
                    "model parameter dtype/device/nonfinite")
            require(mode is not RunMode.TRAIN or param.requires_grad, "train weights must retain gradient")
        if mode is RunMode.TRAIN:
            pointers = set()
            for state, episode in zip(states, episode_ids):
                _validate_train_storage(state)
                require(state.owner_id == owner_id and state.episode_id == episode and state.seed == seed, "train state context")
                require(state.model_key in (None, self._instance_key), "foreign model episode graph")
                pointer = state.hidden.untyped_storage().data_ptr()
                require(pointer not in pointers, "shared train hidden storage")
                pointers.add(pointer)
        else:
            for state, episode in zip(states, episode_ids):
                require(type(state) is CoreState, "eval CoreState tuple snapshot required")
                validate_contract(state)
                require(state.owner_id == owner_id and state.episode_id == episode and state.seed == seed, "eval state context")
                h = state.hidden
                require(type(h) is tuple and (h == () or (len(h) == 64 and
                        all(type(value) is float and math.isfinite(value) for value in h))), "eval hidden tuple")
            validate_contract(BatchedState(dict(zip(batch.item_ids, states))))
        return mode

    def forward(self, batch: TensorBatch, states: tuple, *, owner_id: str,
                episode_ids: tuple[str, ...], seed: int, mode: RunMode) -> PredictionBatch:
        mode = self._validate_context(batch, states, owner_id=owner_id, episode_ids=episode_ids, seed=seed, mode=mode)
        # No state mutates in-place. Every row is validated before a GRU computation.
        with torch.autocast("cpu", enabled=False), (torch.enable_grad() if mode is RunMode.TRAIN else torch.no_grad()):
            rows = len(states)
            if mode is RunMode.TRAIN:
                initial = torch.stack([state.hidden for state in states]) if rows else torch.zeros((0, 64), dtype=torch.float32, device="cpu")
            else:
                initial = torch.tensor([state.hidden or (0.0,) * 64 for state in states],
                                       dtype=torch.float32, device="cpu").reshape(rows, 64)
            observed = batch.lengths > 0
            active = observed.nonzero(as_tuple=False).flatten()
            final = initial
            if active.numel():
                embeddings = self.embedding(batch.ids.index_select(0, active))
                packed = pack_padded_sequence(embeddings, batch.lengths.index_select(0, active),
                                               batch_first=True, enforce_sorted=False)
                _, active_hidden = self.gru(packed, initial.index_select(0, active).unsqueeze(0))
                final = initial.index_copy(0, active, active_hidden.squeeze(0))
            relation = torch.stack([head(final) for head in self.relation_heads], dim=1)
            allowed = self.allowed_head(final)
            relation = relation * observed[:, None, None]
            allowed = allowed * observed[:, None]
            require(bool(torch.isfinite(final).all()) and bool(torch.isfinite(relation).all()) and
                    bool(torch.isfinite(allowed).all()), "nonfinite model output")
            updated = []
            for index, state in enumerate(states):
                if not bool(observed[index]):
                    updated.append(state)
                elif mode is RunMode.TRAIN:
                    updated.append(TrainEpisodeState(owner_id, episode_ids[index], final[index].clone(), seed,
                                                     self._instance_key, state.observations + 1))
                else:
                    updated.append(CoreState(owner_id, episode_ids[index], hidden=tuple(map(float, final[index].tolist())),
                                             memory_view=state.memory_view, mutable_trace=deepcopy(state.mutable_trace), seed=seed))
            return PredictionBatch(relation, allowed, observed, tuple(updated), batch.item_ids, episode_ids,
                                   owner_id, seed, mode, self._instance_key)
