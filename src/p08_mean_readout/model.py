"""Separate parameter-free readout adapter; original model/state implementation unchanged."""
import torch
from torch.nn.utils.rnn import pad_packed_sequence
from idea_model.contracts import require
from p08_encoder import TextRelationModel


class _MeanReadout:
    def __init__(self, model):
        self.batch = self.pooled = None
        self.handles = [
            model.register_forward_pre_hook(self.begin, with_kwargs=True),
            model.gru.register_forward_hook(self.capture),
            *[head.register_forward_pre_hook(self.read) for head in (*model.relation_heads, model.allowed_head)],
            model.register_forward_hook(self.finish, always_call=True),
        ]

    def begin(self, model, args, kwargs):
        require(self.batch is None, "mean readout cannot overlap one individual forward")
        self.batch = args[0] if args else kwargs["batch"]
        self.pooled = None

    def capture(self, gru, args, output):
        # Runs inside the original model's validated CPU/autograd context.
        sequence, lengths = pad_packed_sequence(output[0], batch_first=True)
        active = (self.batch.lengths > 0).nonzero(as_tuple=False).flatten()
        require(torch.equal(lengths, self.batch.lengths.index_select(0, active)), "mean active row order")
        valid = torch.arange(sequence.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
        sums = torch.where(valid.unsqueeze(-1), sequence, 0.0).sum(1)
        mean = sums / lengths.to(sequence.dtype).unsqueeze(1)
        self.pooled = sequence.new_zeros((self.batch.ids.shape[0], 64)).index_copy(0, active, mean)

    def read(self, head, args):
        if self.batch is None:
            return None
        if self.pooled is None:
            require(not bool((self.batch.lengths > 0).any()), "missing mean GRU capture")
            self.pooled = torch.zeros_like(args[0])
        return (self.pooled, *args[1:])

    def finish(self, model, args, output):
        # Also runs on validation/computation failure; no graph is retained between calls.
        self.batch = self.pooled = None


def build_model(owner_id, *, seed, readout="last"):
    require(readout in ("last", "valid_token_mean"), "readout policy")
    model = TextRelationModel(owner_id, seed=seed)
    model.readout_policy = readout
    if readout == "valid_token_mean":
        model._mean_readout = _MeanReadout(model)
    return model
