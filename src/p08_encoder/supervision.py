"""Cooperative train-only capability, separate from target-free model forward."""
from dataclasses import dataclass
import math
import torch
from torch.nn import functional as F
from idea_model.contracts import ContractError, RunMode, require
from .model import PredictionBatch, TextRelationModel, RELATION_NAMES
from .vocabulary import FAMILY, PORT


@dataclass(frozen=True)
class TrainTargets:
    item_ids: tuple[str, ...]
    episode_ids: tuple[str, ...]
    relations: torch.Tensor
    allowed: torch.Tensor
    schema_version: int = 1


class TrainOnlySupervisor:
    """Owns this individual's optimizer. Training steps are caller-controlled, never inference."""
    def __init__(self, model: TextRelationModel, *, owner_id: str, seed: int, mode: RunMode, lr: float):
        try:
            requested = RunMode(mode)
        except (ValueError, TypeError) as exc:
            raise ContractError("supervision mode") from exc
        require(requested is RunMode.TRAIN, "train-only supervision")
        require(type(model) is TextRelationModel and type(owner_id) is str and owner_id == model.owner_id,
                "supervisor owner/model")
        require(type(seed) is int and seed == model.seed, "supervisor seed")
        require(type(lr) in (float, int) and math.isfinite(lr) and lr > 0, "optimizer learning rate")
        self.model, self.owner_id, self.seed = model, owner_id, seed
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def _validate(self, prediction, targets):
        require(type(prediction) is PredictionBatch and prediction.mode is RunMode.TRAIN, "train prediction required")
        require(prediction.model_key == self.model._instance_key and prediction.owner_id == self.owner_id and
                prediction.seed == self.seed, "prediction model/owner/seed")
        require(type(prediction.schema_version) is int and prediction.schema_version == 1 and
                prediction.family == FAMILY and prediction.port == PORT and prediction.relation_names == RELATION_NAMES,
                "prediction schema/port")
        require(type(targets) is TrainTargets and type(targets.schema_version) is int and targets.schema_version == 1,
                "train target schema/type; generic TrainingTarget/TestTarget is not consumed")
        require(type(targets.item_ids) is tuple and type(targets.episode_ids) is tuple and
                targets.item_ids == prediction.item_ids and targets.episode_ids == prediction.episode_ids,
                "train target item/episode mismatch")
        rows = len(prediction.item_ids)
        for values in (targets.relations, targets.allowed):
            require(type(values) is torch.Tensor and values.dtype == torch.int64 and values.device.type == "cpu",
                    "train target CPU int64 required")
        require(targets.relations.shape == (rows, 4) and targets.allowed.shape == (rows,), "train target shapes")
        require(bool(((targets.relations >= 0) & (targets.relations < 4)).all()) and
                bool(((targets.allowed >= 0) & (targets.allowed < 2)).all()), "train target ranges")
        observed = prediction.observed_mask
        require(type(observed) is torch.Tensor and observed.dtype == torch.bool and
                observed.device.type == "cpu" and observed.shape == (rows,), "observed mask")
        for values, shape in ((prediction.relation_logits, (rows, 4, 4)), (prediction.allowed_logits, (rows, 2))):
            require(type(values) is torch.Tensor and values.dtype == torch.float32 and values.device.type == "cpu" and
                    values.shape == shape and values.requires_grad and bool(torch.isfinite(values).all()),
                    "train predicted logits")
        return observed

    def relation_loss(self, prediction: PredictionBatch, targets: TrainTargets) -> torch.Tensor:
        require(not torch.is_inference_mode_enabled(), "TRAIN loss denied inside inference_mode")
        observed = self._validate(prediction, targets)
        # All validation finishes before a loss is built; malformed labels leave weights/state/grads untouched.
        with torch.enable_grad():
            valid_episodes = int(observed.sum().item())
            if valid_episodes == 0:
                return (prediction.relation_logits.sum() + prediction.allowed_logits.sum()) * 0.0
            relation = F.cross_entropy(prediction.relation_logits[observed].reshape(-1, 4),
                                       targets.relations[observed].reshape(-1), reduction="sum")
            allowed = F.cross_entropy(prediction.allowed_logits[observed], targets.allowed[observed], reduction="sum")
            # Five equal heads per valid episode. Other unimplemented candidate losses are not redistributed.
            return (relation + allowed) / (valid_episodes * 5)
