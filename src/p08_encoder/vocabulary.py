"""Fixed327 surface port over the byte-pinned, lossless P02 scalar codec."""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
from idea_model import contracts, codec, batching
from idea_model.contracts import EncodedText, EncodedBatch, ContractError, require, validate_contract
from idea_model.codec import CodecError, LengthError, decode_text

VOCAB_SIZE = 327
PAD_ID, BYTE_OFFSET, BYTE_START, BYTE_END, UNK_ID = 0, 68, 324, 325, 326
MAX_BYTES, MAX_TOKENS = 512, 1536
FAMILY, PORT, PORT_VERSION = "idea-relational-gru-v1", "text-relations-327-v1", 1
P02_HASHES = {
    "contracts": "9000ca9aa47fb0543a325ea073f439c4f72940956986d4eb466d027260848180",
    "codec": "27e4f9e8ffd88e7eb7dc4a4c1fc6c52782be3cb4e99b10f2aeda0bd2a69a8857",
    "batching": "645180a7590d83c4f51c71f24d004047c0725f646d508976a6660ea26b3d247d",
}
_MODERN = tuple(range(0x1100, 0x1113)) + tuple(range(0x1161, 0x1176)) + tuple(range(0x11A8, 0x11C3))
_JAMO_TO_ID = {scalar: index + 1 for index, scalar in enumerate(_MODERN)}


def verify_p02_sources() -> dict[str, str]:
    """Bind active import locations, including during integration, to schema1 bytes."""
    actual = {module.__name__.rsplit(".", 1)[-1]: sha256(Path(module.__file__).read_bytes()).hexdigest()
              for module in (contracts, codec, batching)}
    require(actual == P02_HASHES, "P02 source hash mismatch")
    return actual


# One startup check; forwards do not rescan the original project.
verify_p02_sources()


@dataclass(frozen=True)
class FixedTokenText:
    ids: tuple[int, ...]
    encoded: EncodedText
    port_version: int = PORT_VERSION


@dataclass(frozen=True)
class FixedTokenBatch:
    ids: tuple[tuple[int, ...], ...]
    lengths: tuple[int, ...]
    valid_mask: tuple[tuple[bool, ...], ...]
    item_ids: tuple[str, ...]
    schema_version: int = 1
    port: str = PORT


@dataclass(frozen=True)
class TensorBatch:
    ids: Any
    lengths: Any
    valid_mask: Any
    item_ids: tuple[str, ...]
    schema_version: int = 1
    port: str = PORT


def to_fixed_tokens(value: EncodedText, *, max_tokens: int = MAX_TOKENS) -> FixedTokenText:
    if type(max_tokens) is not int or not 0 < max_tokens <= MAX_TOKENS:
        raise CodecError("token bound")
    # This verifies original surface, jamo_ids and reconstruction_map together.
    decode_text(value)
    if len(value.raw_utf8) > MAX_BYTES:
        raise LengthError(f"{len(value.raw_utf8)} > {MAX_BYTES} raw bytes")
    ids = []
    for source_id in value.jamo_ids:
        scalar = source_id - 1
        modern_id = _JAMO_TO_ID.get(scalar)
        if modern_id is not None:
            ids.append(modern_id)
        else:
            ids.append(BYTE_START)
            ids.extend(BYTE_OFFSET + byte for byte in chr(scalar).encode("utf-8"))
            ids.append(BYTE_END)
    if len(ids) > max_tokens:
        raise LengthError(f"{len(ids)} > {max_tokens} model tokens")
    return FixedTokenText(tuple(ids), value)


def validate_fixed_batch(value: FixedTokenBatch) -> None:
    require(type(value) is FixedTokenBatch, "fixed327 batch type required; P02 scalar batch is a different port")
    require(type(value.schema_version) is int and value.schema_version == 1 and value.port == PORT, "fixed327 schema/port")
    require(all(type(field) is tuple for field in (value.ids, value.lengths, value.valid_mask, value.item_ids)), "fixed327 tuple fields")
    require(all(type(row) is tuple for row in value.ids + value.valid_mask), "fixed327 tuple rows")
    compatible = EncodedBatch(value.ids, value.lengths, value.valid_mask, value.item_ids)
    validate_contract(compatible)
    width = len(value.ids[0]) if value.item_ids else 0
    require(width <= MAX_TOKENS, "fixed327 length bound")
    require(all(token < VOCAB_SIZE for row in value.ids for token in row), "unsupported fixed327 ID")


def make_fixed_batch(items: tuple[EncodedText, ...], *, item_ids: tuple[str, ...],
                     max_tokens: int = MAX_TOKENS) -> FixedTokenBatch:
    require(type(items) is tuple and type(item_ids) is tuple and len(items) == len(item_ids), "fixed327 items/identities")
    if type(max_tokens) is not int or not 0 < max_tokens <= MAX_TOKENS:
        raise CodecError("token bound")
    converted = tuple(to_fixed_tokens(item, max_tokens=max_tokens) for item in items)
    lengths = tuple(len(item.ids) for item in converted)
    width = max(lengths, default=0)
    result = FixedTokenBatch(tuple(item.ids + (PAD_ID,) * (width - len(item.ids)) for item in converted),
                             lengths, tuple(tuple(index < length for index in range(width)) for length in lengths), item_ids)
    validate_fixed_batch(result)
    return result


def to_tensor_batch(value: FixedTokenBatch) -> TensorBatch:
    validate_fixed_batch(value)
    import torch
    rows = len(value.item_ids)
    width = len(value.ids[0]) if rows else 0
    return TensorBatch(torch.tensor(value.ids, dtype=torch.int64, device="cpu").reshape(rows, width),
                       torch.tensor(value.lengths, dtype=torch.int64, device="cpu"),
                       torch.tensor(value.valid_mask, dtype=torch.bool, device="cpu").reshape(rows, width),
                       value.item_ids)
