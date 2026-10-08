"""Immutable token batches. Consumers must use the validated mask."""
from .contracts import EncodedText,EncodedBatch,ContractError,validate_contract
from .codec import decode_text
def make_batch(items:tuple[EncodedText,...])->EncodedBatch:
    if type(items)is not tuple:raise ContractError("tuple input required")
    for item in items:decode_text(item)
    lengths=tuple(len(item.jamo_ids) for item in items)
    width=max(lengths,default=0)
    result=EncodedBatch(
        tuple(item.jamo_ids+(0,)*(width-len(item.jamo_ids)) for item in items),
        lengths,
        tuple(tuple(j<length for j in range(width)) for length in lengths),
        tuple(f"item-{i}" for i in range(len(items))))
    validate_contract(result);return result
def valid_tokens(batch:EncodedBatch,item_id:str)->tuple[int,...]:
    validate_contract(batch)
    try:index=batch.item_ids.index(item_id)
    except ValueError as exc:raise ContractError("unknown item") from exc
    return tuple(token for token,valid in zip(batch.ids[index],batch.valid_mask[index]) if valid)
