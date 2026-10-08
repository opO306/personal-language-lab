"""Lossless Unicode surface plus deterministic modern Hangul jamo IDs."""
from .contracts import EncodedText,validate_contract,ContractError
class CodecError(ValueError):pass
class LengthError(CodecError):pass
def encode_text(text:str,*,max_bytes:int=512)->EncodedText:
    if type(text)is not str or type(max_bytes)is not int or max_bytes<=0:
        raise CodecError("text/byte bound")
    try:raw=text.encode("utf-8")
    except UnicodeEncodeError as exc:raise CodecError("surrogate is not Unicode scalar text") from exc
    if len(raw)>max_bytes:raise LengthError(f"{len(raw)} > {max_bytes}")
    ids=[];spans=[]
    for char in text:
        start=len(ids);cp=ord(char)
        if 0xac00<=cp<=0xd7a3:
            syllable=cp-0xac00
            scalars=[0x1100+syllable//588,0x1161+(syllable%588)//28]
            if syllable%28:scalars.append(0x11a7+syllable%28)
        else:scalars=[cp]
        ids.extend(x+1 for x in scalars)  # ID0 reserved solely for padding.
        spans.append((start,len(ids)))
    return EncodedText(raw,tuple(ids),tuple(spans))
def decode_text(value:EncodedText)->str:
    if not isinstance(value,EncodedText):raise CodecError("EncodedText required")
    try:
        validate_contract(value);text=value.raw_utf8.decode("utf-8")
        expected=encode_text(text,max_bytes=max(1,len(value.raw_utf8)))
    except (ContractError,UnicodeDecodeError) as exc:raise CodecError("invalid encoded surface") from exc
    if value.jamo_ids!=expected.jamo_ids or value.reconstruction_map!=expected.reconstruction_map:
        raise CodecError("reconstruction mismatch")
    return text
