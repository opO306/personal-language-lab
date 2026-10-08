"""Independent synthetic RED fixtures. Do not run without the parent's slot."""
import dataclasses
import importlib
import importlib.util
import pytest
from idea_model.codec import encode_text, decode_text, CodecError, LengthError
from idea_model.contracts import ContractError


def api():
    assert importlib.util.find_spec("p08_encoder") is not None, "P08 component absent: fixed327 implementation required"
    return importlib.import_module("p08_encoder")


def test_fixed327_mapping_matches_p02_and_byte_fallback():
    # Reordering jamo IDs or using scalar+1 dense addresses must fail this test.
    p = api()
    encoded = encode_text("각각A🙂\x00")
    result = p.to_fixed_tokens(encoded)
    assert p.VOCAB_SIZE == 327
    assert result.encoded is encoded
    assert result.ids == (1, 20, 41, 1, 20, 41, 324, 133, 325,
                          324, 308, 227, 221, 198, 325, 324, 68, 325)
    assert decode_text(result.encoded) == "각각A🙂\x00"
    assert 0 not in result.ids and 326 not in result.ids


def test_all_67_modern_jamo_have_exact_fixed_ids():
    p = api()
    scalars = tuple(range(0x1100, 0x1113)) + tuple(range(0x1161, 0x1176)) + tuple(range(0x11A8, 0x11C3))
    assert p.to_fixed_tokens(encode_text("".join(map(chr, scalars)))).ids == tuple(range(1, 68))


@pytest.mark.parametrize("text", ["", "ㄱㅏ", "가 각", "e\u0301", "é", "A7!?🙂", "\x00", "\U0010ffff"])
def test_unicode_surface_is_preserved_without_normalization(text):
    p = api()
    encoded = encode_text(text)
    fixed = p.to_fixed_tokens(encoded)
    assert fixed.encoded == encoded and decode_text(fixed.encoded) == text
    assert all(0 < token < 327 for token in fixed.ids)


def test_ascii_byte_bound_expands_to_1536_tokens_without_truncation():
    p = api()
    encoded = encode_text("A" * 512)
    assert len(p.to_fixed_tokens(encoded).ids) == 1536
    with pytest.raises(LengthError):
        p.to_fixed_tokens(encoded, max_tokens=1535)
    with pytest.raises(LengthError):
        p.to_fixed_tokens(encode_text("A" * 513))


@pytest.mark.parametrize("limit", [True, 0, -1, 1.5])
def test_bad_token_bound_is_explicit_error(limit):
    with pytest.raises(CodecError):
        api().to_fixed_tokens(encode_text("A"), max_tokens=limit)


def test_forged_codec_payload_and_unknown_schema_rejected():
    p = api()
    original = encode_text("A")
    for forged in (dataclasses.replace(original, jamo_ids=(999,)),
                   dataclasses.replace(original, reconstruction_map=()),
                   dataclasses.replace(original, schema_version=2)):
        with pytest.raises((CodecError, ContractError)):
            p.to_fixed_tokens(forged)
    with pytest.raises(CodecError):
        p.to_fixed_tokens("unencoded")


def test_surrogate_rejected_before_model_input():
    api()
    with pytest.raises(CodecError):
        encode_text("\ud800")


def test_fixed_batch_preserves_item_ids_lengths_mask_and_empty_rows():
    p = api()
    batch = p.make_fixed_batch((encode_text("각"), encode_text(""), encode_text("A")),
                              item_ids=("ko", "empty", "ascii"))
    assert batch.item_ids == ("ko", "empty", "ascii")
    assert batch.lengths == (3, 0, 3)
    assert batch.ids == ((1, 20, 41), (0, 0, 0), (324, 133, 325))
    assert batch.valid_mask == ((True, True, True), (False, False, False), (True, True, True))
    empty = p.make_fixed_batch((), item_ids=())
    assert empty.ids == empty.lengths == empty.valid_mask == empty.item_ids == ()
    with pytest.raises(ContractError):
        p.make_fixed_batch((encode_text("a"), encode_text("b")), item_ids=("same", "same"))


def test_active_p02_module_bytes_match_locked_sources():
    # Source drift must fail before a new token port is accepted.
    assert api().verify_p02_sources() == {
        "contracts": "9000ca9aa47fb0543a325ea073f439c4f72940956986d4eb466d027260848180",
        "codec": "27e4f9e8ffd88e7eb7dc4a4c1fc6c52782be3cb4e99b10f2aeda0bd2a69a8857",
        "batching": "645180a7590d83c4f51c71f24d004047c0725f646d508976a6660ea26b3d247d",
    }


def test_p02_scalar_codec_and_fixed327_port_are_separate():
    p = api()
    from idea_model.batching import make_batch
    ga = encode_text("가")
    assert ga.jamo_ids == (4353, 4450)
    assert ga.reconstruction_map == ((0, 2),) and ga.raw_utf8 == "가".encode("utf-8")
    fixed = p.to_fixed_tokens(ga)
    assert fixed.ids == (1, 20) and fixed.encoded is ga
    old_ascii_batch = make_batch((encode_text("A"),))
    assert old_ascii_batch.ids == ((66,),)
    with pytest.raises(ContractError):
        p.to_tensor_batch(old_ascii_batch)
    new_batch = p.make_fixed_batch((ga,), item_ids=("ga",))
    assert type(new_batch) is p.FixedTokenBatch and new_batch.port == "text-relations-327-v1"
    assert new_batch.ids == ((1, 20),) and new_batch.item_ids == ("ga",)
