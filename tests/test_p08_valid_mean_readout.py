"""Public safe subset of previously checked readout fixtures."""

import importlib, importlib.util

import pytest

def api():
    assert importlib.util.find_spec("p08_mean_readout") is not None,"separate valid-token mean readout absent"
    return importlib.import_module("p08_mean_readout")

def batch(texts):
    from idea_model.codec import encode_text
    from p08_encoder import make_fixed_batch,to_tensor_batch
    return to_tensor_batch(make_fixed_batch(tuple(encode_text(text) for text in texts),item_ids=tuple("i"+str(i) for i in range(len(texts)))))

def predict(model,values):
    from idea_model.contracts import RunMode,new_state
    states=tuple(new_state(model.owner_id,ident,seed=model.seed) for ident in values.item_ids)
    return model(values,states,owner_id=model.owner_id,episode_ids=values.item_ids,seed=model.seed,mode=RunMode.SELECTION)

def test_last_policy_preserves_type_weights_logits_and_parameter_count():
    p=api()
    import torch
    from p08_encoder import TextRelationModel
    original=TextRelationModel("fixture",seed=31101)
    last=p.build_model("fixture",seed=31101,readout="last")
    mean=p.build_model("fixture",seed=31101,readout="valid_token_mean")
    assert type(last) is type(mean) is TextRelationModel
    assert sum(v.numel() for v in mean.parameters())==30450
    assert original.state_dict().keys()==last.state_dict().keys()==mean.state_dict().keys()
    assert all(torch.equal(v,last.state_dict()[k]) and torch.equal(v,mean.state_dict()[k]) for k,v in original.state_dict().items())
    a,b=predict(original,batch(("1","2 ABC",""))),predict(last,batch(("1","2 ABC","")))
    assert torch.equal(a.relation_logits,b.relation_logits) and torch.equal(a.allowed_logits,b.allowed_logits)
    assert [s.hidden for s in a.new_states]==[s.hidden for s in b.new_states]

def test_mean_is_length_normalized_valid_state_average_and_keeps_recurrent_state():
    p=api()
    import torch
    from torch.nn.utils.rnn import pad_packed_sequence
    from p08_encoder import TextRelationModel
    model=p.build_model("fixture",seed=31101,readout="valid_token_mean")
    original=TextRelationModel("fixture",seed=31101)
    values=batch(("1","2 "+"가"*15,""))
    capture=[]
    handle=model.gru.register_forward_hook(lambda m,i,o:capture.append(pad_packed_sequence(o[0],batch_first=True)))
    observed=predict(model,values)
    handle.remove()
    sequence,lengths=capture[0]
    active=values.lengths.nonzero().flatten()
    assert lengths.tolist()==values.lengths[active].tolist()
    pooled=torch.zeros((3,64))
    pooled[active]=sequence.sum(1)/lengths[:,None]
    with torch.no_grad():
        expected=torch.stack([h(pooled) for h in model.relation_heads],dim=1)*observed.observed_mask[:,None,None]
        allowed=model.allowed_head(pooled)*observed.observed_mask[:,None]
    assert torch.allclose(observed.relation_logits,expected,atol=1e-7,rtol=1e-6)
    assert torch.allclose(observed.allowed_logits,allowed,atol=1e-7,rtol=1e-6)
    baseline=predict(original,values)
    assert [s.hidden for s in observed.new_states]==[s.hidden for s in baseline.new_states]
    assert not torch.equal(observed.relation_logits[:2],baseline.relation_logits[:2])

def test_extra_pad_changes_neither_readout_nor_recurrent_state():
    p=api()
    import torch
    from p08_encoder import TensorBatch
    model=p.build_model("fixture",seed=31101,readout="valid_token_mean")
    values=batch(("1","2 ABC",""))
    extra=TensorBatch(torch.cat((values.ids,torch.zeros((3,9),dtype=torch.int64)),1),values.lengths,torch.cat((values.valid_mask,torch.zeros((3,9),dtype=torch.bool)),1),values.item_ids)
    a,b=predict(model,values),predict(model,extra)
    assert torch.equal(a.relation_logits,b.relation_logits) and torch.equal(a.allowed_logits,b.allowed_logits)
    assert [s.hidden for s in a.new_states]==[s.hidden for s in b.new_states]

def test_bad_batch_failure_does_not_poison_next_forward():
    p=api()
    import torch
    from idea_model.contracts import ContractError
    from p08_encoder import TensorBatch
    model=p.build_model("fixture",seed=31101,readout="valid_token_mean");values=batch(("1","2"))
    expected=predict(model,values)
    with pytest.raises(ContractError):
        predict(model,TensorBatch(values.ids.float(),values.lengths,values.valid_mask,values.item_ids))
    observed=predict(model,values)
    assert torch.equal(expected.relation_logits,observed.relation_logits)
    assert [s.hidden for s in expected.new_states]==[s.hidden for s in observed.new_states]
