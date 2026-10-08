"""Versioned public contracts; targets never belong to Episode."""
from dataclasses import dataclass,field
from enum import Enum
from types import MappingProxyType
from typing import Any,Mapping
import math
class ContractError(ValueError):pass
class RunMode(str,Enum):
    TRAIN="train";SELECTION="selection";TEST="test"
class ResetScope(str,Enum):
    EPISODE="episode";INDIVIDUAL="individual"
class ReaderRole(str,Enum):
    MODEL="model";EVALUATOR="evaluator"
def version():return field(default=1,kw_only=True)
@dataclass(frozen=True)
class ObservationAtTime:
    time:float
    public_observations:tuple[str,...]
    schema_version:int=version()
@dataclass(frozen=True)
class PostActionFeedback:
    action_id:str
    result:str
    public_at:float
    provenance_hash:str
    schema_version:int=version()
@dataclass(frozen=True)
class Episode:
    episode_id:str
    split:str
    task_id:str
    input_text:str
    public_constraints:tuple[str,...]
    observation_at_time:ObservationAtTime
    provenance_hash:str
    schema_version:int=version()
@dataclass(frozen=True)
class TrainingTarget:
    episode_id:str
    target:bytes
    schema_version:int=version()
@dataclass(frozen=True)
class TestTarget:
    episode_id:str
    target:bytes
    schema_version:int=version()
@dataclass(frozen=True)
class MemoryRecord:
    memory_id:str
    owner_id:str
    source_episode_ids:tuple[str,...]
    representation_kind:str
    payload:bytes
    payload_bytes:int
    confidence:float
    created_step:int
    last_used_step:int
    archive_refs:tuple[str,...]
    schema_version:int=version()
@dataclass(frozen=True)
class MemoryView:
    records:tuple[MemoryRecord,...]=()
    total_bytes:int=0
    schema_version:int=version()
@dataclass
class CoreState:
    owner_id:str
    episode_id:str
    hidden:Any=()
    memory_view:MemoryView=field(default_factory=MemoryView)
    mutable_trace:dict=field(default_factory=dict)
    seed:int=0
    schema_version:int=version()
@dataclass(frozen=True)
class EncodedText:
    raw_utf8:bytes
    jamo_ids:tuple[int,...]
    reconstruction_map:tuple[tuple[int,int],...]
    schema_version:int=version()
@dataclass(frozen=True)
class EncodedBatch:
    ids:tuple[tuple[int,...],...]
    lengths:tuple[int,...]
    valid_mask:tuple[tuple[bool,...],...]
    item_ids:tuple[str,...]
    schema_version:int=version()
@dataclass(frozen=True)
class BatchedState:
    states:Mapping[str,CoreState]
    schema_version:int=version()
    def __post_init__(self):
        object.__setattr__(self,"states",MappingProxyType(dict(self.states)))
@dataclass(frozen=True)
class StepResult:
    output:Mapping[str,Any]
    new_states:BatchedState
    cost:Mapping[str,float]
    diagnostics:Mapping[str,Any]
    schema_version:int=version()
@dataclass(frozen=True)
class IdeaCandidate:
    candidate_id:str
    parent_ids:tuple[str,...]
    proposed_relations:tuple[str,...]
    text:str
    assumptions:tuple[str,...]
    predicted_outcomes:tuple[str,...]
    claim_status:str
    evidence_refs:tuple[str,...]
    schema_version:int=version()
@dataclass(frozen=True)
class LineageRecord:
    child_id:str
    parent_ids:tuple[str,...]
    parent_genotype_hashes:tuple[str,...]
    child_genotype_hash:str
    operator:str
    seed:int
    compatible_schema:int
    inherited_parts:tuple[str,...]
    rejected_parts:tuple[str,...]
    schema_version:int=version()
@dataclass(frozen=True)
class PilotConfig:
    split_sizes:Mapping[str,int]
    world_groups:tuple[str,...]
    max_bytes:int=512
    schema_version:int=version()
@dataclass(frozen=True)
class DatasetManifest:
    item_ids:Mapping[str,tuple[str,...]]
    world_ids:Mapping[str,tuple[str,...]]
    hashes:Mapping[str,str]
    generation_seed:int
    schema_version:int=version()
@dataclass(frozen=True)
class RunConfig:
    adapter_id:str
    dataset_manifest:DatasetManifest
    run_id:str
    deadlines:Mapping[str,float]
    resource_caps:Mapping[str,int]
    schema_version:int=version()
@dataclass(frozen=True)
class RunBundle:
    config:RunConfig
    source_hashes:Mapping[str,str]
    checkpoint_refs:tuple[str,...]
    logs:tuple[str,...]
    process_exit_evidence:Mapping[str,Any]
    schema_version:int=version()
@dataclass(frozen=True)
class AuditBudget:
    deadline:float
    cleanup_reserve:float
    schema_version:int=version()
def require(condition,message):
    if not condition:raise ContractError(message)
def strings(value):
    return isinstance(value,tuple) and all(type(x)is str for x in value)
def mutable_state_objects(*roots)->set[int]:
    """P01 storage policy: exact builtin containers and immutable scalar leaves.

    Arrays, tensors, memoryviews and custom objects/subclasses fail closed until
    a P04 adapter supplies a storage-aware ownership contract. Identity of an
    opaque view cannot prove that its backing storage is independent.
    Cycles and aliases inside one state are allowed; no state is frozen/copied.
    Callers must validate a quiescent batch before each use after reassignment.
    """
    pending=list(roots);seen=set();mutable=set()
    while pending:
        value=pending.pop();kind=type(value)
        if kind in (type(None),bool,int,float,complex,str,bytes):continue
        require(kind in (tuple,frozenset,list,dict,set,bytearray),"unsupported state storage")
        ident=id(value)
        if ident in seen:continue
        seen.add(ident)
        if kind in (list,dict,set,bytearray):mutable.add(ident)
        if kind is dict:
            pending.extend(value.keys());pending.extend(value.values())
        elif kind is not bytearray:pending.extend(value)
    return mutable

def validate_contract(value:object)->None:
    require(type(getattr(value,"schema_version",None))is int and value.schema_version==1,"unknown schema")
    if isinstance(value,ObservationAtTime):
        require(type(value.time)in (int,float) and math.isfinite(value.time) and value.time>=0,"observation time")
        require(strings(value.public_observations),"public observations")
    elif isinstance(value,Episode):
        require(value.split in ("train","selection","test"),"split")
        require(all(type(x)is str and x for x in (value.episode_id,value.task_id,value.provenance_hash)),"episode identity")
        require(type(value.input_text)is str and strings(value.public_constraints),"public fields")
        require(isinstance(value.observation_at_time,ObservationAtTime),"observation type")
        validate_contract(value.observation_at_time)
    elif isinstance(value,CoreState):
        require(all(type(x)is str and x for x in (value.owner_id,value.episode_id)),"state owner/episode")
        require(type(value.seed)is int and type(value.mutable_trace)is dict,"state fields")
        mutable_state_objects(value.hidden,value.mutable_trace)
        validate_contract(value.memory_view)
        require(all(r.owner_id==value.owner_id for r in value.memory_view.records),"memory owner mismatch")
    elif isinstance(value,MemoryRecord):
        require(all(type(x)is str and x for x in (value.memory_id,value.owner_id,value.representation_kind)),"memory identity")
        require(type(value.payload)is bytes and type(value.payload_bytes)is int and value.payload_bytes==len(value.payload),"payload bytes")
        require(strings(value.source_episode_ids) and strings(value.archive_refs),"memory refs")
        require(type(value.confidence)in (int,float) and math.isfinite(value.confidence) and 0<=value.confidence<=1,"confidence")
        require(type(value.created_step)is int and type(value.last_used_step)is int and 0<=value.created_step<=value.last_used_step,"memory steps")
    elif isinstance(value,MemoryView):
        require(type(value.records)is tuple and all(isinstance(r,MemoryRecord) for r in value.records),"read-only records")
        for r in value.records:validate_contract(r)
        require(type(value.total_bytes)is int and value.total_bytes==sum(r.payload_bytes for r in value.records),"memory total bytes")
    elif isinstance(value,BatchedState):
        require(all(type(k)is str and k for k in value.states),"item identities")
        require(len({id(s) for s in value.states.values()})==len(value.states),"shared CoreState")
        owners=set();owned_mutable=set()
        for state in value.states.values():
            require(isinstance(state,CoreState),"state type");validate_contract(state)
            key=(state.owner_id,state.episode_id)
            require(key not in owners,"duplicate owner/episode")
            owners.add(key)
            mutable=mutable_state_objects(state.hidden,state.mutable_trace)
            require(owned_mutable.isdisjoint(mutable),"shared mutable state descendant")
            owned_mutable.update(mutable)
    elif isinstance(value,IdeaCandidate):
        require(value.claim_status in ("unverified","supported","refuted","inconclusive"),"claim status")
    elif isinstance(value,EncodedText):
        require(type(value.raw_utf8)is bytes and type(value.jamo_ids)is tuple and all(type(x)is int and x>0 for x in value.jamo_ids),"encoded text")
        require(type(value.reconstruction_map)is tuple,"reconstruction map")
    elif isinstance(value,EncodedBatch):
        n=len(value.item_ids)
        require(len(set(value.item_ids))==n and strings(value.item_ids) and all(value.item_ids),"batch identities")
        require(len(value.ids)==len(value.lengths)==len(value.valid_mask)==n,"batch rows")
        width=len(value.ids[0]) if n else 0
        for ids,length,mask in zip(value.ids,value.lengths,value.valid_mask):
            require(type(length)is int and 0<=length<=width and len(ids)==len(mask)==width,"batch shape")
            require(all(type(x)is int and x>=0 for x in ids) and all(type(x)is bool for x in mask),"batch types")
            require(mask==tuple(i<length for i in range(width)),"valid mask")
            require(all((x>0 if i<length else x==0) for i,x in enumerate(ids)),"padding")
    elif isinstance(value,AuditBudget):
        require(math.isfinite(value.deadline) and math.isfinite(value.cleanup_reserve) and value.cleanup_reserve>=0,"audit budget")
    elif isinstance(value,PostActionFeedback):
        require(all(type(x)is str and x for x in (value.action_id,value.result,value.provenance_hash)),"feedback identity")
        require(type(value.public_at)in (int,float) and math.isfinite(value.public_at) and value.public_at>=0,"feedback time")
    elif isinstance(value,(TrainingTarget,TestTarget)):
        require(type(value.episode_id)is str and bool(value.episode_id) and type(value.target)is bytes,"target")
    elif isinstance(value,StepResult):
        require(isinstance(value.new_states,BatchedState),"step states");validate_contract(value.new_states)
        require(set(value.output)==set(value.new_states.states),"step items")
        require(all(type(v)in (int,float) and math.isfinite(v) and v>=0 for v in value.cost.values()),"step costs")
    elif isinstance(value,LineageRecord):
        require(all(type(x)is str and x for x in (value.child_id,value.child_genotype_hash,value.operator)),"lineage identity")
        require(strings(value.parent_ids) and strings(value.parent_genotype_hashes) and len(value.parent_ids)==len(value.parent_genotype_hashes),"parent hashes")
        require(type(value.seed)is int and type(value.compatible_schema)is int and value.compatible_schema==1,"lineage schema/seed")
        require(strings(value.inherited_parts) and strings(value.rejected_parts),"lineage parts")
    elif isinstance(value,PilotConfig):
        require(set(value.split_sizes)=={"train","selection","test"} and all(type(n)is int and n>0 for n in value.split_sizes.values()),"split sizes")
        require(strings(value.world_groups) and type(value.max_bytes)is int and value.max_bytes>0,"pilot bounds")
    elif isinstance(value,DatasetManifest):
        keys=set(value.item_ids)
        require(bool(keys) and keys<= {"train","selection","test"} and keys==set(value.world_ids)==set(value.hashes),"manifest splits")
        require(type(value.generation_seed)is int,"manifest seed")
        seen_items=set();seen_worlds=set()
        for split in keys:
            items=value.item_ids[split];worlds=value.world_ids[split]
            require(strings(items) and strings(worlds) and all(items) and all(worlds),"manifest identities")
            require(len(set(items))==len(items) and not seen_items.intersection(items),"duplicate/overlapping items")
            require(not seen_worlds.intersection(worlds),"overlapping worlds")
            require(type(value.hashes[split])is str and bool(value.hashes[split]),"manifest hash")
            seen_items.update(items);seen_worlds.update(worlds)
    elif isinstance(value,RunConfig):
        require(all(type(x)is str and x for x in (value.adapter_id,value.run_id)),"run identity")
        require(isinstance(value.dataset_manifest,DatasetManifest),"run manifest");validate_contract(value.dataset_manifest)
        require(bool(value.deadlines) and all(type(n)in (int,float) and math.isfinite(n) and n>0 for n in value.deadlines.values()),"deadlines")
        require(bool(value.resource_caps) and all(type(n)is int and n>0 for n in value.resource_caps.values()),"resource caps")
    elif isinstance(value,RunBundle):
        require(isinstance(value.config,RunConfig),"bundle config");validate_contract(value.config)
        require(bool(value.source_hashes) and all(type(k)is str and k and type(v)is str and v for k,v in value.source_hashes.items()),"source hashes")
        require(strings(value.checkpoint_refs) and strings(value.logs) and isinstance(value.process_exit_evidence,Mapping),"bundle refs")
    else:raise ContractError("unregistered contract")
def new_state(owner_id:str,episode_id:str,*,seed:int)->CoreState:
    state=CoreState(owner_id,episode_id,seed=seed);validate_contract(state);return state
def reset(state:CoreState,*,scope:ResetScope)->CoreState:
    validate_contract(state)
    try:ResetScope(scope)
    except (ValueError,TypeError) as exc:raise ContractError("reset scope") from exc
    # Memory retention policy is not silently enabled. P04 adapters must declare it.
    return new_state(state.owner_id,state.episode_id,seed=state.seed)