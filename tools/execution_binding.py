"""Start/end byte binding for small checks; no model/fixture payload decoding."""
import hashlib,json,os,pathlib
def packed(value):return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf-8")
def digest(value):return hashlib.sha256(packed(value)).hexdigest()
def capture_execution_binding(root,argv,*,execution_paths=(),check_resources=None):
    root=pathlib.Path(root).resolve()
    if not root.is_dir() or type(argv) not in (list,tuple) or not all(type(x)is str for x in argv):raise ValueError("binding root/argv")
    def resources():
        if check_resources is not None:check_resources()
    def boundary(path):
        if path.is_symlink() or (getattr(path,"is_junction",lambda:False)()):raise ValueError("binding symlink/junction refused")
        if not path.resolve().is_relative_to(root):raise ValueError("binding path outside root")
    def metadata(path):
        resources();boundary(path);h=hashlib.sha256();size=0
        with path.open("rb") as f:
            for chunk in iter(lambda:f.read(262144),b""):
                resources();h.update(chunk);size+=len(chunk)
        return {"bytes":size,"sha256":h.hexdigest()}
    resources()
    result={"schema":"execution-file-binding-v1","source_files":{},"config_files":{},"input_files":{},"argv":list(argv),"argv_sha256":digest(list(argv)),"input_policy":"fixture file bytes; generated fault/temp inputs are defined by pinned test source"}
    def scan(rel,key,code_only=False):
        folder=root/rel
        if not folder.exists():return
        boundary(folder)
        for base,dirs,names in os.walk(folder,followlinks=False):
            for name in list(dirs):
                p=pathlib.Path(base)/name;boundary(p)
                if name=="__pycache__":dirs.remove(name)
            for name in names:
                p=pathlib.Path(base)/name
                if not code_only or p.suffix==".py":result[key][p.relative_to(root).as_posix()]=metadata(p)
    sources=("src","tests","tools","workspaces/p03-current-base-20261007/current_base/iam","preserved/historical-b-15ae151/engine","preserved/historical-b-15ae151/tests","preserved/binary-originals-20261008-e515d7b0/sources/signal-growth-probe-v1","preserved/binary-originals-20261008-e515d7b0/sources/validated-growth-observer-v1")
    for rel in sources:scan(rel,"source_files",True)
    for rel in ("preserved/historical-b-15ae151/source_lock_complete.json","preserved/binary-originals-20261008-e515d7b0/source_import_manifest.json","preserved/binary-originals-20261008-e515d7b0/2026-10-07.zip","manifests/source_map_historical_b_and_binary_20261008.json"):
        p=root/rel
        if p.is_file():result["source_files"][rel]=metadata(p)
    scan("configs","config_files");scan("tests/fixtures","input_files")
    for rel in ("pytest.ini","reports/binary_validated_port_contract_KO_20261008.txt"):
        p=root/rel
        if p.is_file():result["config_files"][rel]=metadata(p)
    if type(execution_paths) not in (list,tuple):raise ValueError("execution_paths list/tuple required")
    explicit=set()
    def execution_path(value,required=False):
        path=pathlib.Path(value)
        if not path.is_absolute():path=root/path
        for part in (path,*path.parents):
            boundary(part)
            if part==root:break
        path=path.resolve()
        if not path.exists():
            if required:raise ValueError("execution file missing: "+str(path))
            return
        if path.is_dir():
            scan(path.relative_to(root),"source_files",True)
            folder=path
        else:
            if path.suffix!=".py":raise ValueError("execution Python script required")
            name=path.relative_to(root).as_posix();result["source_files"][name]=metadata(path);explicit.add(name);folder=path.parent
        for part in (folder,*folder.parents):
            candidate=part/"conftest.py"
            if candidate.exists():
                for parent in (candidate,*candidate.parents):
                    boundary(parent)
                    if parent==root:break
                result["source_files"][candidate.relative_to(root).as_posix()]=metadata(candidate)
            if part==root:break
    for value in execution_paths:execution_path(value,True)
    for value in argv:
        if value.startswith("-"):continue
        token=value.split("::",1)[0]
        path=pathlib.Path(token)
        candidate=path if path.is_absolute() else root/path
        if path.suffix==".py":execution_path(token,True)
        elif candidate.is_dir():execution_path(token)
    result["explicit_execution_files"]=sorted(explicit)
    result["execution_file_policy"]="actual argv Python/node-id files, explicit scripts, target directory Python and parent conftest; root boundary enforced"
    result["sha256"]=digest(result)
    return result
def verify_execution_binding(root,before,argv,*,execution_paths=(),check_resources=None):
    if type(before)is not dict or before.get("sha256")!=digest({k:v for k,v in before.items() if k!="sha256"}):
        raise RuntimeError("source/config/input binding changed: corrupted start snapshot")
    after=capture_execution_binding(root,argv,execution_paths=execution_paths,check_resources=check_resources)
    if before!=after:raise RuntimeError("source/config/input binding changed during check")
    return after
