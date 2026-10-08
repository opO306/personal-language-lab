"""Small-test guardian; not the future P06 model runner."""
import ctypes as c
from ctypes import wintypes as w
import datetime as dt
import json, os, pathlib, subprocess, sys, time, uuid
from execution_binding import capture_execution_binding, verify_execution_binding

sys.stdout.reconfigure(encoding="utf-8",errors="backslashreplace")
ROOT=pathlib.Path(__file__).resolve().parents[1]
GIB=1024**3
class IO(c.Structure):
    _fields_=[(n,c.c_ulonglong) for n in ("ReadOperationCount","WriteOperationCount","OtherOperationCount","ReadTransferCount","WriteTransferCount","OtherTransferCount")]
class BASIC(c.Structure):
    _fields_=[("PerProcessUserTimeLimit",c.c_longlong),("PerJobUserTimeLimit",c.c_longlong),("LimitFlags",w.DWORD),("MinimumWorkingSetSize",c.c_size_t),("MaximumWorkingSetSize",c.c_size_t),("ActiveProcessLimit",w.DWORD),("Affinity",c.c_size_t),("PriorityClass",w.DWORD),("SchedulingClass",w.DWORD)]
class LIMIT(c.Structure):
    _fields_=[("BasicLimitInformation",BASIC),("IoInfo",IO),("ProcessMemoryLimit",c.c_size_t),("JobMemoryLimit",c.c_size_t),("PeakProcessMemoryUsed",c.c_size_t),("PeakJobMemoryUsed",c.c_size_t)]
class MEMORY(c.Structure):
    _fields_=[("length",w.DWORD),("load",w.DWORD)]+[(n,c.c_ulonglong) for n in ("total_phys","avail_phys","total_page","avail_page","total_virtual","avail_virtual","avail_extended")]
k=c.WinDLL("kernel32",use_last_error=True)
k.CreateJobObjectW.argtypes=[c.c_void_p,w.LPCWSTR];k.CreateJobObjectW.restype=w.HANDLE
k.GetCurrentProcess.restype=w.HANDLE
k.SetInformationJobObject.argtypes=[w.HANDLE,c.c_int,c.c_void_p,w.DWORD]
k.QueryInformationJobObject.argtypes=[w.HANDLE,c.c_int,c.c_void_p,w.DWORD,c.c_void_p]
k.AssignProcessToJobObject.argtypes=[w.HANDLE,w.HANDLE]
k.GetProcessAffinityMask.argtypes=[w.HANDLE,c.POINTER(c.c_size_t),c.POINTER(c.c_size_t)]
k.SetProcessAffinityMask.argtypes=[w.HANDLE,c.c_size_t]
k.TerminateJobObject.argtypes=[w.HANDLE,w.UINT]
k.GlobalMemoryStatusEx.argtypes=[c.POINTER(MEMORY)]
k.GetProcessTimes.argtypes=[w.HANDLE,c.POINTER(w.FILETIME),c.POINTER(w.FILETIME),c.POINTER(w.FILETIME),c.POINTER(w.FILETIME)]
def checked(value):
    if not value:raise c.WinError(c.get_last_error())
    return value
def available():
    m=MEMORY();m.length=c.sizeof(m);checked(k.GlobalMemoryStatusEx(c.byref(m)));return m.avail_phys
def creation(handle=None):
    fields=[w.FILETIME() for _ in range(4)]
    checked(k.GetProcessTimes(handle or k.GetCurrentProcess(),*(c.byref(f) for f in fields)))
    return (fields[0].dwHighDateTime<<32)|fields[0].dwLowDateTime
def main():
    os.chdir(ROOT)
    ident="unit-"+dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:6]
    folder=ROOT/"runs"/ident;folder.mkdir(parents=True)
    rec={"run_id":ident,"root_pid":os.getpid(),"root_creation_filetime":creation(),"start_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"argv":sys.argv[1:],"deadline_seconds":1500,"cleanup_reserve_seconds":300,"status":"preparing","resource_caps":{"logical_cpus":6,"job_memory_bytes":8*GIB,"minimum_available_bytes":4*GIB}}
    def save():
        (folder/"execution.json").write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding="utf-8")
    save()
    if available()<4*GIB:
        rec["status"]="low_system_memory_before_start";save();return 2
    proc=k.GetCurrentProcess();allowed=c.c_size_t();system=c.c_size_t()
    checked(k.GetProcessAffinityMask(proc,c.byref(allowed),c.byref(system)))
    bits=[1<<i for i in range(c.sizeof(c.c_size_t)*8) if allowed.value & (1<<i)]
    mask=sum(bits[:6])
    job=checked(k.CreateJobObjectW(None,None))
    limits=LIMIT();limits.BasicLimitInformation.LimitFlags=0x2000|0x200|0x10
    limits.BasicLimitInformation.Affinity=mask;limits.JobMemoryLimit=8*GIB
    checked(k.SetInformationJobObject(job,9,c.byref(limits),c.sizeof(limits)))
    checked(k.AssignProcessToJobObject(job,proc));checked(k.SetProcessAffinityMask(proc,mask))
    actual=LIMIT();checked(k.QueryInformationJobObject(job,9,c.byref(actual),c.sizeof(actual),None))
    if actual.JobMemoryLimit!=8*GIB or actual.BasicLimitInformation.Affinity!=mask or actual.BasicLimitInformation.LimitFlags & (0x2000|0x200|0x10)!=(0x2000|0x200|0x10):
        raise RuntimeError("JOB_LIMIT_READBACK_MISMATCH")
    rec["caps_verified"]=True;rec["affinity_mask"]=mask;rec["job_limit_flags"]=actual.BasicLimitInformation.LimitFlags
    env=os.environ.copy();env.update({n:"6" for n in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS")})
    env["PYTHONPATH"]=str(ROOT/"src");env["PYTHONDONTWRITEBYTECODE"]="1";env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"]="1";env["PYTHONIOENCODING"]="utf-8"
    args=sys.argv[1:]
    def binding_resources():
        if available()<4*GIB:raise RuntimeError("LOW_SYSTEM_MEMORY_STOP_DURING_BINDING")
    cmd=[sys.executable,"-B","-m","pytest",*args] if args!=["--probe"] else [sys.executable,"-B","-c","import ctypes,os; print('small child',os.getpid())"]
    if args[:1]==["--tool"]:
        if len(args)!=2 or args[1] not in ("preserve_base.py","review_package.py","limit_probe.py"):
            raise ValueError("UNAPPROVED_TOOL")
        cmd=[sys.executable,"-B",str(ROOT/"tools"/args[1])]
    execution_paths=(pathlib.Path(__file__),pathlib.Path(__file__).with_name("execution_binding.py"))
    rec["binding_before"]=capture_execution_binding(ROOT,cmd,execution_paths=execution_paths,check_resources=binding_resources)
    rec["binding_unchanged"]=None
    rec["command"]=cmd;save();begin=time.monotonic()
    with (folder/"stdout.txt").open("w",encoding="utf-8") as output:
        child=subprocess.Popen(cmd,env=env,stdout=output,stderr=subprocess.STDOUT)
        rec["child_pid"]=child.pid;rec["child_creation_filetime"]=creation(w.HANDLE(int(child._handle)));rec["status"]="running";save()
        while child.poll() is None:
            reason="low_system_memory" if available()<4*GIB else ("computation_deadline" if time.monotonic()-begin>=1500 else None)
            if reason:
                rec["status"]=reason;rec["end_utc"]=dt.datetime.now(dt.timezone.utc).isoformat();save()
                checked(k.TerminateJobObject(job,3));return 3
            time.sleep(.1)
        rec["exit_code"]=child.returncode;rec["child_exit_code"]=child.returncode;rec["child_closed"]=True
    checked(k.QueryInformationJobObject(job,9,c.byref(actual),c.sizeof(actual),None))
    rec.update(status="completed",end_utc=dt.datetime.now(dt.timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-begin,peak_job_memory_bytes=actual.PeakJobMemoryUsed,system_available_bytes_after=available())
    rec["guardian_exit_code"]=child.returncode
    try:
        rec["binding_after"]=verify_execution_binding(ROOT,rec["binding_before"],cmd,execution_paths=execution_paths,check_resources=binding_resources)
        rec["binding_unchanged"]=True
    except Exception as exc:
        rec["binding_unchanged"]=False;rec["binding_error"]=str(exc);rec["status"]="source_config_input_binding_failed";rec["guardian_exit_code"]=2
    save()
    brief={key:value for key,value in rec.items() if key not in ("binding_before","binding_after")}
    brief["binding_before_sha256"]=rec["binding_before"]["sha256"]
    brief["binding_after_sha256"]=rec.get("binding_after",{}).get("sha256")
    print(json.dumps(brief,ensure_ascii=False));print((folder/"stdout.txt").read_text(encoding="utf-8"));sys.stdout.flush()
    # OS closes the job handle when this guardian exits; KILL_ON_JOB_CLOSE cleans descendants.
    return rec["guardian_exit_code"]
if __name__=="__main__":
    try:sys.exit(main())
    except Exception as exc:
        print("GUARDIAN_FAILED_CLOSED",repr(exc),flush=True);sys.exit(2)
