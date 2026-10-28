#!/usr/bin/env python3
"""Capture immutable metadata/hashes for a completed training or evaluation run."""
from __future__ import annotations
import argparse, hashlib, json, platform, subprocess, sys
from pathlib import Path
from datetime import datetime, timezone

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def pkg_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception: return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",required=True)
    ap.add_argument("--run-type",required=True)
    ap.add_argument("--seed",type=int,required=True)
    ap.add_argument("--config")
    ap.add_argument("--split",action="append",default=[])
    ap.add_argument("--artifact",action="append",default=[])
    args=ap.parse_args()
    obj={"created_utc":datetime.now(timezone.utc).isoformat(),"run_type":args.run_type,"seed":args.seed,
         "python":sys.version,"platform":platform.platform(),
         "software":{"torch":pkg_version("torch"),"transformers":pkg_version("transformers"),"peft":pkg_version("peft"),"numpy":pkg_version("numpy"),"pandas":pkg_version("pandas")},
         "config":None,"splits":[],"artifacts":[]}
    if args.config:
        p=Path(args.config); obj["config"]={"path":str(p),"sha256":sha256(p)}
    for item in args.split:
        p=Path(item); obj["splits"].append({"path":str(p),"sha256":sha256(p)})
    for item in args.artifact:
        p=Path(item); obj["artifacts"].append({"path":str(p),"bytes":p.stat().st_size,"sha256":sha256(p)})
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(obj,indent=2),encoding="utf-8")
    print(out)
if __name__=="__main__": main()
