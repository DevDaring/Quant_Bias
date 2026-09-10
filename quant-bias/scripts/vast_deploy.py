#!/usr/bin/env python3
"""Provision a GPU host on Vast.ai.

    python3 vast_deploy.py create --offer 47195595 --disk 250
    python3 vast_deploy.py status
    python3 vast_deploy.py destroy

Unlike Akash, a Vast instance keeps its disk across restarts, so the checkpoint
cache survives. The instance id is recorded in state_vast.json so nothing is
left running by accident.
"""
from __future__ import annotations
import argparse, json, os, pathlib, sys, time, urllib.error, urllib.parse, urllib.request

API = "https://console.vast.ai/api/v0"
STATE = pathlib.Path(__file__).resolve().parent / "state_vast.json"


def key() -> str:
    k = os.environ.get("Vast_AI_PHD_API_KEY") or os.environ.get("VAST_API_KEY")
    if not k:
        sys.exit("Vast_AI_PHD_API_KEY not in environment")
    return k


def call(method: str, path: str, body: dict | None = None, timeout: int = 120):
    url = f"{API}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {key()}", "Content-Type": "application/json",
        "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
            return r.status, (json.loads(raw) if raw.strip() else {})
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, {"raw": raw[:600]}


def save(**kw):
    s = json.loads(STATE.read_text()) if STATE.exists() else {}
    s.update(kw); s["updated"] = time.strftime("%F %T")
    STATE.write_text(json.dumps(s, indent=2)); return s


def cmd_create(a):
    pub = pathlib.Path(a.pubkey).expanduser().read_text().strip()
    onstart = "\n".join([
        "#!/bin/bash",
        "mkdir -p /root/.ssh && chmod 700 /root/.ssh",
        f"echo '{pub}' >> /root/.ssh/authorized_keys",
        "chmod 600 /root/.ssh/authorized_keys",
        "touch /workspace/ONSTART_DONE 2>/dev/null || (mkdir -p /workspace && touch /workspace/ONSTART_DONE)",
    ])
    body = {
        "client_id": "me",
        "image": a.image,
        "disk": a.disk,
        "label": a.label,
        "runtype": "ssh",
        "onstart": onstart,
        "env": {"-p 22:22": "1"},
    }
    print(f"renting offer {a.offer} (image={a.image}, disk={a.disk}GB)")
    code, d = call("PUT", f"/asks/{a.offer}/", body)
    if code not in (200, 201) or not d.get("success", True):
        print(f"  failed [{code}]: {json.dumps(d)[:500]}"); return 2
    iid = d.get("new_contract") or d.get("id")
    print(f"  instance id = {iid}")
    save(instance_id=iid, offer=a.offer, created=time.strftime("%F %T"))

    print("waiting for the instance to come up ...")
    for i in range(a.wait // 10):
        time.sleep(10)
        code, s = call("GET", f"/instances/{iid}/")
        inst = (s or {}).get("instances") or {}
        st = inst.get("actual_status")
        host, port = inst.get("ssh_host"), inst.get("ssh_port")
        if st == "running" and host and port:
            save(ssh_host=host, ssh_port=port, status=st,
                 gpu=inst.get("gpu_name"), dph=inst.get("dph_total"))
            print(f"\nREADY  ssh -p {port} root@{host}")
            print(json.dumps({"gpu": inst.get("gpu_name"), "dph": inst.get("dph_total"),
                              "host": host, "port": port}, indent=2))
            return 0
        if i % 6 == 5:
            print(f"  {10*(i+1)}s status={st}")
    print("  did not reach running state in time; check `status`")
    return 3


def _iid(a):
    if getattr(a, "id", None):
        return a.id
    if STATE.exists():
        return json.loads(STATE.read_text()).get("instance_id")
    return None


def cmd_status(a):
    iid = _iid(a)
    if not iid: sys.exit("no instance id")
    code, s = call("GET", f"/instances/{iid}/")
    inst = (s or {}).get("instances") or {}
    print(json.dumps({k: inst.get(k) for k in
                      ("id", "actual_status", "cur_state", "gpu_name", "dph_total",
                       "ssh_host", "ssh_port", "disk_space", "duration")}, indent=2))


def cmd_list(a):
    code, s = call("GET", "/instances/")
    for i in (s or {}).get("instances", []):
        print(f"  id={i.get('id')} status={i.get('actual_status')} gpu={i.get('gpu_name')} "
              f"${i.get('dph_total')}/h label={i.get('label')}")


def cmd_destroy(a):
    iid = _iid(a)
    if not iid: sys.exit("no instance id")
    code, d = call("DELETE", f"/instances/{iid}/")
    print(f"destroy [{code}]: {json.dumps(d)[:300]}")
    if code == 200: save(destroyed=time.strftime("%F %T"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create"); c.set_defaults(f=cmd_create)
    c.add_argument("--offer", required=True)
    c.add_argument("--image", default="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
    c.add_argument("--disk", type=int, default=250)
    c.add_argument("--label", default="quantbias")
    c.add_argument("--pubkey", default="~/.ssh/quantbias_ed25519.pub")
    c.add_argument("--wait", type=int, default=600)
    s = sub.add_parser("status"); s.set_defaults(f=cmd_status); s.add_argument("--id")
    l = sub.add_parser("list"); l.set_defaults(f=cmd_list)
    d = sub.add_parser("destroy"); d.set_defaults(f=cmd_destroy); d.add_argument("--id")
    a = p.parse_args(); sys.exit(a.f(a) or 0)
