#!/usr/bin/env python3
"""Provision a GPU host on Akash via the Console API.

    python3 akash_deploy.py create   --gpu a100 --ram 80Gi --hours 30
    python3 akash_deploy.py status   --dseq 1234567
    python3 akash_deploy.py close    --dseq 1234567
    python3 akash_deploy.py list

Flow: POST /v1/deployments -> poll GET /v1/bids -> accept the CHEAPEST bid via
POST /v1/leases -> read the forwarded SSH port from GET /v1/deployments/{dseq}.

The deployment is billed while it lives, so ``close`` is not optional. Every
create writes the dseq to state_akash.json so nothing is orphaned.
"""
from __future__ import annotations
import argparse, json, os, pathlib, sys, time, urllib.error, urllib.request

API = "https://console-api.akash.network"
# console-api sits behind Cloudflare, which rejects default urllib/python
# signatures with "error code: 1010". A normal browser UA is required.
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
STATE = pathlib.Path(__file__).resolve().parent / "state_akash.json"


def key() -> str:
    k = os.environ.get("Bharat_AKASH_API_KEY") or os.environ.get("AKASH_API_KEY")
    if not k:
        sys.exit("Bharat_AKASH_API_KEY not in environment")
    return k


def call(method: str, path: str, body: dict | None = None, timeout: int = 120):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{API}{path}", data=data, method=method,
                                 headers={"x-api-key": key(), "Content-Type": "application/json",
                                          "User-Agent": UA, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
            return r.status, (json.loads(raw) if raw.strip() else {})
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, {"raw": raw[:800]}


def sdl(pubkey: str, gpu: str, ram: str, cpu: int, mem: str, disk: str, price: int,
        image: str = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
        deploy_key: str = "", hf_token: str = "", autorun: str = "") -> str:
    """Build the SDL. The container is self-healing.

    Akash containers are ephemeral: a restart wipes the filesystem entirely.
    Rather than depend on an operator re-bootstrapping by hand, the boot script
    re-clones the repository, reinstalls, and re-enters the run loop on every
    start. Stage state lives in the git remote, so a restart resumes at the next
    unfinished stage instead of repeating completed work.

    The script must never exit -- a foreground process that dies turns into a
    crash loop that is indistinguishable from a slow image pull.
    """
    import yaml
    lines = [
        "set -x",
        "export DEBIAN_FRONTEND=noninteractive",
        "mkdir -p /run/sshd /root/.ssh /workspace",
        "echo \"$PUBKEY\" >> /root/.ssh/authorized_keys",
        "chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys",
        "apt-get update -qq || true",
        "apt-get install -yqq openssh-server git curl ca-certificates tmux || true",
        "printf 'PermitRootLogin prohibit-password\\nPasswordAuthentication no\\nClientAliveInterval 60\\n' >> /etc/ssh/sshd_config",
        "/usr/sbin/sshd || true",
        # repo-scoped deploy key, delivered via env
        "if [ -n \"$DEPLOY_KEY\" ]; then printf '%s\\n' \"$DEPLOY_KEY\" > /root/.ssh/quantbias_deploy; chmod 600 /root/.ssh/quantbias_deploy; fi",
        "ssh-keyscan -t ed25519 github.com >> /root/.ssh/known_hosts 2>/dev/null",
        "export GIT_SSH_COMMAND='ssh -i /root/.ssh/quantbias_deploy -o StrictHostKeyChecking=no -o UserKnownHostsFile=/root/.ssh/known_hosts'",
        "touch /workspace/BOOT_DONE",
        # self-healing supervisor: re-bootstrap and resume on every container start
        "if [ -n \"$AUTORUN\" ]; then",
        "  ( cd /workspace",
        "    git clone -q git@github.com:DevDaring/Quant_Bias.git Quant_Bias 2>/dev/null || (cd Quant_Bias && git pull -q --rebase origin main)",
        "    export HF_TOKEN=\"$HF_TOKEN\" HF_HOME=/workspace/hf_cache WORK=/workspace ATTN=flash_attention_2 TOKENIZERS_PARALLELISM=false",
        "    bash /workspace/Quant_Bias/quant-bias/scripts/vm_bootstrap.sh >> /workspace/bootstrap.log 2>&1",
        "    bash /workspace/Quant_Bias/quant-bias/scripts/vm_run.sh \"$AUTORUN\" >> /workspace/run.log 2>&1",
        "  ) &",
        "fi",
        "sleep infinity",
    ]
    env = [f"PUBKEY={pubkey}"]
    if deploy_key:
        env.append(f"DEPLOY_KEY={deploy_key}")
    if hf_token:
        env.append(f"HF_TOKEN={hf_token}")
    if autorun:
        env.append(f"AUTORUN={autorun}")
    doc = {
        "version": "2.0",
        "services": {
            "gpu": {
                "image": image,
                "expose": [{"port": 22, "as": 22, "to": [{"global": True}]}],
                "env": env,
                "command": ["bash"],
                "args": ["-c", "\n".join(lines)],
            }
        },
        "profiles": {
            "compute": {
                "gpu": {
                    "resources": {
                        "cpu": {"units": cpu},
                        "memory": {"size": mem},
                        "storage": [{"size": disk}],
                        "gpu": {"units": 1,
                                "attributes": {"vendor": {"nvidia": [{"model": gpu, "ram": ram}]}}},
                    }
                }
            },
            "placement": {"dc": {"pricing": {"gpu": {"denom": "uakt", "amount": price}}}},
        },
        "deployment": {"gpu": {"dc": {"profile": "gpu", "count": 1}}},
    }
    return yaml.safe_dump(doc, sort_keys=False, default_flow_style=False, width=10**6)


def save(**kw):
    s = json.loads(STATE.read_text()) if STATE.exists() else {}
    s.update(kw); s["updated"] = time.strftime("%F %T")
    STATE.write_text(json.dumps(s, indent=2)); return s


def lease_health(dseq: str):
    """(lease_state, ready_replicas, endpoint) for the first lease."""
    code, d = call("GET", f"/v1/deployments/{dseq}")
    if code != 200:
        return None, None, None
    for L in ((d.get("data") or {}).get("leases") or []):
        st = L.get("status") or {}
        svc = (st.get("services") or {}).get("gpu", {})
        ep = None
        for _n, ports in (st.get("forwarded_ports") or {}).items():
            for pt in ports:
                if int(pt.get("port", 0)) == 22:
                    ep = {"host": pt.get("host"), "port": pt.get("externalPort"),
                          "provider": (L.get("id") or {}).get("provider")}
        return L.get("state"), svc.get("ready_replicas"), ep
    return None, None, None


def ssh_endpoint(dseq: str):
    code, d = call("GET", f"/v1/deployments/{dseq}")
    if code != 200:
        return None, (code, d)
    leases = (d.get("data") or {}).get("leases") or []
    for L in leases:
        st = L.get("status") or {}
        for name, ports in (st.get("forwarded_ports") or {}).items():
            for p in ports:
                if int(p.get("port", 0)) == 22:
                    return {"host": p.get("host"), "port": p.get("externalPort"),
                            "provider": (L.get("id") or {}).get("provider"),
                            "state": L.get("state")}, d
        uris = [u for s_ in (st.get("services") or {}).values() for u in (s_.get("uris") or [])]
        if uris:
            return {"host": uris[0], "port": 22, "provider": (L.get("id") or {}).get("provider"),
                    "state": L.get("state")}, d
    return None, d


def cmd_create(a):
    pub = pathlib.Path(a.pubkey).expanduser().read_text().strip()
    dk = pathlib.Path(a.deploy_key).expanduser().read_text().strip() if a.deploy_key else ""
    manifest_sdl = sdl(pub, a.gpu, a.ram, a.cpu, a.mem, a.disk, a.price, a.image,
                       dk, os.environ.get("HUGGINGFACE_TOKEN", ""), a.autorun)
    print(f"creating deployment: gpu={a.gpu} ram={a.ram} disk={a.disk} limit={a.hours}h")
    code, d = call("POST", "/v1/deployments",
                   {"data": {"sdl": manifest_sdl, "runtimeLimitHours": a.hours}})
    if code not in (200, 201):
        print(f"  create failed [{code}]: {json.dumps(d)[:700]}")
        return 2
    data = d.get("data") or {}
    dseq = str(data.get("dseq") or "")
    manifest = data.get("manifest")
    if not dseq:
        print(f"  no dseq in response: {json.dumps(d)[:500]}"); return 2
    save(dseq=dseq, created=time.strftime("%F %T"), gpu=a.gpu, hours=a.hours)
    print(f"  dseq={dseq}  (recorded in {STATE.name})")

    print("polling for bids ...")
    bids = []
    for i in range(a.bid_wait // 5):
        code, b = call("GET", f"/v1/bids?dseq={dseq}")
        bids = [x for x in (b.get("data") or [])
                if (x.get("bid") or {}).get("state") == "open"]
        if bids:
            break
        time.sleep(5)
        if i % 6 == 5:
            print(f"  {5*(i+1)}s ...")
    if not bids:
        print("  no bids; closing so nothing is billed")
        call("DELETE", f"/v1/deployments/{dseq}")
        return 3

    def amt(x):
        try: return float(((x.get("bid") or {}).get("price") or {}).get("amount", 1e18))
        except Exception: return 1e18
    bids.sort(key=amt)
    excl = set(a.exclude or [])
    bids = [b for b in bids if b["bid"]["id"]["provider"] not in excl] or bids
    print(f"  {len(bids)} usable bid(s):")
    for x in bids[:6]:
        print(f"    {amt(x):>10.0f} uakt/block  {x['bid']['id']['provider']}")

    # A bid is only useful if the provider actually brings the pod up. Some
    # providers accept and then close the lease, so each candidate is verified
    # and the next one tried on failure.
    for rank, cand in enumerate(bids[:a.max_tries]):
        win = cand["bid"]["id"]
        print(f"\n  [{rank+1}/{min(len(bids), a.max_tries)}] leasing from {win['provider']} "
              f"at {amt(cand):.0f} uakt/block")
        code, L = call("POST", "/v1/leases",
                       {"manifest": manifest,
                        "leases": [{"dseq": dseq, "gseq": win["gseq"], "oseq": win["oseq"],
                                    "provider": win["provider"]}]})
        if code not in (200, 201):
            print(f"    lease rejected [{code}]: {json.dumps(L)[:300]}")
            continue
        save(provider=win["provider"], gseq=win["gseq"], oseq=win["oseq"],
             price_uakt_block=amt(cand))
        ok = False
        for i in range(a.ready_wait // 15):
            time.sleep(15)
            state, ready, ep = lease_health(dseq)
            if state == "closed":
                print(f"    provider closed the lease after {15*(i+1)}s -- trying the next bid")
                break
            if ready and int(ready) >= 1 and ep and ep.get("port"):
                save(**{f"ssh_{k}": v for k, v in ep.items()}, lease_state=state)
                print(f"\nREADY  ssh -p {ep['port']} root@{ep['host']}")
                print(json.dumps(ep, indent=2))
                ok = True
                break
            if i % 4 == 3:
                print(f"    {15*(i+1)}s state={state} ready_replicas={ready}")
        if ok:
            return 0
    print("\n  no provider brought the workload up; closing so nothing is billed")
    call("DELETE", f"/v1/deployments/{dseq}")
    return 5


def cmd_status(a):
    dseq = a.dseq or (json.loads(STATE.read_text()).get("dseq") if STATE.exists() else None)
    if not dseq: sys.exit("no dseq")
    ep, d = ssh_endpoint(dseq)
    print(json.dumps({"dseq": dseq, "ssh": ep,
                      "deployment": ((d.get("data") or {}).get("deployment") if isinstance(d, dict) else d)},
                     indent=2)[:2000])


def cmd_close(a):
    dseq = a.dseq or (json.loads(STATE.read_text()).get("dseq") if STATE.exists() else None)
    if not dseq: sys.exit("no dseq")
    code, d = call("DELETE", f"/v1/deployments/{dseq}")
    print(f"close [{code}]: {json.dumps(d)[:300]}")
    if code == 200: save(closed=time.strftime("%F %T"))


def cmd_list(a):
    code, d = call("GET", "/v1/deployments")
    rows = (d.get("data") or {}).get("deployments") if isinstance(d.get("data"), dict) else d.get("data")
    print(json.dumps(rows, indent=2)[:3000])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create"); c.set_defaults(f=cmd_create)
    c.add_argument("--gpu", default="a100"); c.add_argument("--ram", default="80Gi")
    c.add_argument("--cpu", type=int, default=16); c.add_argument("--mem", default="64Gi")
    c.add_argument("--disk", default="300Gi"); c.add_argument("--hours", type=int, default=30)
    c.add_argument("--price", type=int, default=100000, help="max uakt per block (ceiling only)")
    c.add_argument("--pubkey", default="~/.ssh/quantbias_ed25519.pub")
    c.add_argument("--image", default="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
    c.add_argument("--deploy-key", default="", help="path to the repo-scoped private key")
    c.add_argument("--autorun", default="", help="smoke|full: self-heal and resume after a restart")
    c.add_argument("--bid-wait", type=int, default=180)
    c.add_argument("--ready-wait", type=int, default=420)
    c.add_argument("--max-tries", type=int, default=4, help="how many bids to try before giving up")
    c.add_argument("--exclude", nargs="*", default=[], help="provider addresses to skip")
    s = sub.add_parser("status"); s.set_defaults(f=cmd_status); s.add_argument("--dseq")
    k = sub.add_parser("close");  k.set_defaults(f=cmd_close);  k.add_argument("--dseq")
    l = sub.add_parser("list");   l.set_defaults(f=cmd_list)
    a = p.parse_args(); sys.exit(a.f(a) or 0)
