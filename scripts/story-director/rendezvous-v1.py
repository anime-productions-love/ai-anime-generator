#!/usr/bin/env python3
"""Private GitHub Actions artifact rendezvous for four CPU RPC runners (v1).

Reusable mechanics from the Contract-Automation eight-rank workflow:
ephemeral Tailscale identity, per-peer immutable artifacts, controller waits
for three exact peers, and bounded polling. Never exposes RPC to public net.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time


def gh(*cmd):
    return subprocess.check_output(["gh", *cmd], text=True).strip()


def valid_id(value):
    if not re.fullmatch(r"[0-9]{4,24}", value or ""):
        raise ValueError("Invalid GitHub run ID")
    return value


def valid_ip(value):
    if not re.fullmatch(r"100\.(?:[0-9]{1,3}\.){2}[0-9]{1,3}", value):
        raise ValueError("Expected ephemeral private Tailscale IPv4")
    if any(int(octet) > 255 for octet in value.split(".")):
        raise ValueError("Invalid Tailscale IPv4 octet")
    return value


def announce(a):
    valid_id(a.run_id)
    if a.rank not in (1, 2, 3):
        raise ValueError("Only remote RPC ranks 1, 2, 3 can announce")
    addr = valid_ip(a.address)
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "version": "v1", "run_id": a.run_id,
        "rank": a.rank, "address": addr,
        "model": a.model, "rpc_port": 50052,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared ephemeral RPC peer advertisement: rank {a.rank}", flush=True)


def collect(a):
    valid_id(a.run_id)
    if os.environ.get("GITHUB_REPOSITORY") != "anime-productions-love/ai-anime-generator":
        raise ValueError("Wrong repository for rendezvous")
    root = Path(a.destination)
    root.mkdir(parents=True, exist_ok=True)
    results = {}
    for attempt in range(60):
        result = json.loads(gh(
            "api", f"repos/{os.environ['GITHUB_REPOSITORY']}/actions/"
                   f"runs/{a.run_id}/artifacts?per_page=100"
        ))
        found = {item["name"] for item in result.get("artifacts", [])
                 if not item.get("expired")}
        for rank in (1, 2, 3):
            name = f"story-director-rpc-v1-{a.run_id}-rank-{rank}"
            if rank in results or name not in found:
                continue
            folder = root / str(rank)
            folder.mkdir(exist_ok=True)
            subprocess.run([
                "gh", "run", "download", a.run_id,
                "--repo", os.environ["GITHUB_REPOSITORY"],
                "--name", name, "--dir", str(folder),
            ], check=True)
            paths = list(folder.rglob("rendezvous.json"))
            if len(paths) != 1:
                raise ValueError(f"Peer {rank} has ambiguous rendezvous receipt")
            info = json.loads(paths[0].read_text())
            if (info.get("version") != "v1" or info.get("rank") != rank
                or info.get("run_id") != a.run_id
                or info.get("model") != a.model
                or info.get("rpc_port") != 50052):
                raise ValueError(f"Peer {rank} identity mismatch")
            results[rank] = valid_ip(info["address"]) + ":50052"
        if len(results) == 3:
            break
        print(f"Waiting for exact three RPC peers ({len(results)}/3), "
              f"attempt {attempt+1}/60", flush=True)
        time.sleep(15)
    if len(results) != 3:
        raise RuntimeError("Four-runner RPC peer rendezvous timed out")
    addresses = [results[i] for i in (1, 2, 3)]
    Path(a.output).write_text(",".join(addresses) + "\n")
    print("All three exact private RPC peers acquired", flush=True)


def monitor(a):
    valid_id(a.run_id)
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    for attempt in range(1320):
        result = json.loads(gh(
            "api", f"repos/{repo}/actions/runs/{a.run_id}/jobs?per_page=100"
        ))
        coordinators = [j for j in result.get("jobs", [])
                        if j.get("name") == "Story Director node 0"]
        if len(coordinators) == 1 and coordinators[0].get("status") == "completed":
            print("Coordinator completed: " +
                  str(coordinators[0].get("conclusion")), flush=True)
            if coordinators[0].get("conclusion") != "success":
                raise RuntimeError("Coordinator failed: RPC shard stopped")
            return
        time.sleep(15)
    raise RuntimeError("Coordinator did not finish within bounded shard lifetime")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sp = p.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("announce")
    q.add_argument("--rank", required=True, type=int)
    q.add_argument("--address", required=True)
    q.add_argument("--model", required=True)
    q.add_argument("--run-id", required=True)
    q.add_argument("--output", required=True)
    q = sp.add_parser("collect")
    q.add_argument("--run-id", required=True)
    q.add_argument("--model", required=True)
    q.add_argument("--destination", required=True)
    q.add_argument("--output", required=True)
    q = sp.add_parser("monitor")
    q.add_argument("--run-id", required=True)
    a = p.parse_args()
    {"announce": announce, "collect": collect, "monitor": monitor}[a.cmd](a)


if __name__ == "__main__":
    main()
