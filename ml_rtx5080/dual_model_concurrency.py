"""Mixed-load validation of both selector models on one Triton GPU instance.

Starts from explicit model control, records idle / single-model / dual-model GPU
memory, then sends DINOv2 and DINOv3 traffic concurrently through the research
API. Reads only the first non-test image of the frozen parity protocol.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import tritonclient.grpc as grpcclient

_PKG = Path(__file__).resolve().parent
_REPO = _PKG.parent
for path in (_PKG, _REPO / "backend"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

OUT = _PKG / "experiments" / "dual_model_triton_v1" / "dual_model_concurrency.json"
PROTOCOL = _PKG / "experiments" / "dual_model_triton_v1" / "parity_protocol_v1.json"
SPLIT_MANIFEST = _PKG / "experiments" / "dataset_audit" / "split_manifest_v2.json"
FROZEN_SHA = "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f"
DINOV2 = "dinov2_vitb14_live"
DINOV3 = "dinov3_authenticity_candidate"


def _summary(ms: list[float]) -> dict:
    ordered = sorted(ms)
    pick = lambda q: ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]
    return {
        "n": len(ms),
        "mean_ms": statistics.fmean(ms) if ms else None,
        "p50_ms": pick(0.50) if ms else None,
        "p90_ms": pick(0.90) if ms else None,
        "p99_ms": pick(0.99) if ms else None,
        "max_ms": ordered[-1] if ms else None,
    }


def _gpu_process_memory_mib() -> int | None:
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=process_name,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    values = [int(line.split(",")[1]) for line in out.strip().splitlines() if "tritonserver" in line]
    return sum(values) if values else None


def _gpu_used_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout.strip().split()[0])


def _input_image() -> tuple[Path, str, bytes]:
    case = json.loads(PROTOCOL.read_text())["inputs"][0]
    path = (_REPO / case["sample_id"]).resolve()
    test = {str(Path(p).resolve()) for p in (json.loads(SPLIT_MANIFEST.read_text()).get("membership") or {}).get("test") or []}
    if case["split"] == "test" or str(path) in test:
        raise RuntimeError("refusing to open a final-test image")
    brand = path.parent.name
    return path, brand, path.read_bytes()


def _config(client, name: str) -> dict:
    cfg = client.get_model_config(name, "1", as_json=True)["config"]
    groups = cfg.get("instance_group") or []
    kinds = sorted({g.get("kind") for g in groups})
    gpus = sorted({int(g) for group in groups for g in (group.get("gpus") or [0])})
    cuda: dict[str, str] = {}
    accelerators = (cfg.get("optimization") or {}).get("execution_accelerators") or {}
    for accelerator in accelerators.get("gpu_execution_accelerator") or []:
        if accelerator.get("name") == "cuda":
            cuda.update({str(k): str(v) for k, v in (accelerator.get("parameters") or {}).items()})
    return {
        "name": cfg.get("name"),
        "backend": cfg.get("backend"),
        "kinds": kinds,
        "gpus": gpus,
        "cuda_parameters": cuda,
        "use_tf32": cuda.get("use_tf32"),
    }


def _count(client, name: str) -> int:
    stats = client.get_inference_statistics(name, "1", as_json=True)["model_stats"]
    return int(stats[0].get("inference_count", 0)) if stats else 0


def _compute_ns(client, name: str) -> tuple[int, int]:
    stats = client.get_inference_statistics(name, "1", as_json=True)["model_stats"][0]
    return int(stats.get("inference_count", 0)), int(stats["inference_stats"]["compute_infer"].get("ns", 0))


def wait_ready(client, name: str, timeout: float = 300.0) -> float:
    started = time.perf_counter()
    while time.perf_counter() - started < timeout:
        try:
            if client.is_model_ready(name, "1"):
                return time.perf_counter() - started
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError(f"{name} did not become ready")


def wait_unready(client, name: str, timeout: float = 60.0) -> bool:
    started = time.perf_counter()
    while time.perf_counter() - started < timeout:
        try:
            if not client.is_model_ready(name, "1"):
                return True
        except Exception:
            return True
        time.sleep(0.2)
    return False


def post_research(api: str, token: str, image_path: Path, image: bytes, brand: str, logical: str, timeout: float = 120.0):
    return httpx.post(
        f"{api}/research/verify",
        headers={"Authorization": f"Bearer {token}"},
        files={"image": (image_path.name, image, "image/jpeg")},
        data={"brand": brand, "logical_model": logical},
        timeout=timeout,
    )


def register_token(api: str, email: str) -> str:
    import secrets

    password = "E2e" + secrets.token_hex(8) + "9"
    with httpx.Client(base_url=api, timeout=30) as client:
        created = client.post("/auth/register", json={"email": email, "password": password, "role": "buyer"})
        if created.status_code not in (201, 409):
            raise RuntimeError(f"register {email} returned {created.status_code}: {created.text}")
        if created.status_code == 409:
            login = client.post("/auth/login", json={"email": email, "password": password})
            if login.status_code != 200:
                raise RuntimeError(f"{email} already exists with an unknown password")
        else:
            login = client.post("/auth/login", json={"email": email, "password": password})
        login.raise_for_status()
        return login.json()["access_token"]


def measure_api_latency(api: str, token: str, image_path: Path, image: bytes, brand: str, logical: str, n: int) -> dict:
    samples: list[float] = []
    statuses: set[int] = set()
    bodies: list[dict] = []
    for _ in range(n):
        started = time.perf_counter()
        response = post_research(api, token, image_path, image, brand, logical)
        samples.append((time.perf_counter() - started) * 1000)
        statuses.add(response.status_code)
        bodies.append(response.json())
    return {
        "http_statuses": sorted(statuses),
        "latency": _summary(samples),
        "decisions": sorted({b.get("decision") for b in bodies}),
        "models": sorted({b.get("model") for b in bodies}),
        "checkpoint_shas": sorted({b.get("checkpoint_sha") for b in bodies}),
        "first_body": bodies[0] if bodies else None,
        "last_body": bodies[-1] if bodies else None,
        "decision_consistent": len({(b.get("decision"), b.get("model"), b.get("checkpoint_sha")) for b in bodies}) == 1,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="localhost:18001")
    parser.add_argument("--triton-http", default="http://localhost:18000")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--research-email", default="e2e-research@example.com")
    parser.add_argument("--per-model", type=int, default=200)
    parser.add_argument("--latency-samples", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    client = grpcclient.InferenceServerClient(url=args.url)
    image_path, brand, image = _input_image()
    results: dict = {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "input": str(image_path.relative_to(_REPO)),
        "brand": brand,
        "split": "non-test parity protocol image 0",
        "triton": client.get_server_metadata(as_json=True),
        "gpu": subprocess.run(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
    }

    for name in (DINOV2, DINOV3):
        try:
            client.unload_model(name)
        except Exception:
            pass
    wait_unready(client, DINOV2)
    wait_unready(client, DINOV3)
    time.sleep(2)
    results["memory"] = {
        "idle_no_models_tritonserver_mib": _gpu_process_memory_mib(),
        "idle_gpu_used_mib": _gpu_used_mib(),
    }

    load: dict = {}
    for name in (DINOV2, DINOV3):
        started = time.perf_counter()
        client.load_model(name)
        load_s = wait_ready(client, name)
        load[name] = {
            "load_seconds": time.perf_counter() - started,
            "ready_wait_seconds": load_s,
            "config": _config(client, name),
            "memory_tritonserver_mib": _gpu_process_memory_mib(),
            "gpu_used_mib": _gpu_used_mib(),
            "ready": client.is_model_ready(name, "1"),
        }
        if name == DINOV2:
            results["memory"]["after_dinov2_only_tritonserver_mib"] = _gpu_process_memory_mib()
    results["load"] = load
    results["both_ready"] = client.is_model_ready(DINOV2, "1") and client.is_model_ready(DINOV3, "1")
    results["memory"]["dual_loaded_idle_tritonserver_mib"] = _gpu_process_memory_mib()
    results["memory"]["dual_loaded_idle_gpu_used_mib"] = _gpu_used_mib()

    token = register_token(args.api, args.research_email)
    models = httpx.get(
        f"{args.api}/research/models",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    results["research_models"] = models.json() if models.status_code == 200 else {"http_status": models.status_code, "body": models.text}

    # Sequential API latency with both models loaded.
    api_latency = {}
    triton_compute = {}
    for logical, triton_name in (("dinov2_legacy", DINOV2), ("dinov3_experimental", DINOV3)):
        c0, ns0 = _compute_ns(client, triton_name)
        api_latency[logical] = measure_api_latency(args.api, token, image_path, image, brand, logical, args.latency_samples)
        c1, ns1 = _compute_ns(client, triton_name)
        n = max(1, c1 - c0)
        triton_compute[logical] = {
            "inference_count_delta": c1 - c0,
            "mean_compute_infer_ms": (ns1 - ns0) / n / 1e6,
        }
    results["api_latency_both_loaded_sequential"] = api_latency
    results["triton_compute_during_api_latency"] = triton_compute

    # Mixed concurrent API load.
    mixed_counts = {DINOV2: _count(client, DINOV2), DINOV3: _count(client, DINOV3)}
    compute0 = {DINOV2: _compute_ns(client, DINOV2), DINOV3: _compute_ns(client, DINOV3)}
    mixed: dict[str, dict] = {
        "dinov2_legacy": {"ok": 0, "fail": 0, "ms": [], "decisions": set(), "errors": []},
        "dinov3_experimental": {"ok": 0, "fail": 0, "ms": [], "decisions": set(), "errors": []},
    }
    lock = threading.Lock()
    peak_mem = results["memory"]["dual_loaded_idle_tritonserver_mib"] or 0

    def one(logical: str) -> None:
        nonlocal peak_mem
        started = time.perf_counter()
        try:
            response = post_research(args.api, token, image_path, image, brand, logical)
            body = response.json()
            elapsed = (time.perf_counter() - started) * 1000
            mem = _gpu_process_memory_mib() or 0
            with lock:
                peak_mem = max(peak_mem, mem)
                row = mixed[logical]
                row["ms"].append(elapsed)
                if response.status_code == 200 and body.get("decision") in {"AUTHENTIC", "FAKE", "REVIEW"}:
                    row["ok"] += 1
                    row["decisions"].add(body.get("decision"))
                    if logical == "dinov3_experimental" and body.get("checkpoint_sha") != FROZEN_SHA:
                        row["fail"] += 1
                        row["ok"] -= 1
                        row["errors"].append("checkpoint_sha_mismatch")
                    if logical == "dinov2_legacy" and body.get("model") != "LEGACY_DINOV2":
                        row["fail"] += 1
                        row["ok"] -= 1
                        row["errors"].append(f"unexpected_model:{body.get('model')}")
                    if logical == "dinov3_experimental" and body.get("model") != "DINOV3_RESEARCH_PROTOTYPE":
                        row["fail"] += 1
                        row["ok"] -= 1
                        row["errors"].append(f"unexpected_model:{body.get('model')}")
                else:
                    row["fail"] += 1
                    row["errors"].append(f"{response.status_code}:{body.get('status')}")
        except Exception as exc:
            with lock:
                mixed[logical]["fail"] += 1
                mixed[logical]["errors"].append(type(exc).__name__)

    jobs = (["dinov2_legacy"] * args.per_model) + (["dinov3_experimental"] * args.per_model)
    wall = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, logical) for logical in jobs]
        for future in as_completed(futures):
            future.result()
    wall = time.perf_counter() - wall
    mixed_after = {DINOV2: _count(client, DINOV2), DINOV3: _count(client, DINOV3)}
    compute1 = {DINOV2: _compute_ns(client, DINOV2), DINOV3: _compute_ns(client, DINOV3)}
    results["mixed_concurrent_api"] = {
        "workers": args.workers,
        "requested_per_model": args.per_model,
        "wall_seconds": wall,
        "combined_requests_per_second": (args.per_model * 2) / wall if wall else None,
        "peak_tritonserver_mib": peak_mem,
        "models": {
            logical: {
                "completed_ok": mixed[logical]["ok"],
                "failed": mixed[logical]["fail"],
                "latency": _summary(mixed[logical]["ms"]),
                "decisions": sorted(mixed[logical]["decisions"]),
                "error_samples": mixed[logical]["errors"][:10],
                "throughput_rps": mixed[logical]["ok"] / wall if wall else None,
            }
            for logical in mixed
        },
        "inference_count_before": mixed_counts,
        "inference_count_after": mixed_after,
        "inference_count_delta": {k: mixed_after[k] - mixed_counts[k] for k in mixed_counts},
        "triton_compute_mean_ms": {
            name: ((compute1[name][1] - compute0[name][1]) / max(1, compute1[name][0] - compute0[name][0]) / 1e6)
            for name in (DINOV2, DINOV3)
        },
    }
    results["memory"]["peak_during_mixed_tritonserver_mib"] = peak_mem
    results["memory"]["after_mixed_gpu_used_mib"] = _gpu_used_mib()

    # Reload while the other model stays loaded.
    reload = {}
    for name in (DINOV2, DINOV3):
        other = DINOV3 if name == DINOV2 else DINOV2
        client.unload_model(name)
        unloaded = wait_unready(client, name)
        other_still = client.is_model_ready(other, "1")
        started = time.perf_counter()
        client.load_model(name)
        wait_ready(client, name)
        reload[name] = {
            "ready_after_unload": not unloaded,
            "other_stayed_ready": other_still,
            "reload_seconds": time.perf_counter() - started,
            "both_ready_after": client.is_model_ready(DINOV2, "1") and client.is_model_ready(DINOV3, "1"),
        }
    results["reload_while_peer_loaded"] = reload
    results["both_ready_finally"] = client.is_model_ready(DINOV2, "1") and client.is_model_ready(DINOV3, "1")
    results["final_configs"] = {name: _config(client, name) for name in (DINOV2, DINOV3)}

    dinov2_delta = results["mixed_concurrent_api"]["inference_count_delta"][DINOV2]
    dinov3_delta = results["mixed_concurrent_api"]["inference_count_delta"][DINOV3]
    results["pass"] = bool(
        results["both_ready"]
        and results["both_ready_finally"]
        and load[DINOV2]["config"]["kinds"] == ["KIND_GPU"]
        and load[DINOV3]["config"]["kinds"] == ["KIND_GPU"]
        and load[DINOV2]["config"]["use_tf32"] == "0"
        and load[DINOV3]["config"]["use_tf32"] == "0"
        and mixed["dinov2_legacy"]["ok"] == args.per_model
        and mixed["dinov3_experimental"]["ok"] == args.per_model
        and mixed["dinov2_legacy"]["fail"] == 0
        and mixed["dinov3_experimental"]["fail"] == 0
        and dinov2_delta == args.per_model
        and dinov3_delta == args.per_model
        and api_latency["dinov3_experimental"]["decision_consistent"]
        and api_latency["dinov2_legacy"]["decision_consistent"]
        and FROZEN_SHA in (api_latency["dinov3_experimental"]["checkpoint_shas"] or [])
    )
    args.out.write_text(json.dumps(results, indent=2, default=list) + "\n")
    print(json.dumps({"pass": results["pass"], "out": str(args.out), "memory": results["memory"], "mixed": results["mixed_concurrent_api"]["models"]}, indent=2, default=list))
    return 0 if results["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
