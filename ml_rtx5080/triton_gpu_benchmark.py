"""Per-model GPU latency, throughput, memory, and stability for the two selector models.

Runs against a Triton started with explicit model control. Each model is measured alone:
the other is unloaded first. The input is the first non-test image of the frozen parity
protocol, preprocessed with that model's authoritative function. Run only after GPU parity passes.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

_PKG = Path(__file__).resolve().parent
_REPO = _PKG.parent
for path in (_PKG, _REPO / "backend"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

OUT = _PKG / "experiments" / "dual_model_triton_v1" / "gpu_performance_results.json"
PROTOCOL = _PKG / "experiments" / "dual_model_triton_v1" / "parity_protocol_v1.json"
SPLIT_MANIFEST = _PKG / "experiments" / "dataset_audit" / "split_manifest_v2.json"
MODELS = ("dinov2_vitb14_live", "dinov3_authenticity_candidate")


def _input_for(model: str) -> np.ndarray:
    from PIL import Image

    case = json.loads(PROTOCOL.read_text())["inputs"][0]
    path = (_REPO / case["sample_id"]).resolve()
    test = {str(Path(p).resolve()) for p in (json.loads(SPLIT_MANIFEST.read_text()).get("membership") or {}).get("test") or []}
    if case["split"] == "test" or str(path) in test:
        raise RuntimeError("refusing to open a final-test image")
    with Image.open(path) as handle:
        rgb = handle.convert("RGB")
        if model == "dinov2_vitb14_live":
            from inference.triton_client import preprocess_chw

            return preprocess_chw(np.array(rgb), side=504)
        from dinov3_serving import preprocess_serving_image

        return preprocess_serving_image(rgb).unsqueeze(0).numpy().astype(np.float32)


def _gpu_process_memory_mib() -> int | None:
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=process_name,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    values = [int(line.split(",")[1]) for line in out.strip().splitlines() if "tritonserver" in line]
    return sum(values) if values else None


def _summary(ms: list[float]) -> dict:
    ordered = sorted(ms)
    pick = lambda q: ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]
    return {
        "n": len(ms),
        "mean_ms": statistics.fmean(ms),
        "p50_ms": pick(0.50),
        "p90_ms": pick(0.90),
        "p99_ms": pick(0.99),
        "max_ms": ordered[-1],
    }


class Bench:
    def __init__(self, url: str) -> None:
        import tritonclient.grpc as grpcclient

        self.grpc = grpcclient
        self.url = url
        self.client = grpcclient.InferenceServerClient(url=url)

    def infer(self, model: str, batch: np.ndarray, client=None) -> np.ndarray:
        client = client or self.client
        inputs = [self.grpc.InferInput("input__0", list(batch.shape), "FP32")]
        inputs[0].set_data_from_numpy(batch)
        return client.infer(model, inputs, model_version="1", client_timeout=60.0).as_numpy("output__0").reshape(-1)

    def compute_ns(self, model: str) -> tuple[int, int]:
        stats = self.client.get_inference_statistics(model, "1", as_json=True)["model_stats"][0]
        infer = stats["inference_stats"]
        return int(stats.get("execution_count", 0)), int(infer["compute_infer"].get("ns", 0))

    def wait_ready(self, model: str, timeout: float = 300.0) -> float:
        started = time.perf_counter()
        while time.perf_counter() - started < timeout:
            try:
                if self.client.is_model_ready(model, "1"):
                    return time.perf_counter() - started
            except Exception:
                pass
            time.sleep(0.2)
        raise RuntimeError(f"{model} did not become ready")


def measure(bench: Bench, model: str) -> dict:
    for other in MODELS:
        try:
            bench.client.unload_model(other)
        except Exception:
            pass
    time.sleep(3)
    memory_no_model = _gpu_process_memory_mib()

    load_started = time.perf_counter()
    bench.client.load_model(model)
    bench.wait_ready(model)
    load_seconds = time.perf_counter() - load_started
    kinds = sorted({g["kind"] for g in bench.client.get_model_config(model, "1", as_json=True)["config"]["instance_group"]})
    if kinds != ["KIND_GPU"]:
        raise RuntimeError(f"{model} is not on a GPU instance: {kinds}")

    x1 = _input_for(model)
    for _ in range(10):
        bench.infer(model, x1)
    memory_warm = _gpu_process_memory_mib()
    reference = bench.infer(model, x1)[0]

    count0, ns0 = bench.compute_ns(model)
    latencies = []
    for _ in range(100):
        started = time.perf_counter()
        bench.infer(model, x1)
        latencies.append((time.perf_counter() - started) * 1000)
    count1, ns1 = bench.compute_ns(model)
    warm = {
        "client_round_trip": _summary(latencies),
        "triton_compute_infer_mean_ms": (ns1 - ns0) / max(1, count1 - count0) / 1e6,
    }

    throughput = {}
    for size in (1, 2, 4, 8):
        batch = np.ascontiguousarray(np.repeat(x1, size, axis=0))
        bench.infer(model, batch)
        times = []
        for _ in range(20):
            started = time.perf_counter()
            out = bench.infer(model, batch)
            times.append(time.perf_counter() - started)
        throughput[str(size)] = {
            "requests": 20,
            "mean_request_ms": statistics.fmean(times) * 1000,
            "images_per_second": size * 20 / sum(times),
            "max_abs_deviation_from_batch1_output": float(np.max(np.abs(out - reference))),
        }
    memory_peak = _gpu_process_memory_mib()

    sequential = [bench.infer(model, x1)[0] for _ in range(200)]
    seq_dev = float(np.max(np.abs(np.array(sequential) - reference)))

    errors: list[str] = []
    values: list[float] = []
    conc_ms: list[float] = []
    lock = threading.Lock()

    def worker() -> None:
        client = bench.grpc.InferenceServerClient(url=bench.url)
        for _ in range(25):
            started = time.perf_counter()
            try:
                value = float(bench.infer(model, x1, client)[0])
            except Exception as exc:
                with lock:
                    errors.append(type(exc).__name__)
                continue
            with lock:
                values.append(value)
                conc_ms.append((time.perf_counter() - started) * 1000)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    wall = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - wall

    bench.client.unload_model(model)
    unloaded_ready = None
    for _ in range(50):
        try:
            unloaded_ready = bench.client.is_model_ready(model, "1")
        except Exception:
            unloaded_ready = False
        if not unloaded_ready:
            break
        time.sleep(0.2)
    reload_started = time.perf_counter()
    bench.client.load_model(model)
    bench.wait_ready(model)
    reload_seconds = time.perf_counter() - reload_started
    after_reload = float(bench.infer(model, x1)[0])

    return {
        "instance_kind": kinds,
        "input": "first parity-protocol image (non-test), batch tiled for larger sizes",
        "load_seconds": load_seconds,
        "gpu_memory_mib_tritonserver": {"no_model_loaded": memory_no_model, "after_load_and_warmup": memory_warm, "peak_after_batch8": memory_peak},
        "warm_batch1": warm,
        "throughput_by_batch_size": throughput,
        "sequential_stability": {"requests": 200, "errors": 0, "max_abs_deviation": seq_dev},
        "concurrent_stability": {
            "threads": 8,
            "requests_per_thread": 25,
            "completed": len(values),
            "errors": len(errors),
            "error_types": sorted(set(errors)),
            "wall_seconds": wall,
            "requests_per_second": len(values) / wall,
            "latency": _summary(conc_ms) if conc_ms else None,
            "max_abs_deviation": float(np.max(np.abs(np.array(values) - reference))) if values else None,
        },
        "reload": {
            "ready_after_unload": unloaded_ready,
            "reload_seconds": reload_seconds,
            "abs_deviation_after_reload": abs(after_reload - float(reference)),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="localhost:18001")
    args = parser.parse_args()
    bench = Bench(args.url)
    results = {
        "triton": bench.client.get_server_metadata(as_json=True).get("version"),
        "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
        "method": "single client on the host over gRPC (localhost); client_round_trip includes serialization and transport; triton_compute_infer is Triton's own compute time",
        "models": {},
    }
    for model in MODELS:
        results["models"][model] = measure(bench, model)
    for model in MODELS:
        try:
            bench.client.load_model(model)
            bench.wait_ready(model)
        except Exception:
            pass
    OUT.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
