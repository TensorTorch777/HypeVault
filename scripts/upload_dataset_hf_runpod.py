#!/usr/bin/env python3
"""Upload local scraper dataset to a private HF dataset repo, then pull on RunPod."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, login

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_DATASET_ROOT = Path("/home/tensortorch26/Desktop/scraper")
DEFAULT_REPO_NAME = "sneakers-watches-dataset"
DEFAULT_RUNPOD_HOST = "root@154.54.102.53"
DEFAULT_RUNPOD_PORT = "18332"
DEFAULT_SSH_KEY = Path.home() / ".ssh" / "id_ed25519"
DEFAULT_POD_DATA_ROOT = "/workspace/data"
EXPECTED_FOLDERS = (
    "Label_0_Sneakers",
    "Label_0_Watches",
    "Label_1_Sneakers",
    "Label_1_Watches",
)
EXPECTED_PER_FOLDER = 15_000
MAX_UPLOAD_ATTEMPTS = 3
MAX_DOWNLOAD_ATTEMPTS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload dataset to Hugging Face and download it on RunPod."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Local dataset root containing Label_* folders.",
    )
    parser.add_argument(
        "--repo-name",
        default=DEFAULT_REPO_NAME,
        help="Dataset repo name on your Hugging Face account.",
    )
    parser.add_argument(
        "--runpod-host",
        default=DEFAULT_RUNPOD_HOST,
        help="RunPod SSH destination, e.g. root@IP.",
    )
    parser.add_argument(
        "--runpod-port",
        default=DEFAULT_RUNPOD_PORT,
        help="RunPod SSH port.",
    )
    parser.add_argument(
        "--ssh-key",
        type=Path,
        default=DEFAULT_SSH_KEY,
        help="SSH private key for RunPod.",
    )
    parser.add_argument(
        "--pod-data-root",
        default=DEFAULT_POD_DATA_ROOT,
        help="Dataset destination on the pod.",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Only download on RunPod from an existing HF dataset repo.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Only upload to Hugging Face.",
    )
    return parser.parse_args()


def require_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print(
            "Set HF_TOKEN to a Hugging Face token with write access before running.",
            file=sys.stderr,
        )
        sys.exit(1)
    return token


def count_images(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            total += 1
    return total


def folder_counts(root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for folder in EXPECTED_FOLDERS:
        folder_path = root / folder
        counts[folder] = count_images(folder_path) if folder_path.is_dir() else 0
    return counts


def folder_size_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


def human_bytes(num_bytes: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def human_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def validate_local_dataset(root: Path) -> tuple[int, int]:
    if not root.is_dir():
        print(f"Dataset root not found: {root}", file=sys.stderr)
        sys.exit(1)

    counts = folder_counts(root)
    missing = [name for name, count in counts.items() if count == 0]
    if missing:
        print(f"Missing or empty folders under {root}: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    total = sum(counts.values())
    size_bytes = folder_size_bytes(root)
    print("Local dataset check")
    for folder, count in counts.items():
        print(f"  {folder}: {count:,} images")
    print(f"  total images: {total:,}")
    print(f"  total size: {human_bytes(size_bytes)}")
    return total, size_bytes


def upload_dataset(api: HfApi, repo_id: str, dataset_root: Path) -> None:
    print(f"Creating or reusing private dataset repo: {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)

    for attempt in range(1, MAX_UPLOAD_ATTEMPTS + 1):
        print(f"Upload attempt {attempt}/{MAX_UPLOAD_ATTEMPTS} via upload_large_folder")
        try:
            api.upload_large_folder(
                repo_id=repo_id,
                folder_path=str(dataset_root),
                repo_type="dataset",
                private=True,
                print_report=True,
            )
            print("Hugging Face upload finished.")
            return
        except Exception as exc:  # noqa: BLE001 - retry wrapper
            print(f"Upload failed on attempt {attempt}: {exc}", file=sys.stderr)
            if attempt == MAX_UPLOAD_ATTEMPTS:
                raise
            time.sleep(30)


def ssh_base_cmd(args: argparse.Namespace) -> list[str]:
    return [
        "ssh",
        "-i",
        str(args.ssh_key),
        "-p",
        args.runpod_port,
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=10",
        args.runpod_host,
    ]


def run_remote_download(args: argparse.Namespace, repo_id: str, token: str) -> None:
    remote_cmd = " && ".join(
        [
            "set -euo pipefail",
            "python3 -m pip install -q -U huggingface_hub",
            f"export HF_TOKEN={shlex.quote(token)}",
            f"mkdir -p {shlex.quote(args.pod_data_root)}",
            (
                "hf download "
                f"{shlex.quote(repo_id)} "
                "--repo-type dataset "
                f"--local-dir {shlex.quote(args.pod_data_root)} "
                "--token $HF_TOKEN"
            ),
        ]
    )

    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        print(f"RunPod download attempt {attempt}/{MAX_DOWNLOAD_ATTEMPTS}")
        try:
            subprocess.run(
                ssh_base_cmd(args) + [remote_cmd],
                check=True,
            )
            print("RunPod download finished.")
            return
        except subprocess.CalledProcessError as exc:
            print(f"RunPod download failed on attempt {attempt}: {exc}", file=sys.stderr)
            if attempt == MAX_DOWNLOAD_ATTEMPTS:
                raise
            time.sleep(30)


def verify_remote_dataset(args: argparse.Namespace) -> tuple[int, str]:
    verify_cmd = (
        f"for d in {' '.join(EXPECTED_FOLDERS)}; do "
        f"printf '%s: ' \"$d\"; "
        f"find {shlex.quote(args.pod_data_root)}/\"$d\" -type f "
        "\\( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' -o -name '*.webp' \\) "
        "2>/dev/null | wc -l; "
        "done; "
        f"find {shlex.quote(args.pod_data_root)} -type f "
        "\\( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' -o -name '*.webp' \\) "
        "| wc -l; "
        f"du -sh {shlex.quote(args.pod_data_root)} | awk '{{print $1}}'"
    )
    result = subprocess.run(
        ssh_base_cmd(args) + [verify_cmd],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    print("RunPod verification")
    for line in lines[:-2]:
        print(f"  {line}")
    total = int(lines[-2])
    size = lines[-1]
    print(f"  total images: {total:,}")
    print(f"  total size: {size}")
    return total, size


def main() -> None:
    args = parse_args()
    started = time.time()
    token = require_token()
    login(token=token)

    api = HfApi(token=token)
    whoami = api.whoami()
    username = whoami["name"]
    repo_id = f"{username}/{args.repo_name}"

    local_total, local_size_bytes = validate_local_dataset(args.dataset_root)

    if not args.skip_upload:
        upload_dataset(api, repo_id, args.dataset_root)

    if not args.skip_download:
        run_remote_download(args, repo_id, token)
        remote_total, remote_size = verify_remote_dataset(args)
        if remote_total != local_total:
            print(
                f"Remote image count {remote_total:,} does not match local {local_total:,}.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        remote_total = local_total
        remote_size = human_bytes(local_size_bytes)

    elapsed = time.time() - started
    print("Summary")
    print(f"  repo: {repo_id}")
    print(f"  local path: {args.dataset_root}")
    print(f"  pod path: {args.pod_data_root}")
    print(f"  total files uploaded: {local_total:,}")
    print(f"  total size: {human_bytes(local_size_bytes)}")
    print(f"  pod size after download: {remote_size}")
    print(f"  time taken: {human_duration(elapsed)}")


if __name__ == "__main__":
    main()
