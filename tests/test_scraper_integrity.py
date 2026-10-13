"""Offline scraper integrity tests. No network, GPU, or protected datasets."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scraper import common
from scraper.common import (
    STATUS_SYNTHETIC,
    STATUS_UNVERIFIED_WEB,
    STATUS_VERIFIED_AUTHENTIC,
    STATUS_VERIFIED_COUNTERFEIT,
    ContentIndex,
    OOMRetryController,
    ResumeStateError,
    StoreStats,
    ValidationError,
    assert_not_legacy_training_dir,
    average_hash,
    build_manifest_record,
    count_valid_images,
    decode_image,
    load_resume_state,
    make_tiny_jpeg_bytes,
    mark_query_completed,
    normalize_image_bytes,
    repo_root,
    require_generation_opt_in,
    require_network_opt_in,
    resolve_under_repo,
    save_resume_state,
    sha256_bytes,
    store_validated_image,
)


class ImportSafetyTests(unittest.TestCase):
    def test_importing_scripts_does_not_start_collection_or_generation(self) -> None:
        modules = [
            "scraper.scraper",
            "scraper.balance_scraper",
            "scraper.sneaker_scraper",
            "scraper.generate_label1",
            "scraper.generate_label1_sneakers",
        ]
        for name in modules:
            with self.subTest(module=name):
                with mock.patch.dict(sys.modules):
                    # Ensure heavy deps are not required at import for collectors;
                    # generators must not call load_pipeline on import.
                    mod = importlib.import_module(name)
                    self.assertTrue(hasattr(mod, "main"))
                    self.assertTrue(hasattr(mod, "parse_args"))


class PathAndLabelTests(unittest.TestCase):
    def test_repo_root_and_relative_resolution_from_other_cwd(self) -> None:
        self.assertEqual(repo_root(), REPO)
        with tempfile.TemporaryDirectory() as tmp:
            outside = Path(tmp)
            resolved = resolve_under_repo("local_data/scraper_artifacts", "local_data/scraper_artifacts")
            self.assertTrue(str(resolved).startswith(str(REPO)))
            # Changing cwd must not change repo-relative resolution.
            old = Path.cwd()
            try:
                import os

                os.chdir(outside)
                again = resolve_under_repo("local_data/scraper_artifacts", "local_data/scraper_artifacts")
                self.assertEqual(resolved, again)
            finally:
                os.chdir(old)

    def test_unverified_and_synthetic_never_get_verified_labels(self) -> None:
        for status in (STATUS_UNVERIFIED_WEB, STATUS_SYNTHETIC):
            record = build_manifest_record(
                relative_image_path="unverified_web_images/watches/Demo/a.jpg",
                stored_sha256="a" * 64,
                original_sha256="b" * 64,
                width=96,
                height=96,
                image_format="JPEG",
                byte_size=100,
                family="watches",
                claimed_brand="Demo",
                claimed_model=None,
                search_query="q",
                search_provider="test",
                source_image_url="UNKNOWN",
                source_page_url="UNKNOWN",
                label_status=status,
            )
            self.assertIsNone(record["target_label"])
            self.assertFalse(record["intended_use_eligibility"]["authenticity_benchmark"])
            self.assertNotIn(record["label_status"], {STATUS_VERIFIED_AUTHENTIC, STATUS_VERIFIED_COUNTERFEIT})

        with self.assertRaises(ValueError):
            build_manifest_record(
                relative_image_path="x.jpg",
                stored_sha256="a" * 64,
                original_sha256=None,
                width=1,
                height=1,
                image_format="JPEG",
                byte_size=1,
                family="watches",
                claimed_brand="Demo",
                claimed_model=None,
                search_query=None,
                search_provider=None,
                source_image_url=None,
                source_page_url=None,
                label_status=STATUS_VERIFIED_AUTHENTIC,
            )


class ImageValidationTests(unittest.TestCase):
    def test_html_and_corrupt_rejected_valid_accepted(self) -> None:
        with self.assertRaises(ValidationError):
            decode_image(b"<html><body>error</body></html>")
        with self.assertRaises(ValidationError):
            decode_image(b"not-an-image")
        data = make_tiny_jpeg_bytes()
        image, fmt = decode_image(data)
        self.assertEqual(fmt, "JPEG")
        self.assertGreaterEqual(image.width, 64)

    def test_sha256_deterministic_and_exact_duplicate_not_double_counted(self) -> None:
        data = make_tiny_jpeg_bytes((1, 2, 3))
        self.assertEqual(sha256_bytes(data), sha256_bytes(data))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dest = root / "unverified_web_images" / "watches" / "Demo"
            dest.mkdir(parents=True)
            stats = StoreStats()
            index = ContentIndex.load(dest / "content_index.json")
            first = store_validated_image(
                data,
                dest_dir=dest,
                artifact_root=root,
                family="watches",
                claimed_brand="Demo",
                claimed_model=None,
                label_status=STATUS_UNVERIFIED_WEB,
                index=index,
                stats=stats,
            )
            second = store_validated_image(
                data,
                dest_dir=dest,
                artifact_root=root,
                family="watches",
                claimed_brand="Demo",
                claimed_model=None,
                label_status=STATUS_UNVERIFIED_WEB,
                index=index,
                stats=stats,
            )
            self.assertIsNotNone(first)
            self.assertIsNone(second)
            self.assertEqual(stats.stored, 1)
            self.assertEqual(stats.exact_duplicates, 1)
            self.assertEqual(count_valid_images(dest), 1)

    def test_near_duplicate_flagged_not_auto_identical(self) -> None:
        data_a = make_tiny_jpeg_bytes((10, 20, 30))
        data_b = make_tiny_jpeg_bytes((200, 100, 50))
        self.assertNotEqual(sha256_bytes(data_a), sha256_bytes(data_b))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dest = root / "unverified_web_images" / "watches" / "Demo"
            dest.mkdir(parents=True)
            stats = StoreStats()
            index = ContentIndex.load(dest / "content_index.json")
            first = store_validated_image(
                data_a,
                dest_dir=dest,
                artifact_root=root,
                family="watches",
                claimed_brand="Demo",
                claimed_model=None,
                label_status=STATUS_UNVERIFIED_WEB,
                index=index,
                stats=stats,
            )
            self.assertIsNotNone(first)
            # Seed a perceptual hash equal to the second image so it is flagged for review
            # without being treated as an exact duplicate.
            from io import BytesIO

            phash_b = average_hash(Image.open(BytesIO(data_b)))
            index.perceptual_hashes[phash_b] = first["image_path"]
            record = store_validated_image(
                data_b,
                dest_dir=dest,
                artifact_root=root,
                family="watches",
                claimed_brand="Demo",
                claimed_model=None,
                label_status=STATUS_UNVERIFIED_WEB,
                index=index,
                stats=stats,
            )
            self.assertIsNotNone(record)
            self.assertEqual(record["duplicate_status"], "NEAR_DUPLICATE_CANDIDATE")
            self.assertEqual(record["near_duplicate_of"], first["image_path"])
            self.assertEqual(stats.near_duplicate_candidates, 1)
            self.assertEqual(stats.stored, 2)

    def test_oversized_and_invalid_dimensions_rejected(self) -> None:
        huge = make_tiny_jpeg_bytes()
        with self.assertRaises(ValidationError):
            decode_image(huge, max_bytes=10)
        tiny = Image.new("RGB", (8, 8), (1, 1, 1))
        with self.assertRaises(ValidationError):
            decode_image(normalize_image_bytes(tiny))

    def test_partial_invalid_image_not_counted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            partial = dest / "broken.jpg"
            partial.write_bytes(b"\xff\xd8\xff incomplete")
            self.assertEqual(count_valid_images(dest), 0)


class ManifestAndResumeTests(unittest.TestCase):
    def test_manifest_required_fields_and_missing_source_unknown(self) -> None:
        record = build_manifest_record(
            relative_image_path="unverified_web_images/watches/Demo/x.jpg",
            stored_sha256="a" * 64,
            original_sha256=None,
            width=96,
            height=96,
            image_format="JPEG",
            byte_size=12,
            family="watches",
            claimed_brand="Demo",
            claimed_model=None,
            search_query="q",
            search_provider="test",
            source_image_url=None,
            source_page_url=None,
            label_status=STATUS_UNVERIFIED_WEB,
        )
        required = {
            "schema_version",
            "record_id",
            "image_path",
            "stored_sha256",
            "original_sha256",
            "label_status",
            "target_label",
            "evidence_status",
            "rights_review_status",
            "intended_use_eligibility",
            "provenance_completeness",
            "source_image_url",
        }
        self.assertTrue(required <= set(record))
        self.assertEqual(record["source_image_url"], "UNKNOWN")
        self.assertEqual(record["provenance_completeness"], "INCOMPLETE")
        self.assertEqual(record["original_sha256"], "NOT_RECORDED")

    def test_resume_valid_and_malformed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resume.json"
            state = load_resume_state(path)
            mark_query_completed(state, "q1", success=True)
            mark_query_completed(state, "q2", success=False)
            save_resume_state(path, state)
            loaded = load_resume_state(path)
            self.assertIn("q1", loaded["completed_queries"])
            self.assertIn("q2", loaded["failed_queries"])
            path.write_text("{not-json", encoding="utf-8")
            with self.assertRaises(ResumeStateError):
                load_resume_state(path)

    def test_network_and_generation_blocked_without_opt_in(self) -> None:
        with self.assertRaises(SystemExit):
            require_network_opt_in(False)
        with self.assertRaises(SystemExit):
            require_generation_opt_in(False)
        require_network_opt_in(True)
        require_generation_opt_in(True)

    def test_oom_retry_terminates_without_gpu(self) -> None:
        controller = OOMRetryController(max_retries=2)
        class OutOfMemoryError(RuntimeError):
            pass

        self.assertTrue(controller.should_retry(OutOfMemoryError("CUDA out of memory")))
        self.assertTrue(controller.should_retry(OutOfMemoryError("CUDA out of memory")))
        self.assertFalse(controller.should_retry(OutOfMemoryError("CUDA out of memory")))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "failures.jsonl"
            controller.record_failure(path, prompt="p", seed=1, error="oom")
            line = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertEqual(line["error_category"], "CUDA_OOM")

    def test_existing_outputs_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dest = root / "synthetic_research_only" / "watches" / "Demo"
            dest.mkdir(parents=True)
            data = make_tiny_jpeg_bytes((9, 9, 9))
            stats = StoreStats()
            index = ContentIndex.load(dest / "content_index.json")
            first = store_validated_image(
                data,
                dest_dir=dest,
                artifact_root=root,
                family="watches",
                claimed_brand="Demo",
                claimed_model=None,
                label_status=STATUS_SYNTHETIC,
                generation={"seed": 1, "prompt": "p", "model_id": "test"},
                index=index,
                stats=stats,
            )
            path = root / first["image_path"]
            before = path.read_bytes()
            store_validated_image(
                data,
                dest_dir=dest,
                artifact_root=root,
                family="watches",
                claimed_brand="Demo",
                claimed_model=None,
                label_status=STATUS_SYNTHETIC,
                generation={"seed": 1, "prompt": "p", "model_id": "test"},
                index=index,
                stats=stats,
            )
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(stats.stored, 1)

    def test_legacy_training_dirs_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            banned = Path(tmp) / "Label_0_Watches" / "Demo"
            banned.mkdir(parents=True)
            with self.assertRaises(common.ScraperError):
                assert_not_legacy_training_dir(banned)

    def test_cli_dry_run_and_help(self) -> None:
        for script in (
            "scraper/scraper.py",
            "scraper/balance_scraper.py",
            "scraper/sneaker_scraper.py",
            "scraper/generate_label1.py",
            "scraper/generate_label1_sneakers.py",
        ):
            help_proc = subprocess.run(
                [sys.executable, str(REPO / script), "--help"],
                cwd=str(REPO),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(help_proc.returncode, 0, help_proc.stderr)
            self.assertIn("usage", help_proc.stdout.lower())

            dry = subprocess.run(
                [sys.executable, str(REPO / script), "--dry-run"],
                cwd=str(REPO),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(dry.returncode, 0, dry.stderr)
            self.assertIn("DRY-RUN", dry.stdout)

    def test_gitignore_covers_artifact_paths(self) -> None:
        root_ignore = (REPO / ".gitignore").read_text(encoding="utf-8")
        scraper_ignore = (REPO / "scraper" / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("local_data/scraper_artifacts/", root_ignore)
        self.assertIn("Watches.csv", root_ignore)
        self.assertIn("*.jpg", scraper_ignore)
        # Verify git check-ignore behavior for intended paths.
        sample = "local_data/scraper_artifacts/unverified_web_images/watches/Demo/a.jpg"
        proc = subprocess.run(
            ["git", "check-ignore", "-v", sample],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("local_data/scraper_artifacts", proc.stdout)


if __name__ == "__main__":
    unittest.main()
