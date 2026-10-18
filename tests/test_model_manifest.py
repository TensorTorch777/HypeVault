"""The model manifest is the architecture contract, not backend defaults."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml_rtx5080"))

import manifest as mf


class ModelManifestTests(unittest.TestCase):
    def test_schema_lists_the_required_contract_fields(self) -> None:
        schema = mf.load_schema()
        self.assertEqual(schema["required"], list(mf.REQUIRED_FIELDS))
        self.assertIn("dinov2", schema["properties"]["model_family"]["enum"])
        self.assertIn("dinov3", schema["properties"]["model_family"]["enum"])
        self.assertNotIn(518, schema["properties"]["image_size"].get("enum", []))

    def test_rtx_dinov2_manifest_records_504_and_leaves_thresholds_unset(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "watch.jpg"
            image.write_bytes(b"not-a-real-jpeg")
            checkpoint = root / "best_model.pt"
            checkpoint.write_bytes(b"checkpoint-bytes")
            run = root / "run"
            run.mkdir()
            config = {
                "model_name": "vit_base_patch14_dinov2.lvd142m",
                "img_size": 504,
                "seed": 42,
            }
            written = mf.write_training_manifest(
                run,
                backbone=config["model_name"],
                image_size=config["img_size"],
                training_config=config,
                checkpoint_path=checkpoint,
                dataset_samples=[(image, 0)],
                repo_root=root,
            )
            self.assertEqual(written["model_family"], "dinov2")
            self.assertEqual(written["backbone"], "vit_base_patch14_dinov2.lvd142m")
            self.assertEqual(written["backbone_revision"], "lvd142m")
            self.assertEqual(written["image_size"], 504)
            self.assertEqual(written["patch_size"], 14)
            self.assertIsNone(written["calibration_version"])
            self.assertIsNone(written["fake_threshold"])
            self.assertIsNone(written["review_thresholds"])
            self.assertEqual(written["checkpoint_sha256"], mf.sha256_file(checkpoint))
            self.assertEqual(len(written["training_config_hash"]), 64)
            self.assertEqual(len(written["dataset_hash"]), 64)
            self.assertNotEqual(written["image_size"], 518)

    def test_dinov3_rejects_504(self) -> None:
        with self.assertRaises(ValueError):
            mf.build_model_manifest(
                backbone="vit_base_patch16_dinov3.lvd1689m",
                image_size=504,
                training_config={"img_size": 504},
            )

    def test_dinov3_512_is_accepted_without_invented_thresholds(self) -> None:
        manifest = mf.build_model_manifest(
            backbone="vit_base_patch16_dinov3.lvd1689m",
            image_size=512,
            training_config={"img_size": 512, "seed": 42},
            repo_root=Path("/tmp"),
        )
        self.assertEqual(manifest["model_family"], "dinov3")
        self.assertEqual(manifest["patch_size"], 16)
        self.assertEqual(manifest["image_size"], 512)
        self.assertEqual(manifest["backbone_revision"], "lvd1689m")
        self.assertIsNone(manifest["fake_threshold"])

    def test_write_does_not_touch_the_checkpoint(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "run"
            out.mkdir()
            checkpoint = root / "best_model.pt"
            payload = b"leave-me-alone"
            checkpoint.write_bytes(payload)
            mf.write_training_manifest(
                out,
                backbone="vit_base_patch14_dinov2.lvd142m",
                image_size=504,
                training_config={"model_name": "vit_base_patch14_dinov2.lvd142m"},
                checkpoint_path=checkpoint,
                repo_root=root,
            )
            self.assertEqual(checkpoint.read_bytes(), payload)
            saved = json.loads((out / "model_manifest.json").read_text())
            self.assertEqual(saved["checkpoint_sha256"], mf.sha256_bytes(payload))
            self.assertFalse((out / "best_model.pt").exists())


if __name__ == "__main__":
    unittest.main()
