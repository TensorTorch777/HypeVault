"""Browser validation for the research explanation panel.

A skipped run is not a successful browser validation.

Generic CI, including `python -m unittest discover`, does not provide an
authorized account, a password, a running frontend and API, or Playwright
Chromium. When `HYPEVAULT_E2E_RESEARCH_PASSWORD` is unset and the run was not
explicitly requested, this module skips.

An explicit local run sets `HYPEVAULT_E2E_BROWSER_E2E=1`. That run fails if the
password is missing. A real browser E2E also needs:

- `HYPEVAULT_E2E_RESEARCH_PASSWORD` in the environment only, never in the repo
- `HYPEVAULT_E2E_RESEARCH_EMAIL` (default `e2e-research@example.com`) on the API research allowlist
- the frontend (`HYPEVAULT_E2E_FRONTEND`, default `http://localhost:3000`) and its API
- Playwright Chromium
- an eligible non-test image from the parity protocol on disk, not the locked final test set

This file does not contain a password.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest import mock

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO / "ml_rtx5080" / "experiments" / "dual_model_triton_v1" / "parity_protocol_v1.json"
SHOTS = Path("/tmp/hypevault-xai-browser")
FRONTEND = os.environ.get("HYPEVAULT_E2E_FRONTEND", "http://localhost:3000")
UNAVAILABLE_REASON = "The explanation timed out before every region was measured."
NOT_VALIDATION = "This skip is not a successful browser validation."


def browser_e2e_gate() -> str | None:
    """Return a skip reason, or None when the credentialed browser E2E should run.

    An explicit request without a password fails. A generic run without a password skips.
    """
    explicit = os.environ.get("HYPEVAULT_E2E_BROWSER_E2E", "").strip().lower() in {"1", "true", "yes"}
    password = os.environ.get("HYPEVAULT_E2E_RESEARCH_PASSWORD", "").strip()
    if password:
        return None
    if explicit:
        raise AssertionError(
            "HYPEVAULT_E2E_RESEARCH_PASSWORD is required for an explicitly requested browser E2E run. "
            "This failure is not a successful browser validation."
        )
    return (
        f"{NOT_VALIDATION} Generic CI does not run this credentialed browser E2E. "
        "Set HYPEVAULT_E2E_BROWSER_E2E=1 and HYPEVAULT_E2E_RESEARCH_PASSWORD, "
        "and provide an allowlisted research account, the frontend, the API, Playwright Chromium, "
        "and an eligible non-test image."
    )


def _eligible_image() -> tuple[Path, str]:
    protocol = json.loads(PROTOCOL.read_text())
    for case in protocol["inputs"]:
        sample_id = case["sample_id"]
        lowered = sample_id.lower()
        if case.get("split") == "test":
            continue
        if any(token in lowered for token in ("final_test", "/ood", "watch_like_ood")):
            continue
        path = REPO / sample_id
        if not path.is_file():
            continue
        return path, path.parent.name
    raise unittest.SkipTest("No eligible non-test image was found in the parity protocol")


def _explain_body(status: str, *, model: str, decision: str, reason: str | None = None) -> dict:
    if status == "unavailable":
        return {
            "status": "unavailable",
            "reason": reason,
            "research_only": True,
            "publication_decision": "BLOCKED",
            "independent_authentication": False,
            "sensitivity": None,
            "explanation": None,
        }
    return {
        "status": "ok",
        "research_only": True,
        "publication_decision": "BLOCKED",
        "independent_authentication": False,
        "classification": {"decision": decision, "model": model, "model_version": "1"},
        "sensitivity": {
            "method": {"name": "region_occlusion_v1", "version": "1", "weak_abs_delta": 0.001},
            "model_id": model,
            "model_version": "1",
            "preprocessing_id": "legacy_square_resize_504_imagenet",
            "baseline_logit": 1.0,
            "baseline_decision": decision,
            "max_abs_delta_logit": 0.2,
            "evidence_supports_visual_summary": True,
            "patches": [
                {"row": 0, "col": 0, "x": 0, "y": 0, "width": 1, "height": 1, "delta_logit": 0.2}
            ],
        },
        "explanation": {
            "observation": f"The existing model classified this image as {decision}.",
            "hypothesis": "Only measured score changes are reported.",
            "limitations": "A sensitivity map is not proof of authenticity.",
            "source": "deterministic_fallback",
        },
    }


class BrowserE2EGateTests(unittest.TestCase):
    def test_generic_run_without_a_password_skips_and_does_not_claim_validation(self) -> None:
        env = os.environ.copy()
        env.pop("HYPEVAULT_E2E_RESEARCH_PASSWORD", None)
        env.pop("HYPEVAULT_E2E_BROWSER_E2E", None)
        with mock.patch.dict(os.environ, env, clear=True):
            reason = browser_e2e_gate()
        self.assertIsNotNone(reason)
        self.assertIn(NOT_VALIDATION, reason or "")

    def test_explicit_run_without_a_password_fails(self) -> None:
        env = os.environ.copy()
        env.pop("HYPEVAULT_E2E_RESEARCH_PASSWORD", None)
        env["HYPEVAULT_E2E_BROWSER_E2E"] = "1"
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(AssertionError) as caught:
                browser_e2e_gate()
        self.assertIn("not a successful browser validation", str(caught.exception))


class ResearchExplanationBrowserTests(unittest.TestCase):
    def test_research_explanation_end_to_end(self) -> None:
        reason = browser_e2e_gate()
        if reason is not None:
            self.skipTest(reason)
        email = os.environ.get("HYPEVAULT_E2E_RESEARCH_EMAIL", "e2e-research@example.com")
        password = os.environ["HYPEVAULT_E2E_RESEARCH_PASSWORD"]
        image, brand = _eligible_image()
        self.assertNotIn("final_test", str(image))
        SHOTS.mkdir(parents=True, exist_ok=True)
        passed: list[str] = []

        def check(name: str, condition: bool, detail: str = "") -> None:
            if not condition:
                raise AssertionError(f"{name} failed{': ' + detail if detail else ''}")
            passed.append(name)

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page.goto(f"{FRONTEND}/login", wait_until="networkidle")
            page.locator("#email").fill(email)
            page.locator("#pw").fill(password)
            page.get_by_role("button", name="Continue", exact=True).click()
            page.wait_for_url(lambda url: "/login" not in url, timeout=20000)
            check("signed in with the authorized research account", "/login" not in page.url)

            page.goto(f"{FRONTEND}/research", wait_until="networkidle")
            page.get_by_role("heading", name="Five-brand prototype").wait_for()
            page.get_by_text("Serving status").first.wait_for()
            serving = page.locator("li").all_inner_texts()
            dinov2 = next(item for item in serving if "DINOv2 — Legacy" in item and "Serving status" in item)
            dinov3 = next(item for item in serving if "DINOv3 — Experimental" in item and "Serving status" in item)
            check("DINOv2 serving state is available", "available" in dinov2, dinov2)
            check(
                "DINOv3 stays labeled experimental and available",
                "available" in dinov3 and "Not approved for production" in dinov3,
                dinov3,
            )

            page.locator("#research-brand").select_option(brand)
            page.locator("#research-model").select_option("dinov2_legacy")
            page.locator('input[type="file"]').set_input_files(str(image))
            with page.expect_response(lambda response: response.url.endswith("/research/verify") and response.request.method == "POST") as verify_info:
                with page.expect_response(lambda response: response.url.endswith("/research/explain") and response.request.method == "POST", timeout=90000) as explain_info:
                    page.get_by_role("button", name="Run research demo").click()
            verify = verify_info.value.json()
            explained = explain_info.value.json()
            decision = verify["decision"]
            model = verify["model"]
            page.get_by_text(f"Model classification: {decision}").wait_for(timeout=20000)
            check("original classification is visible", decision in {"AUTHENTIC", "FAKE", "REVIEW"}, decision)
            check("explain model matches the displayed classification", explained["classification"]["model"] == model, model)
            check("explain decision matches the displayed classification", explained["classification"]["decision"] == decision, decision)
            check("explain sensitivity model matches", explained["sensitivity"]["model_id"] == model)
            check("explain sensitivity decision matches", explained["sensitivity"]["baseline_decision"] == decision)
            check("publication stays blocked", explained["publication_decision"] == "BLOCKED")
            check("language layer did not call an external model", explained["explanation"]["source"] == "deterministic_fallback")

            section = page.get_by_role("region", name="What influenced this model classification?")
            alignment = section.evaluate(
                """(node, payload) => {
                  const img = node.querySelector("img");
                  const overlays = [...node.querySelectorAll("div")].filter((el) => el.style.left.endsWith("%"));
                  const threshold = payload.sensitivity.method.weak_abs_delta ?? 0.001;
                  const highlighted = payload.sensitivity.patches.filter(
                    (patch) => Math.abs(patch.delta_logit) >= threshold
                  );
                  const mismatches = highlighted.filter((patch) => {
                    const left = (patch.x / img.naturalWidth) * 100;
                    const top = (patch.y / img.naturalHeight) * 100;
                    const width = (patch.width / img.naturalWidth) * 100;
                    const height = (patch.height / img.naturalHeight) * 100;
                    return !overlays.some((el) =>
                      Math.abs(parseFloat(el.style.left) - left) < 0.2 &&
                      Math.abs(parseFloat(el.style.top) - top) < 0.2 &&
                      Math.abs(parseFloat(el.style.width) - width) < 0.2 &&
                      Math.abs(parseFloat(el.style.height) - height) < 0.2
                    );
                  });
                  const rows = [...node.querySelectorAll("tbody tr")].map((row) =>
                    [...row.children].map((cell) => cell.textContent.trim())
                  );
                  const ranked = [...payload.sensitivity.patches].sort(
                    (a, b) => Math.abs(b.delta_logit) - Math.abs(a.delta_logit)
                  );
                  const expected = ranked.map((patch) => [
                    String(patch.row),
                    String(patch.col),
                    `${patch.delta_logit >= 0 ? "+" : ""}${patch.delta_logit.toFixed(4)}`,
                  ]);
                  return {
                    overlayCount: overlays.length,
                    highlighted: highlighted.length,
                    mismatches: mismatches.length,
                    rowCount: rows.length,
                    rowsMatch: JSON.stringify(rows) === JSON.stringify(expected),
                    grid: ranked.map((patch) => [patch.row, patch.col]),
                  };
                }""",
                explained,
            )
            grid = {tuple(pair) for pair in alignment["grid"]}
            check("overlay cells align with the submitted image", alignment["mismatches"] == 0 and alignment["overlayCount"] == alignment["highlighted"], json.dumps(alignment))
            check("signed logit table has one row per grid cell", alignment["rowCount"] == 16 and alignment["rowsMatch"], json.dumps(alignment))
            check("table covers the 4 by 4 grid", grid == {(row, col) for row in range(4) for col in range(4)})

            page.screenshot(path=str(SHOTS / "desktop-explanation.png"), full_page=True)
            page.set_viewport_size({"width": 390, "height": 844})
            page.screenshot(path=str(SHOTS / "mobile-explanation.png"), full_page=True)
            page.set_viewport_size({"width": 1280, "height": 800})
            check("desktop and mobile screenshots were captured", (SHOTS / "desktop-explanation.png").stat().st_size > 0 and (SHOTS / "mobile-explanation.png").stat().st_size > 0)

            script = {"mode": "unavailable"}

            def fulfill_explain(route) -> None:
                mode = script["mode"]
                if mode == "unavailable":
                    body = _explain_body("unavailable", model=model, decision=decision, reason=UNAVAILABLE_REASON)
                elif mode == "model":
                    body = _explain_body("ok", model="DINOV3_RESEARCH_PROTOTYPE", decision=decision)
                else:
                    body = _explain_body("ok", model=model, decision="AUTHENTIC" if decision != "AUTHENTIC" else "FAKE")
                route.fulfill(status=200, content_type="application/json", body=json.dumps(body))

            page.route("**/research/explain", fulfill_explain)

            def rerun(mode: str) -> None:
                script["mode"] = mode
                with page.expect_response(lambda response: response.url.endswith("/research/verify") and response.request.method == "POST"):
                    with page.expect_response(lambda response: response.url.endswith("/research/explain") and response.request.method == "POST"):
                        page.get_by_role("button", name="Run research demo").click()
                page.get_by_text(f"Model classification: {decision}").wait_for(timeout=20000)

            rerun("unavailable")
            check(
                "unavailable explanation shows the server reason",
                UNAVAILABLE_REASON in section.inner_text(),
            )
            check(
                "unavailable explanation leaves the original classification visible",
                f"Model classification: {decision}" in page.locator("main").inner_text(),
            )
            page.screenshot(path=str(SHOTS / "desktop-unavailable.png"), full_page=True)

            rerun("model")
            check(
                "model mismatch is discarded",
                "A consistent explanation could not be produced." in section.inner_text(),
            )
            check(
                "model mismatch leaves the original classification visible",
                f"Model classification: {decision}" in page.locator("main").inner_text(),
            )

            rerun("decision")
            check(
                "decision mismatch is discarded",
                "A consistent explanation could not be produced." in section.inner_text(),
            )
            check(
                "decision mismatch leaves the original classification visible",
                f"Model classification: {decision}" in page.locator("main").inner_text(),
            )
            browser.close()

        print("PASSED ASSERTIONS:")
        for name in passed:
            print(f"- {name}")


if __name__ == "__main__":
    unittest.main()
