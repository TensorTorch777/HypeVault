"""End-to-end checks for the dual-model research selector against a running stack.

Needs: API on --api (research mode, RESEARCH_USER_EMAILS containing --research-email),
Triton gRPC/HTTP on --triton-grpc/--triton-http, the hypevault-postgres container, and
optionally a second API in production mode on --prod-api. Creates two E2E accounts and one
legacy listing, then deletes exactly those rows. Reads only non-test images listed in the
pre-registered parity protocol. Writes ml_rtx5080/experiments/dual_model_triton_v1/integration_test_results.json.
"""

from __future__ import annotations

import argparse
import json
import secrets
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import tritonclient.grpc as grpcclient

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "ml_rtx5080" / "experiments" / "dual_model_triton_v1" / "integration_test_results.json"
PROTOCOL = REPO / "ml_rtx5080" / "experiments" / "dual_model_triton_v1" / "parity_protocol_v1.json"
FROZEN_SHA = "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f"
LEGACY = "Legacy screening — not verified"
DEMO = "Demo listing — not verified"


def psql(sql: str) -> str:
    return subprocess.run(
        ["docker", "exec", "hypevault-postgres", "psql", "-U", "hypevault", "-d", "hypevault", "-At", "-c", sql],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def listing_count() -> int:
    return int(psql("select count(*) from listings;"))


def triton_count(client, name: str) -> int | None:
    try:
        stats = client.get_inference_statistics(name, "1", as_json=True)["model_stats"]
    except Exception:
        return None
    return int(stats[0].get("inference_count", 0)) if stats else 0


def register_and_login(api: str, email: str, role: str) -> str:
    password = "E2e" + secrets.token_hex(8) + "9"
    with httpx.Client(base_url=api, timeout=30) as client:
        created = client.post("/auth/register", json={"email": email, "password": password, "role": role})
        if created.status_code not in (201, 409):
            raise RuntimeError(f"register {email} returned {created.status_code}: {created.text}")
        if created.status_code == 409:
            raise RuntimeError(f"{email} already exists; refusing to reuse an account with an unknown password")
        login = client.post("/auth/login", json={"email": email, "password": password})
        login.raise_for_status()
        return login.json()["access_token"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--prod-api", default="http://127.0.0.1:8010")
    parser.add_argument("--frontend", default="http://127.0.0.1:3000")
    parser.add_argument("--triton-grpc", default="localhost:18001")
    parser.add_argument("--triton-http", default="http://localhost:18000")
    parser.add_argument("--triton-desc", required=True, help="Image, digest, and whether GPU was used; recorded verbatim")
    parser.add_argument("--gpu-used", action="store_true", help="Set only when Triton metrics showed GPU execution")
    parser.add_argument("--results", default=str(OUT))
    parser.add_argument("--research-email", default="e2e-research@example.com")
    parser.add_argument("--buyer-email", default="e2e-buyer@example.com")
    parser.add_argument("--seller-email", default="e2e-seller@example.com")
    args = parser.parse_args()

    protocol = json.loads(PROTOCOL.read_text())
    authentic_case = next(c for c in protocol["inputs"] if c["sample_id"].startswith("Label_0_Watches/") and c["split"] != "test")
    image_path = REPO / authentic_case["sample_id"]
    brand = image_path.parent.name
    image = image_path.read_bytes()
    triton = grpcclient.InferenceServerClient(args.triton_grpc)

    flows: dict[str, dict] = {}
    models: dict = {"models": []}
    created_emails = [args.research_email, args.buyer_email, args.seller_email]
    baseline_listings = listing_count()
    baseline_users = int(psql("select count(*) from users;"))

    def record(name: str, passed: bool, evidence: dict) -> None:
        flows[name] = {"result": "PASS" if passed else "FAIL", "evidence": evidence}

    def post(token: str | None, data: dict, payload: bytes = image, content_type: str = "image/jpeg", base: str | None = None):
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        return httpx.post(
            f"{base or args.api}/research/verify",
            headers=headers,
            files={"image": (image_path.name, payload, content_type)},
            data=data,
            timeout=120,
        )

    try:
        escalation = httpx.post(
            f"{args.api}/auth/register",
            json={"email": "e2e-admin@example.com", "password": "E2eAdmin12345", "role": "admin"},
            timeout=30,
        )
        record(
            "registration_refuses_privileged_role",
            escalation.status_code == 422 and psql("select count(*) from users where email='e2e-admin@example.com';") == "0",
            {"http_status": escalation.status_code},
        )

        research = register_and_login(args.api, args.research_email, "buyer")
        buyer = register_and_login(args.api, args.buyer_email, "buyer")
        seller = register_and_login(args.api, args.seller_email, "seller")

        page = httpx.get(f"{args.frontend}/research", timeout=30)
        record("1_research_page_renders", page.status_code == 200 and "Five-brand prototype" in page.text, {"http_status": page.status_code})

        models = httpx.get(f"{args.api}/research/models", headers={"Authorization": f"Bearer {research}"}, timeout=30).json()
        ready = {m["logical_id"]: m["ready"] for m in models["models"]}

        reasons = {m["logical_id"]: m.get("unavailable_reason") for m in models["models"]}
        old_before = triton_count(triton, "dinov2_classifier")
        v2_before, v3_before = triton_count(triton, "dinov2_vitb14_live"), triton_count(triton, "dinov3_authenticity_candidate")
        dinov2 = post(research, {"brand": brand, "logical_model": "dinov2_legacy"})
        v2_after, v3_after = triton_count(triton, "dinov2_vitb14_live"), triton_count(triton, "dinov3_authenticity_candidate")
        body = dinov2.json()
        if ready["dinov2_legacy"]:
            dinov2_ok = (
                dinov2.status_code == 200
                and body.get("model") == "LEGACY_DINOV2"
                and v2_after == (v2_before or 0) + 1
                and v3_after == v3_before
            )
        else:
            dinov2_ok = dinov2.status_code == 503 and body["decision"] is None and v2_after == v2_before and v3_after == v3_before
        record(
            "2_dinov2_selection_invokes_only_dinov2_vitb14_live",
            dinov2_ok,
            {
                "dinov2_ready": ready["dinov2_legacy"],
                "dinov2_unavailable_reason": reasons["dinov2_legacy"],
                "http_status": dinov2.status_code,
                "status": body.get("status"),
                "decision": body.get("decision"),
                "dinov2_vitb14_live_inference_count_before_after": [v2_before, v2_after],
                "dinov3_inference_count_before_after": [v3_before, v3_after],
                "inference_executed": bool(ready["dinov2_legacy"]),
            },
        )
        record(
            "2b_old_vitg14_dinov2_classifier_not_routed",
            triton_count(triton, "dinov2_classifier") == old_before,
            {"dinov2_classifier_inference_count_before_after": [old_before, triton_count(triton, "dinov2_classifier")]},
        )

        v3_before = triton_count(triton, "dinov3_authenticity_candidate")
        dinov3 = post(research, {"brand": brand, "logical_model": "dinov3_experimental"})
        v3_after = triton_count(triton, "dinov3_authenticity_candidate")
        body3 = dinov3.json()
        record(
            "3_dinov3_selection_invokes_only_dinov3_authenticity_candidate",
            dinov3.status_code == 200 and v3_after == (v3_before or 0) + 1,
            {"http_status": dinov3.status_code, "dinov3_inference_count_before_after": [v3_before, v3_after], "decision": body3.get("decision")},
        )
        record(
            "4_model_identity_in_response",
            body3.get("model") == "DINOV3_RESEARCH_PROTOTYPE"
            and body3.get("model_version") == "1"
            and body3.get("checkpoint_sha") == FROZEN_SHA
            and body3.get("brand_verification") == "NOT_PERFORMED"
            and body3.get("research_only") is True
            and body3.get("production_ready") is False
            and body3.get("publication_decision") == "BLOCKED",
            {"response": body3, "ui": "checked separately in the browser"},
        )

        unload = httpx.post(f"{args.triton_http}/v2/repository/models/dinov3_authenticity_candidate/unload", timeout=60)
        for _ in range(30):
            if httpx.get(f"{args.triton_http}/v2/models/dinov3_authenticity_candidate/versions/1/ready", timeout=5).status_code != 200:
                break
            time.sleep(1)
        unavailable = post(research, {"brand": brand, "logical_model": "dinov3_experimental"})
        models_down = httpx.get(f"{args.api}/research/models", headers={"Authorization": f"Bearer {research}"}, timeout=30).json()
        load = httpx.post(f"{args.triton_http}/v2/repository/models/dinov3_authenticity_candidate/load", timeout=300)
        for _ in range(60):
            if httpx.get(f"{args.triton_http}/v2/models/dinov3_authenticity_candidate/versions/1/ready", timeout=5).status_code == 200:
                break
            time.sleep(1)
        restored = post(research, {"brand": brand, "logical_model": "dinov3_experimental"})
        record(
            "5_unavailable_model_returns_controlled_error",
            unload.status_code == 200
            and unavailable.status_code == 503
            and unavailable.json()["decision"] is None
            and not any(m["ready"] for m in models_down["models"])
            and load.status_code == 200
            and restored.status_code == 200
            and restored.json()["decision"] == body3.get("decision"),
            {
                "unloaded_http_status": unavailable.status_code,
                "unloaded_body": unavailable.json(),
                "readiness_while_unloaded": {m["logical_id"]: m["ready"] for m in models_down["models"]},
                "reloaded_http_status": restored.status_code,
                "same_decision_after_reload": restored.json().get("decision") == body3.get("decision"),
            },
        )

        try:
            prod = post(research, {"brand": brand, "logical_model": "dinov3_experimental"}, base=args.prod_api)
            prod_live = httpx.post(
                f"{args.prod_api}/verify/authenticate",
                headers={"Authorization": f"Bearer {seller}"},
                files={"image": (image_path.name, image, "image/jpeg")},
                data={"brand": brand, "logical_model": "dinov3_experimental"},
                timeout=60,
            )
            record(
                "6_production_mode_blocks_dinov3",
                prod.status_code == 403
                and prod.json()["status"] == "POLICY_ERROR"
                and prod.json()["decision"] is None
                and prod_live.status_code == 403
                and prod_live.json()["decision"] is None,
                {"research_route": prod.json(), "live_route_with_dinov3_selector": prod_live.json()},
            )
        except httpx.HTTPError as exc:
            record("6_production_mode_blocks_dinov3", False, {"error": f"production-mode API unreachable: {exc}"})

        missing = post(research, {"brand": "", "logical_model": "dinov3_experimental"})
        unsupported = post(research, {"brand": "Rolex", "logical_model": "dinov3_experimental"})
        invalid = post(research, {"brand": brand, "logical_model": "dinov3_experimental"}, payload=b"not-an-image")
        unknown_model = post(research, {"brand": brand, "logical_model": "dinov3_authenticity_candidate"})
        record(
            "7_missing_unsupported_brand_and_invalid_image_fail_closed",
            missing.status_code == 422
            and unsupported.status_code == 422
            and invalid.status_code == 400
            and unknown_model.status_code == 422
            and all(r.json()["decision"] is None for r in (missing, unsupported, invalid, unknown_model)),
            {
                "missing_brand": [missing.status_code, missing.json()["status"]],
                "unsupported_brand": [unsupported.status_code, unsupported.json()["status"]],
                "invalid_image": [invalid.status_code, invalid.json()["status"]],
                "triton_name_as_selector": [unknown_model.status_code, unknown_model.json()["status"]],
            },
        )
        record(
            "8_model_errors_are_not_verdicts",
            all(r.json().get("status") not in ("AUTHENTIC", "FAKE", "REVIEW") for r in (dinov2, unavailable)),
            {"statuses": [dinov2.json()["status"], unavailable.json()["status"]]},
        )
        record(
            "9_research_calls_create_no_listings",
            listing_count() == baseline_listings,
            {"listings_before": baseline_listings, "listings_after_research_calls": listing_count()},
        )

        legacy = httpx.post(
            f"{args.api}/verify/authenticate",
            headers={"Authorization": f"Bearer {seller}"},
            files={"image": (image_path.name, image, "image/jpeg")},
            data={"brand": brand, "product_name": "E2E legacy check", "category": "watch"},
            timeout=300,
        )
        legacy_body = legacy.json()
        legacy_id = legacy_body.get("listing_id")
        legacy_status = psql(f"select status from listings where id='{legacy_id}';") if legacy_id else None
        record(
            "10_legacy_verification_cannot_auto_publish",
            legacy.status_code == 200 and legacy_status in ("pending", "rejected") and legacy_body.get("model") == "LEGACY_DINOV2",
            {
                "http_status": legacy.status_code,
                "model": legacy_body.get("model"),
                "verdict": legacy_body.get("verdict"),
                "listing_status_in_db": legacy_status,
                "backend": "local PyTorch DINOv2 (INFERENCE_BACKEND=torch); not Triton",
            },
        )

        recent = httpx.get(f"{args.api}/listings/recent", params={"limit": 48}, timeout=30).json()
        labels = {}
        for row in recent:
            labels.setdefault(row["customer_label"], 0)
            labels[row["customer_label"]] += 1
        seed_ok = all(r["customer_label"] == DEMO for r in recent if not r.get("s3_url"))
        image_ok = all(r["customer_label"] == LEGACY for r in recent if r.get("s3_url"))
        record(
            "11_historical_listing_labels",
            seed_ok and image_ok and labels.get(DEMO) == 20 and labels.get(LEGACY) == 19,
            {"live_listing_labels": labels},
        )

        denied = post(buyer, {"brand": brand, "logical_model": "dinov3_experimental"})
        denied_models = httpx.get(f"{args.api}/research/models", headers={"Authorization": f"Bearer {buyer}"}, timeout=30)
        anonymous = post(None, {"brand": brand, "logical_model": "dinov3_experimental"})
        record(
            "12_only_authorized_research_users",
            denied.status_code == 403 and denied_models.status_code == 403 and anonymous.status_code == 401,
            {"buyer_verify": denied.status_code, "buyer_models": denied_models.status_code, "anonymous_verify": anonymous.status_code},
        )
    finally:
        uploads = psql(
            "select s3_url from listings where seller_id in (select id from users where email in ("
            + ",".join(f"'{e}'" for e in created_emails)
            + "));"
        ).splitlines()
        psql(
            "delete from price_snapshots where listing_id in (select id from listings where seller_id in "
            "(select id from users where email in (" + ",".join(f"'{e}'" for e in created_emails) + ")));"
        )
        psql(
            "delete from listings where seller_id in (select id from users where email in ("
            + ",".join(f"'{e}'" for e in created_emails)
            + "));"
        )
        psql("delete from users where email in (" + ",".join(f"'{e}'" for e in created_emails) + ");")
        for url in uploads:
            name = url.rsplit("/", 1)[-1] if url else ""
            for candidate in (REPO / "backend" / "local_uploads").rglob(name) if name else []:
                candidate.unlink()

    cleanup = {
        "listings_after_cleanup": listing_count(),
        "users_after_cleanup": int(psql("select count(*) from users;")),
        "listings_before": baseline_listings,
        "users_before": baseline_users,
    }
    results = {
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "stack": {
            "api": "uvicorn main:app, HYPEVAULT_DEPLOYMENT_MODE=research, RESEARCH_USER_EMAILS=<one E2E account>",
            "production_api": "second uvicorn with HYPEVAULT_DEPLOYMENT_MODE=production",
            "triton": args.triton_desc,
            "database": "hypevault-postgres (local dev)",
            "frontend": "next start (production build)",
            "gpu_used": args.gpu_used,
            "readiness_at_start": {m["logical_id"]: {"ready": m["ready"], "reason": m.get("unavailable_reason")} for m in models["models"]},
        },
        "input": {"sample_id": authentic_case["sample_id"], "split": authentic_case["split"], "declared_brand": brand},
        "flows": flows,
        "cleanup": cleanup,
        "all_passed": all(f["result"] == "PASS" for f in flows.values()),
    }
    Path(args.results).write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({name: f["result"] for name, f in flows.items()}, indent=2))
    print(json.dumps(cleanup))
    return 0 if results["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
