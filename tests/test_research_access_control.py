"""Research entitlement and registration-role checks through real FastAPI routing. No model load."""

from __future__ import annotations

import io
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

RESEARCHER = "researcher@example.com"
BUYER = "buyer@example.com"


def _jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), (8, 9, 10)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _client(email: str | None):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from auth.deps import get_current_user
    from inference.research_routes import router

    app = FastAPI()
    app.include_router(router, prefix="/research")
    if email is not None:
        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(email=email, role="buyer")
    return TestClient(app)


class RegistrationRoleTests(unittest.TestCase):
    def test_public_roles_register_and_other_roles_are_refused(self) -> None:
        from pydantic import ValidationError

        from auth.models import PUBLIC_REGISTRATION_ROLES, UserRegister
        from database import UserRole

        self.assertEqual(PUBLIC_REGISTRATION_ROLES, frozenset({UserRole.buyer, UserRole.seller}))
        for role in ("buyer", "seller"):
            self.assertEqual(UserRegister(email=BUYER, password="abcdef12", role=role).role.value, role)
        self.assertEqual(UserRegister(email=BUYER, password="abcdef12").role, UserRole.buyer)
        for role in ("admin", "researcher", "research", "superuser", ""):
            with self.assertRaises(ValidationError):
                UserRegister(email=BUYER, password="abcdef12", role=role)

    def test_a_role_outside_the_public_set_is_refused_even_if_the_enum_grows(self) -> None:
        from pydantic import ValidationError

        import auth.models as models
        from database import UserRole

        with patch.object(models, "PUBLIC_REGISTRATION_ROLES", frozenset({UserRole.buyer})):
            with self.assertRaises(ValidationError):
                models.UserRegister(email=BUYER, password="abcdef12", role="seller")
        routes = (REPO / "backend" / "auth" / "routes.py").read_text()
        self.assertLess(routes.index("PUBLIC_REGISTRATION_ROLES"), routes.index("role=body.role"))

    def test_research_entitlement_is_not_a_registration_field(self) -> None:
        from auth.models import UserRegister

        fields = set(UserRegister.model_fields)
        self.assertEqual(fields, {"email", "password", "role"})
        body = UserRegister.model_validate(
            {"email": BUYER, "password": "abcdef12", "research": True, "research_user": True}
        )
        self.assertFalse(hasattr(body, "research"))


class ResearchAccessTests(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("HYPEVAULT_DEPLOYMENT_MODE", None)

    def test_allowlist_parsing_is_explicit_and_empty_means_nobody(self) -> None:
        from config import research_user_allowlist

        self.assertEqual(research_user_allowlist(""), frozenset())
        self.assertEqual(research_user_allowlist(" , "), frozenset())
        self.assertEqual(
            research_user_allowlist(f" {RESEARCHER.upper()} , other@example.com "),
            frozenset({RESEARCHER, "other@example.com"}),
        )

    def test_unauthenticated_requests_are_refused(self) -> None:
        os.environ["HYPEVAULT_DEPLOYMENT_MODE"] = "research"
        with patch("config.settings.research_user_emails", RESEARCHER):
            client = _client(None)
            verify = client.post(
                "/research/verify",
                files={"image": ("a.jpg", _jpeg(), "image/jpeg")},
                data={"brand": "Patek Philippe", "logical_model": "dinov3_experimental"},
            )
            models = client.get("/research/models")
        self.assertEqual(verify.status_code, 401)
        self.assertEqual(models.status_code, 401)
        self.assertNotIn("decision", verify.json())

    def test_logged_in_user_without_entitlement_is_refused_in_research_mode(self) -> None:
        os.environ["HYPEVAULT_DEPLOYMENT_MODE"] = "research"
        infer = AsyncMock(side_effect=AssertionError("unauthorized request reached inference"))
        readiness = AsyncMock(side_effect=AssertionError("unauthorized request reached readiness"))
        for allowlist in ("", RESEARCHER):
            with (
                patch("config.settings.research_user_emails", allowlist),
                patch("inference.research_routes.infer_allowlisted_model", new=infer),
                patch("inference.research_routes.readiness_report", new=readiness),
            ):
                client = _client(BUYER)
                verify = client.post(
                    "/research/verify",
                    files={"image": ("a.jpg", _jpeg(), "image/jpeg")},
                    data={"brand": "Patek Philippe", "logical_model": "dinov3_experimental"},
                )
                models = client.get("/research/models")
            self.assertEqual(verify.status_code, 403, allowlist)
            self.assertEqual(verify.json(), {"detail": "Research access required"})
            self.assertEqual(models.status_code, 403, allowlist)
        infer.assert_not_awaited()
        readiness.assert_not_awaited()

    def test_entitled_user_reaches_the_gates_and_production_mode_still_blocks(self) -> None:
        infer = AsyncMock(side_effect=AssertionError("blocked request reached inference"))
        with (
            patch("config.settings.research_user_emails", RESEARCHER.upper()),
            patch("inference.research_routes.infer_allowlisted_model", new=infer),
        ):
            client = _client(RESEARCHER)
            for mode in ("production", "not-a-mode", None):
                if mode is None:
                    os.environ.pop("HYPEVAULT_DEPLOYMENT_MODE", None)
                else:
                    os.environ["HYPEVAULT_DEPLOYMENT_MODE"] = mode
                blocked = client.post(
                    "/research/verify",
                    files={"image": ("a.jpg", _jpeg(), "image/jpeg")},
                    data={"brand": "Patek Philippe", "logical_model": "dinov3_experimental"},
                )
                self.assertEqual(blocked.status_code, 403, mode)
                self.assertEqual(blocked.json()["status"], "POLICY_ERROR")
                self.assertIsNone(blocked.json()["decision"])

            os.environ["HYPEVAULT_DEPLOYMENT_MODE"] = "research"
            unsupported = client.post(
                "/research/verify",
                files={"image": ("a.jpg", _jpeg(), "image/jpeg")},
                data={"brand": "Rolex", "logical_model": "dinov3_experimental"},
            )
            unknown = client.post(
                "/research/verify",
                files={"image": ("a.jpg", _jpeg(), "image/jpeg")},
                data={"brand": "Patek Philippe", "logical_model": "dinov3_authenticity_candidate"},
            )
        self.assertEqual(unsupported.status_code, 422)
        self.assertIsNone(unsupported.json()["decision"])
        self.assertEqual(unknown.status_code, 422)
        self.assertIsNone(unknown.json()["decision"])
        infer.assert_not_awaited()

    def test_entitled_user_sees_per_model_readiness(self) -> None:
        report = {
            "server": {"live": True, "ready": False},
            "models": [{"logical_id": "dinov3_experimental", "ready": True}],
            "publication_decision": "BLOCKED",
        }
        with (
            patch("config.settings.research_user_emails", RESEARCHER),
            patch("inference.research_routes.readiness_report", new=AsyncMock(return_value=report)),
        ):
            response = _client(RESEARCHER).get("/research/models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), report)


if __name__ == "__main__":
    unittest.main()
