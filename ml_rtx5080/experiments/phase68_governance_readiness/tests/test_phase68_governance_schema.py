"""Phase 68 schema/template tests using synthetic governance metadata only.

No network, images, acquisition, gate toggles, or production authorization.
"""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = PACKAGE_ROOT / "schemas" / "acquisition_decision_record.schema.json"
TEMPLATE_PATH = PACKAGE_ROOT / "templates" / "acquisition_decision_record.template.json"

# Forbidden authorization / executable toggle field names that must not appear
# on the decision record. Presence would create a false path from docs → gates.
FORBIDDEN_EXECUTABLE_KEYS = frozenset(
    {
        "ACQUISITION_AUTHORIZED",
        "acquisition_authorized",
        "INFERENCE_AUTHORIZED",
        "inference_authorized",
        "DATA_READINESS",
        "data_readiness",
        "AUTHENTICITY_MODEL_PRODUCTION_APPROVED",
        "PRODUCTION_PROMOTION",
        "PRODUCTION_PROMOTION_ALLOWED",
        "DINOv3_PRODUCTION_ALLOWED",
        "env_exports",
        "gate_toggles",
        "cli_flags_to_set",
        "authorize_acquisition",
        "flip_gate",
    }
)

ALLOWED_OUTCOMES = frozenset(
    {
        "DRAFT",
        "NO_GO",
        "CONDITIONAL_NO_GO",
        "GO_RECOMMENDED_SEPARATE_AUTHORIZATION_REQUIRED",
    }
)

SYNTH_SHA = "a" * 64
SYNTH_TS = "2026-10-14T00:00:00Z"


def _evidence(ref_id: str = "eref_SYNTH_0001") -> dict:
    return {
        "ref_id": ref_id,
        "title": "Synthetic governance evidence",
        "uri_or_path": "synthetic://phase68/not-a-real-source",
        "retrieved_at_utc": "NOT_RECORDED",
        "sha256": "NOT_RECORDED",
    }


def _completed_evidence(ref_id: str, slug: str) -> dict:
    return {
        "ref_id": ref_id,
        "title": f"Synthetic completed evidence for {slug}",
        "uri_or_path": f"synthetic://phase68/evidence/{slug}",
        "retrieved_at_utc": SYNTH_TS,
        "sha256": SYNTH_SHA,
    }


def _applicability_block() -> dict:
    return {
        "applicability": "NOT_RECORDED",
        "status": "NOT_STARTED",
        "reviewer_ref": "reviewer_role_synth_VACANT",
        "reviewed_at_utc": "NOT_RECORDED",
        "evidence_refs": [_evidence("eref_SYNTH_APP_0001")],
    }


def _go_applicability(
    reviewer_ref: str,
    evidence_slug: str,
    *,
    applicable: bool = True,
    rationale: str | None = None,
) -> dict:
    block = {
        "applicability": "APPLICABLE" if applicable else "NOT_APPLICABLE",
        "status": "COMPLETE_PASS",
        "reviewer_ref": reviewer_ref,
        "reviewed_at_utc": SYNTH_TS,
        "evidence_refs": [
            _completed_evidence(f"eref_{evidence_slug.upper()}_0001", evidence_slug)
        ],
    }
    if not applicable:
        block["not_applicable_rationale"] = (
            rationale
            or "Synthetic desk-only metadata package; no human-subjects activity asserted"
        )
    return block


def _go_reviewer(role: str, slug: str) -> dict:
    return {
        "reviewer_ref": f"reviewer_role_{slug}_assigned",
        "role": role,
        "decision": "APPROVE_FOR_RECOMMENDATION_REVIEW",
        "decided_at_utc": SYNTH_TS,
        "evidence_refs": [
            _completed_evidence(f"eref_REV_{slug.upper()}_0001", f"reviewer-{slug}")
        ],
    }


def minimal_draft() -> dict:
    """Minimal synthetic DRAFT record; no fabricated approvals."""
    return {
        "schema_version": "phase68_acquisition_decision_record_v1",
        "decision_record_id": "acqdec_SYNTH_DRAFT_0001",
        "created_at_utc": "NOT_RECORDED",
        "research_purpose": {
            "research_questions": ["Synthetic research question only"],
            "permitted_model_development_purpose": "NOT_RECORDED",
            "product_category_scope": ["NOT_RECORDED"],
            "intended_audience": "NOT_RECORDED",
            "publication_boundaries": "NOT_RECORDED",
            "explicit_out_of_scope_uses": ["Acquisition unauthorized by this record"],
        },
        "proposed_sources": [
            {
                "source_id": "source_SYNTH_0001",
                "canonical_name": "Synthetic placeholder source",
                "public_urls": ["NOT_RECORDED"],
                "intended_acquisition_method": "NOT_RECORDED",
                "scope_limit": "NOT_RECORDED",
                "terms_evidence_refs": [_evidence("eref_SYNTH_TERMS_0001")],
                "copyright_redistribution_notes": "NOT_RECORDED",
                "provider_permission_status": "NOT_RECORDED",
                "storage_transformation_publication_restrictions": "NOT_RECORDED",
                "reviewer_ref": "reviewer_role_rights_VACANT",
                "reviewed_at_utc": "NOT_RECORDED",
                "evidence_refs": [_evidence("eref_SYNTH_SRC_0001")],
                "expiration_or_review_by": "NOT_RECORDED",
            }
        ],
        "rights_and_license_review": _applicability_block(),
        "privacy_consent_ethics_review": _applicability_block(),
        "ground_truth_requirements": {
            "independent_verification_procedure_ref": "NOT_RECORDED",
            "reviewer_qualification_requirements": "NOT_RECORDED",
            "dual_review_and_adjudication_defined": False,
            "excludes_filename_directory_scraper_seller_model_synthetic_as_gt": True,
            "distinction_review_eligibility_vs_ml_admission_documented": False,
            "evidence_refs": [_evidence("eref_SYNTH_GT_0001")],
        },
        "dataset_design_and_leakage_plan": {
            "identity_scheme": "NOT_RECORDED",
            "duplicate_detection_policy": "NOT_RECORDED",
            "disjointness_rules": "NOT_RECORDED",
            "split_separation_policy_level_only": "NOT_RECORDED",
            "final_test_ood_inspection_prohibited": True,
            "missing_metadata_reporting": "NOT_RECORDED",
            "proposed_thresholds_require_human_approval": True,
            "evidence_refs": [_evidence("eref_SYNTH_LEAK_0001")],
        },
        "security_storage_retention_plan": {
            "storage_designation": "NOT_RECORDED",
            "access_control_summary": "NOT_RECORDED",
            "checksum_and_audit_requirements": "NOT_RECORDED",
            "retention_and_backup_policy": "NOT_RECORDED",
            "deletion_takedown_quarantine_procedures": "NOT_RECORDED",
            "incident_owner_ref": "owner_security_VACANT",
            "evidence_refs": [_evidence("eref_SYNTH_SEC_0001")],
        },
        "budget_and_ownership": {
            "budget_mode": "ZERO_BUDGET_DESK_ONLY",
            "accountable_owner_ref": "owner_budget_VACANT",
            "time_window": "NOT_RECORDED",
            "volume_or_rate_limits": "NOT_RECORDED",
            "stop_condition_before_expenditure": "No expenditure under Phase 68",
            "evidence_refs": [_evidence("eref_SYNTH_BUD_0001")],
        },
        "reviewers": [
            {
                "reviewer_ref": "reviewer_role_preparer_VACANT",
                "role": "PREPARER",
                "decision": "NOT_STARTED",
                "decided_at_utc": "NOT_RECORDED",
                "evidence_refs": [_evidence("eref_SYNTH_REV_0001")],
            }
        ],
        "unresolved_blockers": [
            {
                "blocker_id": "blocker_SYNTH_MISSING_EVIDENCE",
                "severity": "CRITICAL",
                "description": "Synthetic blocker: mandatory evidence incomplete",
                "remediation_owner_ref": "owner_preparer_VACANT",
                "status": "OPEN",
                "evidence_refs": [_evidence("eref_SYNTH_BLK_0001")],
            }
        ],
        "overall_outcome": "DRAFT",
        "overall_rationale": "Synthetic draft for schema tests only",
        "acquisition_authorized_by_this_record": False,
        "gates_unchanged_acknowledgment": True,
    }


def complete_go_recommendation() -> dict:
    """Fully populated synthetic GO recommendation; still non-authorizing.

    Uses only synthetic URIs and role codes. Does not assert real providers,
    licenses, consent, or institutional approvals.
    """
    return {
        "schema_version": "phase68_acquisition_decision_record_v1",
        "decision_record_id": "acqdec_SYNTH_GOREC_0001",
        "created_at_utc": SYNTH_TS,
        "updated_at_utc": SYNTH_TS,
        "research_purpose": {
            "research_questions": [
                "Synthetic: can offline metadata intake support a future GT review queue?"
            ],
            "permitted_model_development_purpose": "ML_EVALUATION_ONLY",
            "product_category_scope": ["Synthetic luxury watches — desk review only"],
            "intended_audience": "Internal research governance reviewers only",
            "publication_boundaries": "No public authenticity claims; aggregates only if separately approved",
            "explicit_out_of_scope_uses": [
                "Image acquisition, seller contact, paid APIs, production authenticity claims",
                "Treating scraper or model labels as independent ground truth",
            ],
        },
        "proposed_sources": [
            {
                "source_id": "source_SYNTH_PUBLIC_0001",
                "canonical_name": "Synthetic Public Archive Route A",
                "public_urls": ["https://example.invalid/synthetic-source-a"],
                "intended_acquisition_method": "Future manual download only after separate authorization",
                "scope_limit": "Max 0 images until separate authorization; desk metadata only now",
                "terms_evidence_refs": [
                    _completed_evidence("eref_SYNTH_TERMS_GO_0001", "terms-source-a")
                ],
                "copyright_redistribution_notes": "Synthetic notes: redistribution not assumed from public visibility",
                "provider_permission_status": "NOT_REQUIRED",
                "provider_permission_rationale": "Synthetic: public-domain claim unverified; separate auth still required",
                "storage_transformation_publication_restrictions": "Quarantine-only storage; no publication of source images",
                "reviewer_ref": "reviewer_role_rights_assigned",
                "reviewed_at_utc": SYNTH_TS,
                "evidence_refs": [
                    _completed_evidence("eref_SYNTH_SRC_GO_0001", "source-desk-review-a")
                ],
                "expiration_or_review_by": "2027-04-01T00:00:00Z",
            }
        ],
        "rights_and_license_review": _go_applicability(
            "reviewer_role_rights_assigned", "rights-license"
        ),
        "privacy_consent_ethics_review": _go_applicability(
            "reviewer_role_privacy_assigned",
            "privacy-ethics",
            applicable=False,
            rationale=(
                "Synthetic: no personal data collection asserted for desk-only metadata package"
            ),
        ),
        "faculty_or_institutional_review": _go_applicability(
            "reviewer_role_faculty_assigned", "faculty-institutional"
        ),
        "ground_truth_requirements": {
            "independent_verification_procedure_ref": "synthetic://phase68/sop/gt-procedure-v0",
            "reviewer_qualification_requirements": "Two independent qualified reviewers plus adjudicator",
            "dual_review_and_adjudication_defined": True,
            "excludes_filename_directory_scraper_seller_model_synthetic_as_gt": True,
            "distinction_review_eligibility_vs_ml_admission_documented": True,
            "evidence_refs": [_completed_evidence("eref_SYNTH_GT_GO_0001", "gt-procedure")],
        },
        "dataset_design_and_leakage_plan": {
            "identity_scheme": "product_id + source_id + image_sha256 when images exist",
            "duplicate_detection_policy": "Exact hash plus proposed near-duplicate review (thresholds unapproved)",
            "disjointness_rules": "Product-disjoint and source-disjoint splits when applicable",
            "split_separation_policy_level_only": "Calibration/val/final-test/OOD separated at policy level only",
            "final_test_ood_inspection_prohibited": True,
            "missing_metadata_reporting": "Report missing brand/seller/source fields separately from labels",
            "proposed_thresholds_require_human_approval": True,
            "evidence_refs": [
                _completed_evidence("eref_SYNTH_LEAK_GO_0001", "leakage-plan")
            ],
        },
        "security_storage_retention_plan": {
            "storage_designation": "Encrypted research quarantine volume (synthetic designation)",
            "access_control_summary": "Least-privilege role ACL; no world-readable paths",
            "checksum_and_audit_requirements": "SHA-256 on ingest; append-only access audit log",
            "retention_and_backup_policy": "Retention capped; encrypted offline backup with owner approval",
            "deletion_takedown_quarantine_procedures": "Quarantine then delete on dispute/takedown within 7 days",
            "incident_owner_ref": "owner_security_assigned",
            "evidence_refs": [
                _completed_evidence("eref_SYNTH_SEC_GO_0001", "security-retention")
            ],
        },
        "budget_and_ownership": {
            "budget_mode": "ZERO_BUDGET_DESK_ONLY",
            "accountable_owner_ref": "owner_budget_assigned",
            "time_window": "2026-10-14 through 2027-01-14 desk review only",
            "volume_or_rate_limits": "Zero image acquisition; metadata desk review only",
            "stop_condition_before_expenditure": "Any non-zero spend requires a new CEILING_DOCUMENTED review",
            "evidence_refs": [
                _completed_evidence("eref_SYNTH_BUD_GO_0001", "budget-ownership")
            ],
        },
        "reviewers": [
            _go_reviewer("PREPARER", "preparer"),
            _go_reviewer("RIGHTS_LICENSE", "rights"),
            _go_reviewer("PRIVACY_ETHICS", "privacy"),
            _go_reviewer("GROUND_TRUTH", "groundtruth"),
            _go_reviewer("DATASET_INTEGRITY", "integrity"),
            _go_reviewer("SECURITY_RETENTION", "security"),
            _go_reviewer("BUDGET_OWNER", "budget"),
            _go_reviewer("LEAD_REVIEWER", "lead"),
        ],
        "unresolved_blockers": [],
        "exceptions_requested": [],
        "expiration_or_next_review_utc": "2027-01-14T00:00:00Z",
        "overall_outcome": "GO_RECOMMENDED_SEPARATE_AUTHORIZATION_REQUIRED",
        "overall_rationale": (
            "Synthetic structural completeness for schema tests only. "
            "Separate human authorization remains required; acquisition unauthorized."
        ),
        "acquisition_authorized_by_this_record": False,
        "gates_unchanged_acknowledgment": True,
    }


def _collect_keys(obj: object, found: set[str] | None = None) -> set[str]:
    if found is None:
        found = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            found.add(key)
            _collect_keys(value, found)
    elif isinstance(obj, list):
        for item in obj:
            _collect_keys(item, found)
    return found


class Phase68GovernanceSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(cls.schema)
        cls.validator = Draft202012Validator(cls.schema, format_checker=FormatChecker())
        cls.template = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))

    def assert_valid(self, row: dict) -> None:
        errors = list(self.validator.iter_errors(row))
        self.assertEqual(errors, [], "\n".join(error.message for error in errors))

    def assert_invalid(self, row: dict) -> None:
        self.assertTrue(
            list(self.validator.iter_errors(row)),
            "expected schema validation errors",
        )

    def test_blank_template_conforms_to_schema(self) -> None:
        self.assertEqual(self.template["overall_outcome"], "DRAFT")
        self.assertIs(self.template["acquisition_authorized_by_this_record"], False)
        self.assertIs(self.template["gates_unchanged_acknowledgment"], True)
        self.assert_valid(self.template)

    def test_minimal_synthetic_draft_is_valid(self) -> None:
        self.assert_valid(minimal_draft())

    def test_complete_go_recommendation_is_valid_and_non_authorizing(self) -> None:
        row = complete_go_recommendation()
        self.assertEqual(
            row["overall_outcome"],
            "GO_RECOMMENDED_SEPARATE_AUTHORIZATION_REQUIRED",
        )
        self.assertIs(row["acquisition_authorized_by_this_record"], False)
        self.assert_valid(row)

    def test_unknown_overall_outcome_rejected(self) -> None:
        row = minimal_draft()
        row["overall_outcome"] = "ACQUISITION_AUTHORIZED"
        self.assert_invalid(row)
        row["overall_outcome"] = "GO"
        self.assert_invalid(row)
        row["overall_outcome"] = "APPROVED"
        self.assert_invalid(row)

    def test_recommendation_cannot_be_represented_as_authorization(self) -> None:
        row = complete_go_recommendation()
        self.assertIs(row["acquisition_authorized_by_this_record"], False)
        self.assert_valid(row)
        row["acquisition_authorized_by_this_record"] = True
        self.assert_invalid(row)
        self.assertNotIn("ACQUISITION_AUTHORIZED", ALLOWED_OUTCOMES)
        self.assertIn(
            "GO_RECOMMENDED_SEPARATE_AUTHORIZATION_REQUIRED", ALLOWED_OUTCOMES
        )

    def test_go_recommended_rejects_open_critical_blockers(self) -> None:
        row = complete_go_recommendation()
        row["unresolved_blockers"] = [
            {
                "blocker_id": "blocker_SYNTH_CRITICAL_OPEN",
                "severity": "CRITICAL",
                "description": "Synthetic open critical blocker",
                "remediation_owner_ref": "owner_preparer_assigned",
                "status": "OPEN",
            }
        ]
        self.assert_invalid(row)

    def test_minimal_draft_with_blockers_cleared_is_not_valid_go(self) -> None:
        """Regression: empty blockers alone must not unlock a GO recommendation."""
        row = minimal_draft()
        row["overall_outcome"] = "GO_RECOMMENDED_SEPARATE_AUTHORIZATION_REQUIRED"
        row["unresolved_blockers"] = []
        self.assert_invalid(row)

    def test_go_rejects_incomplete_rights_license_review(self) -> None:
        row = complete_go_recommendation()
        row["rights_and_license_review"] = _applicability_block()
        self.assert_invalid(row)

    def test_go_rejects_incomplete_privacy_ethics_review(self) -> None:
        row = complete_go_recommendation()
        row["privacy_consent_ethics_review"]["applicability"] = "NOT_RECORDED"
        self.assert_invalid(row)
        row = complete_go_recommendation()
        row["privacy_consent_ethics_review"]["status"] = "PENDING"
        self.assert_invalid(row)

    def test_go_rejects_missing_faculty_institutional_review(self) -> None:
        row = complete_go_recommendation()
        del row["faculty_or_institutional_review"]
        self.assert_invalid(row)
        row = complete_go_recommendation()
        row["faculty_or_institutional_review"] = _applicability_block()
        self.assert_invalid(row)

    def test_go_rejects_source_placeholders_or_missing_terms(self) -> None:
        row = complete_go_recommendation()
        row["proposed_sources"][0]["canonical_name"] = "NOT_RECORDED"
        self.assert_invalid(row)
        row = complete_go_recommendation()
        row["proposed_sources"][0]["intended_acquisition_method"] = "NOT_RECORDED"
        self.assert_invalid(row)
        row = complete_go_recommendation()
        row["proposed_sources"][0]["terms_evidence_refs"][0]["sha256"] = "NOT_RECORDED"
        self.assert_invalid(row)
        row = complete_go_recommendation()
        row["proposed_sources"][0]["provider_permission_status"] = "REQUIRED_MISSING"
        self.assert_invalid(row)

    def test_go_rejects_incomplete_ground_truth_procedure(self) -> None:
        row = complete_go_recommendation()
        row["ground_truth_requirements"][
            "independent_verification_procedure_ref"
        ] = "NOT_RECORDED"
        self.assert_invalid(row)
        row = complete_go_recommendation()
        row["ground_truth_requirements"]["dual_review_and_adjudication_defined"] = False
        self.assert_invalid(row)
        row = complete_go_recommendation()
        row["ground_truth_requirements"][
            "distinction_review_eligibility_vs_ml_admission_documented"
        ] = False
        self.assert_invalid(row)

    def test_go_rejects_incomplete_leakage_controls(self) -> None:
        row = complete_go_recommendation()
        row["dataset_design_and_leakage_plan"]["identity_scheme"] = "NOT_RECORDED"
        self.assert_invalid(row)
        row = complete_go_recommendation()
        row["dataset_design_and_leakage_plan"][
            "final_test_ood_inspection_prohibited"
        ] = False
        self.assert_invalid(row)

    def test_go_rejects_incomplete_security_retention_plan(self) -> None:
        row = complete_go_recommendation()
        row["security_storage_retention_plan"]["storage_designation"] = "NOT_RECORDED"
        self.assert_invalid(row)
        row = complete_go_recommendation()
        row["security_storage_retention_plan"][
            "incident_owner_ref"
        ] = "owner_security_VACANT"
        self.assert_invalid(row)

    def test_go_rejects_incomplete_budget_and_ownership(self) -> None:
        row = complete_go_recommendation()
        row["budget_and_ownership"]["budget_mode"] = "NOT_RECORDED"
        self.assert_invalid(row)
        row = complete_go_recommendation()
        row["budget_and_ownership"]["accountable_owner_ref"] = "owner_budget_VACANT"
        self.assert_invalid(row)
        row = complete_go_recommendation()
        row["budget_and_ownership"]["budget_mode"] = "CEILING_DOCUMENTED"
        self.assert_invalid(row)
        row["budget_and_ownership"]["budget_ceiling"] = "USD 100 proposed synthetic"
        row["budget_and_ownership"]["funding_source"] = "SYNTHETIC_UNAPPROVED_FUND"
        self.assert_valid(row)

    def test_go_rejects_preparer_only_or_not_started_reviewers(self) -> None:
        row = complete_go_recommendation()
        row["reviewers"] = [
            {
                "reviewer_ref": "reviewer_role_preparer_assigned",
                "role": "PREPARER",
                "decision": "APPROVE_FOR_RECOMMENDATION_REVIEW",
                "decided_at_utc": SYNTH_TS,
                "evidence_refs": [
                    _completed_evidence("eref_REV_ONLYPREP_0001", "only-preparer")
                ],
            }
        ]
        self.assert_invalid(row)
        row = complete_go_recommendation()
        row["reviewers"][0]["decision"] = "NOT_STARTED"
        self.assert_invalid(row)
        row = complete_go_recommendation()
        row["reviewers"] = [
            r for r in row["reviewers"] if r["role"] != "LEAD_REVIEWER"
        ]
        self.assert_invalid(row)

    def test_go_rejects_placeholder_evidence_refs(self) -> None:
        row = complete_go_recommendation()
        row["rights_and_license_review"]["evidence_refs"] = [
            _evidence("eref_SYNTH_INCOMPLETE_0001")
        ]
        self.assert_invalid(row)

    def test_no_go_requires_at_least_one_blocker(self) -> None:
        row = minimal_draft()
        row["overall_outcome"] = "NO_GO"
        row["unresolved_blockers"] = []
        self.assert_invalid(row)
        row["unresolved_blockers"] = [
            {
                "blocker_id": "blocker_SYNTH_LICENSE",
                "severity": "CRITICAL",
                "description": "License evidence missing",
                "remediation_owner_ref": "owner_rights_VACANT",
                "status": "OPEN",
            }
        ]
        self.assert_valid(row)

    def test_missing_mandatory_top_level_field_is_blocker(self) -> None:
        row = minimal_draft()
        del row["overall_rationale"]
        self.assert_invalid(row)
        row = minimal_draft()
        del row["ground_truth_requirements"]
        self.assert_invalid(row)
        row = minimal_draft()
        del row["proposed_sources"]
        self.assert_invalid(row)

    def test_not_applicable_requires_rationale_on_applicability_block(self) -> None:
        row = minimal_draft()
        block = row["privacy_consent_ethics_review"]
        block["applicability"] = "NOT_APPLICABLE"
        if "not_applicable_rationale" in block:
            del block["not_applicable_rationale"]
        self.assert_invalid(row)
        block["not_applicable_rationale"] = "Synthetic rationale: desk-only metadata review"
        self.assert_valid(row)

    def test_not_applicable_reviewer_decision_requires_rationale(self) -> None:
        row = minimal_draft()
        reviewer = row["reviewers"][0]
        reviewer["decision"] = "NOT_APPLICABLE"
        if "not_applicable_rationale" in reviewer:
            del reviewer["not_applicable_rationale"]
        self.assert_invalid(row)
        reviewer["not_applicable_rationale"] = "Role not required for this synthetic case"
        self.assert_valid(row)

    def test_provider_permission_not_required_needs_rationale(self) -> None:
        row = minimal_draft()
        source = row["proposed_sources"][0]
        source["provider_permission_status"] = "NOT_REQUIRED"
        if "provider_permission_rationale" in source:
            del source["provider_permission_rationale"]
        self.assert_invalid(row)
        source["provider_permission_rationale"] = "Synthetic: public-domain claim unverified"
        self.assert_valid(row)

    def test_reviewer_identity_outcome_and_evidence_required(self) -> None:
        row = minimal_draft()
        del row["reviewers"][0]["reviewer_ref"]
        self.assert_invalid(row)
        row = minimal_draft()
        del row["reviewers"][0]["decision"]
        self.assert_invalid(row)
        row = minimal_draft()
        del row["reviewers"][0]["evidence_refs"]
        self.assert_invalid(row)
        row = minimal_draft()
        row["reviewers"][0]["evidence_refs"] = []
        self.assert_invalid(row)

    def test_weak_labels_cannot_be_accepted_as_ground_truth(self) -> None:
        row = minimal_draft()
        row["ground_truth_requirements"][
            "excludes_filename_directory_scraper_seller_model_synthetic_as_gt"
        ] = False
        self.assert_invalid(row)

    def test_final_test_ood_inspection_must_remain_prohibited(self) -> None:
        row = minimal_draft()
        row["dataset_design_and_leakage_plan"][
            "final_test_ood_inspection_prohibited"
        ] = False
        self.assert_invalid(row)

    def test_budget_ceiling_documented_requires_funding_fields(self) -> None:
        row = minimal_draft()
        row["budget_and_ownership"]["budget_mode"] = "CEILING_DOCUMENTED"
        self.assert_invalid(row)
        row["budget_and_ownership"]["budget_ceiling"] = "USD 0 desk-only proposal"
        row["budget_and_ownership"]["funding_source"] = "NOT_APPROVED_SYNTHETIC"
        self.assert_valid(row)

    def test_gates_unchanged_acknowledgment_must_be_true(self) -> None:
        row = minimal_draft()
        row["gates_unchanged_acknowledgment"] = False
        self.assert_invalid(row)

    def test_no_executable_authorization_fields_in_schema_or_template(self) -> None:
        schema_keys = _collect_keys(self.schema)
        template_keys = _collect_keys(self.template)
        draft_keys = _collect_keys(minimal_draft())
        for forbidden in FORBIDDEN_EXECUTABLE_KEYS:
            self.assertNotIn(forbidden, schema_keys)
            self.assertNotIn(forbidden, template_keys)
            self.assertNotIn(forbidden, draft_keys)
        self.assertIn("acquisition_authorized_by_this_record", template_keys)
        self.assertIs(self.template["acquisition_authorized_by_this_record"], False)

    def test_additional_properties_rejected_on_root(self) -> None:
        row = minimal_draft()
        row["ACQUISITION_AUTHORIZED"] = True
        self.assert_invalid(row)
        row = minimal_draft()
        row["env_exports"] = {"ACQUISITION_AUTHORIZED": "true"}
        self.assert_invalid(row)

    def test_schema_outcomes_are_non_authorizing_only(self) -> None:
        outcomes = set(self.schema["properties"]["overall_outcome"]["enum"])
        self.assertEqual(outcomes, ALLOWED_OUTCOMES)
        for forbidden in (
            "ACQUISITION_AUTHORIZED",
            "AUTHORIZED",
            "APPROVED",
            "GO",
            "PASS",
        ):
            self.assertNotIn(forbidden, outcomes)

    def test_mutating_copy_of_template_still_requires_false_authorization(self) -> None:
        row = copy.deepcopy(self.template)
        row["decision_record_id"] = "acqdec_SYNTH_MUTATED_0001"
        row["overall_outcome"] = "GO_RECOMMENDED_SEPARATE_AUTHORIZATION_REQUIRED"
        row["unresolved_blockers"] = []
        row["acquisition_authorized_by_this_record"] = False
        # Template placeholders cannot satisfy GO completeness.
        self.assert_invalid(row)
        row = complete_go_recommendation()
        self.assert_valid(row)
        row["acquisition_authorized_by_this_record"] = True
        self.assert_invalid(row)


if __name__ == "__main__":
    unittest.main()
