import unittest

from src.dossier_schema_v3 import (
    DossierEvidence,
    DossierPassport,
    DossierReport,
    ProductContext,
    RunManifest,
    _normalize_route_family,
    compute_dossier_quality_v2,
    sync_run_manifest_counts,
)
from src.registration_truth import VERDICT_CONFIRMED, infer_registration_verdict
from src.scope_resolver import ScopeResolver


def _ctx(region: str) -> ProductContext:
    return ProductContext(
        context_id=f"ctx-{region.lower()}",
        label=f"{region} - oral",
        region=region,
        route="oral",
        dosage_forms=["tablet"],
        strengths=["5 mg"],
        mah=f"{region} MAH",
        identifiers=[f"{region}-001"],
        context_strength="registration_confirmed",
        context_origin=f"registration: {region}",
    )


class ContextIntegrityTests(unittest.TestCase):
    def test_multi_regional_context_is_green_when_route_converged(self):
        report = DossierReport(
            report_id="rep-1",
            case_id="case-1",
            run_id="run-1",
            generated_at="2026-04-23T00:00:00Z",
            passport=DossierPassport(
                inn="apixaban",
                passport_scope="multi_regional_context",
                passport_notice="4 regional product contexts detected.",
            ),
            product_contexts=[_ctx("US"), _ctx("EU"), _ctx("RU"), _ctx("EAEU")],
        )

        quality = compute_dossier_quality_v2(report)

        self.assertEqual(quality.decision_readiness["context_integrity"], "GREEN")
        self.assertTrue(
            any("kept separate by region/MAH/registration identity" in note for note in quality.notes)
        )

    def test_scope_resolver_treats_multi_regional_context_as_converged(self):
        class RoutedQuestion:
            question_type = "clinical_evidence"
            required_jurisdictions = []

        dossier = {
            "passport": {"passport_scope": "multi_regional_context"},
            "product_contexts": [
                _ctx("US").model_dump(),
                _ctx("EU").model_dump(),
                _ctx("RU").model_dump(),
                _ctx("EAEU").model_dump(),
            ],
            "registrations": [],
        }

        scope = ScopeResolver().resolve(RoutedQuestion(), dossier)

        self.assertEqual(scope.entity_mode, "inn")
        self.assertFalse(scope.scope_warnings)
        self.assertIn("route-converged confirmed contexts", scope.reason)

    def test_multi_regional_context_stays_green_with_same_region_evidence_variant(self):
        report = DossierReport(
            report_id="rep-2",
            case_id="case-2",
            run_id="run-2",
            generated_at="2026-04-23T00:00:00Z",
            passport=DossierPassport(
                inn="apixaban",
                passport_scope="multi_regional_context",
                passport_notice="Route-converged regional contexts detected.",
            ),
            product_contexts=[
                _ctx("US"),
                _ctx("EU"),
                _ctx("RU"),
                _ctx("EAEU"),
                ProductContext(
                    context_id="ctx-us-capsule",
                    label="US - capsule (oral) - [evidence_supported]",
                    region="US",
                    route="oral",
                    dosage_forms=["capsule"],
                    strengths=[],
                    mah=None,
                    identifiers=[],
                    context_strength="evidence_supported",
                    context_origin="evidence: us_fda snippet",
                ),
            ],
        )

        quality = compute_dossier_quality_v2(report)

        self.assertEqual(quality.decision_readiness["context_integrity"], "GREEN")

    def test_pulmonary_embolism_does_not_map_to_inhalation_route(self):
        self.assertIsNone(_normalize_route_family("treatment of pulmonary embolism"))

    def test_approval_letter_status_counts_as_confirmed_registration(self):
        verdict = infer_registration_verdict(
            status="FDA NDA approval letter (approval letter content present)",
            mah="Bristol-Myers Squibb",
            identifiers=["NDA 220073"],
            forms_strengths=["Eliquis 5 mg tablets"],
        )

        self.assertEqual(verdict, VERDICT_CONFIRMED)

    def test_run_manifest_counts_sync_from_coverage_ledger(self):
        report = DossierReport(
            report_id="rep-1",
            case_id="case-1",
            run_id="run-1",
            generated_at="2026-04-23T00:00:00Z",
            passport=DossierPassport(inn="apixaban"),
            evidence_registry=[
                DossierEvidence(evidence_id="ev-1", doc_id="doc-1", snippet="one"),
                DossierEvidence(evidence_id="ev-2", doc_id="doc-2", snippet="two"),
            ],
            coverage_ledger={
                "totals": {
                    "attached_docs": 803,
                    "indexed_docs": 787,
                    "failed_docs": 2,
                }
            },
            run_manifest=RunManifest(
                run_id="run-1",
                report_id="rep-1",
                case_id="case-1",
                docs_attached=0,
                docs_indexed=0,
                docs_failed=0,
            ),
        )

        sync_run_manifest_counts(report)

        self.assertEqual(report.run_manifest.docs_attached, 803)
        self.assertEqual(report.run_manifest.docs_indexed, 787)
        self.assertEqual(report.run_manifest.docs_failed, 2)


if __name__ == "__main__":
    unittest.main()
