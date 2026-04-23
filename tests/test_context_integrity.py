import unittest

from src.dossier_schema_v3 import (
    DossierPassport,
    DossierReport,
    ProductContext,
    compute_dossier_quality_v2,
)
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


if __name__ == "__main__":
    unittest.main()
