import unittest
import tempfile
from pathlib import Path


class _StubCandidate:
    def __init__(self, evidence_id, doc_id, page, snippet):
        self.evidence_id = evidence_id
        self.doc_id = doc_id
        self.page = page
        self.snippet = snippet


class CaseViewV2GeneratorUnitTests(unittest.TestCase):
    def test_citations_from_evidence_ids_includes_source_url(self):
        from src.case_view_v2_generator import CaseViewV2Generator

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gen = CaseViewV2Generator(
                documents_dir=root / "docs",
                vector_db_dir=root / "vdb",
                tenant_id="demo",
                case_id="case1",
            )
            cand_map = {
                "ev1": _StubCandidate("ev1", "doc1", 3, "snippet1"),
                "ev2": _StubCandidate("ev2", "doc1", 3, "snippet1"),  # duplicate snippet/page/doc
                "ev3": _StubCandidate("ev3", "doc2", 10, "snippet2"),
            }
            doc_map = {
                "doc1": {"doc_id": "doc1", "source_url": "https://example.com/1"},
                "doc2": {"doc_id": "doc2"},
            }
            citations = gen._citations_from_evidence_ids(["ev1", "ev2", "ev3", "missing"], cand_map, doc_map)
            self.assertEqual(len(citations), 2)  # dedup
            self.assertEqual(citations[0]["doc_id"], "doc1")
            self.assertEqual(citations[0]["page"], 3)
            self.assertIn("source_url", citations[0])
            self.assertEqual(citations[0]["source_url"], "https://example.com/1")

    def test_merge_extracted_trials_bucket_and_phase(self):
        from src.case_view_v2_generator import CaseViewV2Generator

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gen = CaseViewV2Generator(
                documents_dir=root / "docs",
                vector_db_dir=root / "vdb",
                tenant_id="demo",
                case_id="case1",
            )
            clinical = {
                "global": gen._empty_phase_map(),
                "ru": gen._empty_phase_map(),
                "ongoing": gen._empty_phase_map(),
                "pubmed": {"comparative": [], "abstracts": [], "real_world": [], "combination": []},
            }
            extracted = {
                "trials": [
                    {
                        "trial_id": "NCT00000001",
                        "title": "Trial 1",
                        "phase": "Phase 3",
                        "study_type": "randomized",
                        "countries": ["US"],
                        "enrollment": "100",
                        "comparator": "placebo",
                        "regimen": "drug vs placebo",
                        "status": "Completed",
                        "evidence_ids": ["ev1"],
                    },
                    {
                        "trial_id": "NCT00000002",
                        "title": "Trial 2",
                        "phase": "Phase 2",
                        "study_type": "observational",
                        "countries": ["Russia"],
                        "status": "Recruiting",
                        "evidence_ids": ["ev2"],
                    },
                ]
            }
            cand_map = {
                "ev1": _StubCandidate("ev1", "doc1", 1, "s1"),
                "ev2": _StubCandidate("ev2", "doc2", 2, "s2"),
            }
            doc_map = {"doc1": {"doc_id": "doc1"}, "doc2": {"doc_id": "doc2"}}
            unknowns = []
            ok = gen._merge_extracted_trials_into_clinical(clinical, extracted, cand_map, doc_map, unknowns)
            self.assertTrue(ok)
            # Trial 1 -> global phase3
            self.assertEqual(len(clinical["global"]["phase3"]), 1)
            # Trial 2 -> ongoing (status recruiting) phase2
            self.assertEqual(len(clinical["ongoing"]["phase2"]), 1)

    def test_normalize_coverage_types(self):
        from src.case_view_v2_generator import CaseViewV2Generator

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gen = CaseViewV2Generator(documents_dir=root / "docs", vector_db_dir=root / "vdb")
            self.assertEqual(gen._normalize_coverage_types("composition; treatment"), ["composition", "treatment"])
            self.assertEqual(gen._normalize_coverage_types(["состав", "синтез"]), ["composition", "synthesis"])
            self.assertEqual(gen._normalize_coverage_types("способ лечения"), ["treatment"])

    def test_build_synthesis_fact_uses_step_citations(self):
        from src.case_view_v2_generator import CaseViewV2Generator

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gen = CaseViewV2Generator(documents_dir=root / "docs", vector_db_dir=root / "vdb")
            synthesis = {
                "synthesis_route": {
                    "steps": [
                        {"text": "Step 1", "citations": [{"doc_id": "d1", "page": 1, "snippet": "s"}]},
                    ]
                }
            }
            fact = gen._build_synthesis_fact(synthesis)
            self.assertIsNotNone(fact)
            self.assertTrue(fact.get("citations"))


if __name__ == "__main__":
    unittest.main()
