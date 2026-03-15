"""Sprint 15 acceptance unit tests."""
import json
import os
import sys
import tempfile
from pathlib import Path

print("=== SPRINT 15 UNIT TEST SUITE ===")
print()
errors = 0

# -- Test 1: All modules import cleanly --
print("T1: Module imports...")
try:
    from src.dossier_schema_v3 import (
        build_product_contexts, DossierReport,
        DossierRegistration, DossierEvidence, EvidencedValue,
        DossierPassport, ProductContext,
    )
    from src.dossier_report_generator import DossierReportGenerator
    print("  PASS: all modules imported")
except Exception as e:
    print(f"  FAIL: {e}")
    errors += 1
    sys.exit(1)

# -- Test 2: Deterministic fda_approval_date — finds exact marker --
print("T2: FDA date extractor — exact marker...")
with tempfile.TemporaryDirectory() as tmpdir:
    docs_dir = Path(tmpdir)
    vectors_dir = Path(tmpdir) / "vectors"
    vectors_dir.mkdir()

    # Create a mock chunk JSON with the processor.ts marker
    mock_doc = {
        "metainfo": {
            "doc_id": "test-label-001",
            "doc_kind": "label",
            "title": "FDA Label for Ibuprofen",
            "tenant_id": "demo",
            "case_id": "test-case-001",
        },
        "content": {
            "chunks": [
                {"text": "Some unrelated text about dosage and safety."},
                {"text": "FDA Approval Date (Drugs@FDA): 20001218\nThis drug was approved...", "page": 1},
                {"text": "More text about pharmacokinetics."},
            ]
        }
    }
    with open(docs_dir / "test-label-001.json", "w") as f:
        json.dump(mock_doc, f)

    gen = DossierReportGenerator.__new__(DossierReportGenerator)
    gen.documents_dir = docs_dir
    gen.tenant_id = "demo"
    gen.case_id = "test-case-001"
    gen._evidence_registry = {}

    result = gen._extract_fda_approval_date_deterministic()
    assert result is not None, "Expected non-None result"
    assert result.value == "2000-12-18", f"Expected 2000-12-18, got {result.value}"
    assert len(result.evidence_refs) == 1, f"Expected 1 evidence ref, got {len(result.evidence_refs)}"
    assert result.evidence_refs[0] in gen._evidence_registry, "Evidence not registered"
    print(f"  PASS: found {result.value}")

# -- Test 3: Deterministic fda_approval_date — no match --
print("T3: FDA date extractor — no match...")
with tempfile.TemporaryDirectory() as tmpdir:
    docs_dir = Path(tmpdir)

    mock_doc = {
        "metainfo": {
            "doc_id": "test-label-002",
            "doc_kind": "label",
            "tenant_id": "demo",
            "case_id": "test-case-002",
        },
        "content": {
            "chunks": [
                {"text": "This label contains no approval date information."},
                {"text": "Dosage: 200mg three times daily. Side effects include nausea."},
            ]
        }
    }
    with open(docs_dir / "test-label-002.json", "w") as f:
        json.dump(mock_doc, f)

    gen2 = DossierReportGenerator.__new__(DossierReportGenerator)
    gen2.documents_dir = docs_dir
    gen2.tenant_id = "demo"
    gen2.case_id = "test-case-002"
    gen2._evidence_registry = {}

    result2 = gen2._extract_fda_approval_date_deterministic()
    assert result2 is None, f"Expected None, got {result2}"
    print("  PASS: correctly returned None")

# -- Test 4: Deterministic fda_approval_date — generic approval date pattern --
print("T4: FDA date extractor — generic pattern...")
with tempfile.TemporaryDirectory() as tmpdir:
    docs_dir = Path(tmpdir)

    mock_doc = {
        "metainfo": {
            "doc_id": "test-us-fda-003",
            "doc_kind": "us_fda",
            "tenant_id": "demo",
            "case_id": "test-case-003",
        },
        "content": {
            "chunks": [
                {"text": "Original Approval Date: 1984-01-15. Drug approved for analgesic use.", "page": 2},
            ]
        }
    }
    with open(docs_dir / "test-us-fda-003.json", "w") as f:
        json.dump(mock_doc, f)

    gen3 = DossierReportGenerator.__new__(DossierReportGenerator)
    gen3.documents_dir = docs_dir
    gen3.tenant_id = "demo"
    gen3.case_id = "test-case-003"
    gen3._evidence_registry = {}

    result3 = gen3._extract_fda_approval_date_deterministic()
    assert result3 is not None, "Expected non-None result"
    assert result3.value == "1984-01-15", f"Expected 1984-01-15, got {result3.value}"
    print(f"  PASS: found {result3.value}")

# -- Test 5: Deterministic fda_approval_date — skips non-label doc_kinds --
print("T5: FDA date extractor — skips pubchem doc_kind...")
with tempfile.TemporaryDirectory() as tmpdir:
    docs_dir = Path(tmpdir)

    mock_doc = {
        "metainfo": {
            "doc_id": "test-pubchem-004",
            "doc_kind": "pubchem",
            "tenant_id": "demo",
            "case_id": "test-case-004",
        },
        "content": {
            "chunks": [
                {"text": "FDA Approval Date (Drugs@FDA): 19990301", "page": 1},
            ]
        }
    }
    with open(docs_dir / "test-pubchem-004.json", "w") as f:
        json.dump(mock_doc, f)

    gen4 = DossierReportGenerator.__new__(DossierReportGenerator)
    gen4.documents_dir = docs_dir
    gen4.tenant_id = "demo"
    gen4.case_id = "test-case-004"
    gen4._evidence_registry = {}

    result4 = gen4._extract_fda_approval_date_deterministic()
    assert result4 is None, f"Should skip pubchem docs, got {result4}"
    print("  PASS: pubchem doc correctly skipped")

# -- Test 6: Deterministic fda_approval_date — year-only fallback --
print("T6: FDA date extractor — year-only fallback...")
with tempfile.TemporaryDirectory() as tmpdir:
    docs_dir = Path(tmpdir)

    mock_doc = {
        "metainfo": {
            "doc_id": "test-label-005",
            "doc_kind": "label",
            "tenant_id": "demo",
            "case_id": "test-case-005",
        },
        "content": {
            "chunks": [
                {"text": "This medication was first approved: 1974 by the FDA.", "page": 1},
            ]
        }
    }
    with open(docs_dir / "test-label-005.json", "w") as f:
        json.dump(mock_doc, f)

    gen5 = DossierReportGenerator.__new__(DossierReportGenerator)
    gen5.documents_dir = docs_dir
    gen5.tenant_id = "demo"
    gen5.case_id = "test-case-005"
    gen5._evidence_registry = {}

    result5 = gen5._extract_fda_approval_date_deterministic()
    assert result5 is not None, "Expected non-None for year-only"
    assert result5.value == "1974-01-01", f"Expected 1974-01-01, got {result5.value}"
    print(f"  PASS: year-only fallback -> {result5.value}")

# -- Test 7: Corroboration gate — single doc_id → suppressed --
print("T7: Corroboration gate — single doc suppressed...")
reg = DossierRegistration(
    region="US",
    status=EvidencedValue(value="Approved", evidence_refs=["ev1"]),
    mah=EvidencedValue(value="Corp", evidence_refs=["ev1"]),
    identifiers=[EvidencedValue(value="NDA1", evidence_refs=["ev1"])],
    forms_strengths=[EvidencedValue(value="Tablet 10mg", evidence_refs=["ev1"])],
    primary_docs=[], evidence_refs=["ev1"],
)
evidence_single = [
    DossierEvidence(evidence_id="ev1", doc_id="d1", title="FDA", doc_kind="label",
                    snippet="injectable solution for IV use"),
    DossierEvidence(evidence_id="ev2", doc_id="d2", title="PC", doc_kind="pubchem",
                    snippet="oral administration tablet form"),
]
ctxs, suppressed = build_product_contexts([reg], evidence_single)
oral_weak = [c for c in ctxs if c.context_strength == "weak_signal" and "oral" in (c.route or "")]
assert len(oral_weak) == 0, f"Single-doc oral weak_signal should be suppressed, found {len(oral_weak)}"
assert any(s.get("reason") == "single_source_weak_signal" for s in suppressed), "Missing suppression record"
print(f"  PASS: single-doc weak_signal suppressed ({len(suppressed)} suppressions)")

# -- Test 8: Corroboration gate — two doc_ids → passes --
print("T8: Corroboration gate — two docs same route/form pass...")
evidence_dual = [
    DossierEvidence(evidence_id="ev1", doc_id="d1", title="FDA", doc_kind="label",
                    snippet="injectable solution for IV use"),
    DossierEvidence(evidence_id="ev2", doc_id="d2", title="PC", doc_kind="pubchem",
                    snippet="oral administration tablet form"),
    DossierEvidence(evidence_id="ev3", doc_id="d3", title="Pub", doc_kind="publication",
                    snippet="oral tablet administration 200mg twice daily"),
]
ctxs2, suppressed2 = build_product_contexts([reg], evidence_dual)
oral_contexts = [c for c in ctxs2 if "oral" in (c.route or "")]
assert len(oral_contexts) >= 1, f"Corroborated oral context should pass through, got {[c.route for c in ctxs2]}"
# The oral|tablet key should have 2 doc_ids (d2 and d3)
oral_suppressed = [s for s in suppressed2 if s.get("route_family") == "oral"]
assert len(oral_suppressed) == 0, f"Oral should not be suppressed: {oral_suppressed}"
print(f"  PASS: corroborated oral context passes ({len(oral_contexts)} oral ctx)")

# -- Test 9: Word-boundary matching — short keywords don't false-positive --
print("T9: Word-boundary matching — no false positives for short keywords...")
from src.dossier_schema_v3 import _normalize_route_family, _normalize_form_family

# "po" inside "compound" should NOT match oral
assert _normalize_route_family("PubChem Compound CID 3672") is None, \
    "Compound should not match oral via 'po'"
# "im" inside "improved" should NOT match injectable
assert _normalize_route_family("improved patient outcomes") is None, \
    "improved should not match injectable via 'im'"
# standalone "po" SHOULD match oral
assert _normalize_route_family("given po twice daily") == "oral", \
    "standalone 'po' should match oral"
# standalone "iv" SHOULD match injectable
assert _normalize_route_family("given iv daily") == "injectable", \
    "standalone 'iv' should match injectable"
# "gel" inside "gelatin" should NOT match cream_ointment
assert _normalize_form_family("gelatin encapsulation") is None, \
    "gelatin should not match cream_ointment via 'gel'"
# standalone "gel" SHOULD match
assert _normalize_form_family("topical gel 1%") == "cream_ointment", \
    "standalone 'gel' should match cream_ointment"
# "tab" inside "table" should NOT match tablet
assert _normalize_form_family("data table summary") is None, \
    "table should not match tablet via 'tab'"
print("  PASS: all word-boundary tests passed")

# -- Test 10: PubChem compound snippet → no false route/form signal --
print("T10: PubChem compound snippet — no false route signal...")
pubchem_snippet = ("PubChem Compound: ibuprofen\nCID: 3672\n"
                   "Molecular Formula: C13H18O2\nMolecular Weight: 206.28")
assert _normalize_route_family(pubchem_snippet) is None, \
    f"PubChem snippet should not match any route, got {_normalize_route_family(pubchem_snippet)}"
assert _normalize_form_family(pubchem_snippet) is None, \
    f"PubChem snippet should not match any form, got {_normalize_form_family(pubchem_snippet)}"
print("  PASS: PubChem snippet produces no false signals")

# -- Summary --
print()
if errors == 0:
    print(f"ALL TESTS PASSED (10/10)")
else:
    print(f"FAILED: {errors} test(s)")
    sys.exit(1)
