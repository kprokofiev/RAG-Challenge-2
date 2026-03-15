"""Sprint 14 acceptance unit tests."""
import json
import sys

print("=== SPRINT 14 UNIT TEST SUITE ===")
print()
errors = 0

# -- Test 1: All modules import cleanly --
print("T1: Module imports...")
try:
    from src.coverage_ledger import CoverageLedgerBuilder, _compute_source_verdict, _STATIC_VERDICTS
    from src.dossier_schema_v3 import (
        build_product_contexts, compute_dossier_quality,
        compute_dossier_quality_v2, DossierReport,
        DossierRegistration, DossierEvidence, EvidencedValue,
        DossierClinicalStudy, DossierPassport, ProductContext,
    )
    from src.dossier_report_generator import DossierReportGenerator
    print("  PASS: all modules imported")
except Exception as e:
    print(f"  FAIL: {e}")
    errors += 1
    sys.exit(1)

# -- Test 2: Static verdicts cover all declared sources --
print("T2: Static verdict coverage...")
EXPECTED_SOURCES = {
    "openfda", "dailymed", "drugsatfda", "rems", "fda_label_safety",
    "ema_smpc", "ema_pil", "ema_epar", "ctis",
    "grls", "grls_instruction", "ru_quality_letter", "eaeu_portal",
    "ctgov", "pubmed", "ru_clinical",
    "epo_ops", "epo_register", "google_patents", "patentsview",
    "lens_org", "rospatent", "eapo",
    "pubchem",
    "manufacturing_svc", "esklp", "pharmacompass",
}
missing = EXPECTED_SOURCES - set(_STATIC_VERDICTS.keys())
extra = set(_STATIC_VERDICTS.keys()) - EXPECTED_SOURCES
if missing:
    print(f"  FAIL: missing verdicts for: {missing}")
    errors += 1
elif extra:
    print(f"  WARN: extra verdicts for: {extra}")
else:
    print(f"  PASS: all {len(EXPECTED_SOURCES)} sources have static verdicts")

# -- Test 3: Verdict logic correctness --
print("T3: Verdict logic...")
t3_errors = 0
tests = [
    ("openfda", "ok", 7, ["f1"], "tier1", "KEEP"),
    ("pubmed", "ok", 0, [], "tier2", "DEMOTE"),
    ("eaeu_portal", "unsupported", 0, [], "tier1", "FIX"),
    ("rems", "not_attached", 0, [], "tier2", "PARK"),
    ("grls", "captcha", 0, ["f1"], "tier1", "FIX"),
    ("ctgov", "ok", 5, ["f1"], "tier1", "KEEP"),
    # Dynamic fallbacks
    ("unknown_src", "ok", 3, ["f1"], "tier1", "KEEP"),
    ("unknown_src", "ok", 0, ["f1"], "tier1", "CONNECT"),
    ("unknown_src", "ok", 0, [], "tier2", "DEMOTE"),
    ("unknown_src", "failed", 0, ["f1"], "tier1", "FIX"),
    ("unknown_src", "not_attached", 0, [], "tier2", "PARK"),
    ("unknown_src", "not_attached", 0, ["f1"], "tier1", "CONNECT"),
    ("unknown_src", "unsupported", 0, [], "tier1", "FIX"),
]
for src, status, fc, mhf, tier, expected in tests:
    result = _compute_source_verdict(src, status, fc, mhf, tier)
    if result != expected:
        print(f"  FAIL: {src}/{status}/{fc}/{tier} -> {result}, expected {expected}")
        t3_errors += 1
if t3_errors == 0:
    print(f"  PASS: {len(tests)} verdict logic tests passed")
else:
    errors += t3_errors

# -- Test 4: CoverageLedgerBuilder output structure --
print("T4: CoverageLedger output structure...")
builder = CoverageLedgerBuilder(use_case="ra_regulatory_screening")
ledger = builder.build(db_documents=[], dossier_report=None)

assert "verdict_summary" in ledger["totals"], "verdict_summary missing"
assert "family_health" in ledger, "family_health missing"
for sb in ledger["source_breakdown"]:
    assert "verdict" in sb, f"verdict missing from {sb['source_id']}"

vs = ledger["totals"]["verdict_summary"]
assert vs.get("KEEP", 0) == 7, f"KEEP count {vs.get('KEEP')} != 7"
assert vs.get("FIX", 0) == 6, f"FIX count {vs.get('FIX')} != 6"
assert vs.get("CONNECT", 0) == 5, f"CONNECT count {vs.get('CONNECT')} != 5"
assert vs.get("DEMOTE", 0) == 2, f"DEMOTE count {vs.get('DEMOTE')} != 2"
assert vs.get("PARK", 0) == 7, f"PARK count {vs.get('PARK')} != 7"

fh = ledger["family_health"]
assert len(fh) > 10, f"Only {len(fh)} families, expected >10"
for name, fam in fh.items():
    assert "health" in fam, f"health missing from family {name}"
    assert fam["health"] in ("GREEN", "YELLOW", "RED", "GREY"), f"bad health: {fam['health']}"

for lim in ledger["limitations"]:
    v = lim.get("verdict")
    if v == "PARK":
        assert lim["severity"] != "high", f"PARK source {lim.get('source_id')} severity=high"

print("  PASS: ledger structure verified")

# -- Test 5: Context hardening - suppression --
print("T5: Context suppression...")
reg = DossierRegistration(
    region="US",
    status=EvidencedValue(value="Approved", evidence_refs=["ev1"]),
    mah=EvidencedValue(value="Corp", evidence_refs=["ev1"]),
    identifiers=[EvidencedValue(value="NDA1", evidence_refs=["ev1"])],
    forms_strengths=[EvidencedValue(value="Tablet 10mg", evidence_refs=["ev1"])],
    primary_docs=[], evidence_refs=["ev1"],
)
evidence = [
    DossierEvidence(evidence_id="ev1", doc_id="d1", title="FDA", doc_kind="label",
                    snippet="injectable solution"),
    DossierEvidence(evidence_id="ev2", doc_id="d2", title="PC", doc_kind="pubchem",
                    snippet="oral administration tablet"),
]
ctxs, suppressed = build_product_contexts([reg], evidence)
has_weak_oral = any(c.context_strength == "weak_signal" and "oral" in (c.route or "") for c in ctxs)
assert not has_weak_oral, "PubChem-only oral weak_signal not suppressed"
assert len(suppressed) >= 1, "No suppressed contexts reported"
assert suppressed[0]["reason"] == "single_source_weak_signal"
print(f"  PASS: {len(suppressed)} context(s) suppressed correctly")

# -- Test 6: Corroborated weak_signal passes --
print("T6: Corroborated weak_signal...")
evidence2 = [
    DossierEvidence(evidence_id="ev1", doc_id="d1", title="FDA", doc_kind="label",
                    snippet="injectable solution"),
    DossierEvidence(evidence_id="ev2", doc_id="d2", title="PC", doc_kind="pubchem",
                    snippet="oral administration tablet"),
    DossierEvidence(evidence_id="ev3", doc_id="d3", title="Pub", doc_kind="publication",
                    snippet="oral dosage form tablet"),
]
ctxs2, suppressed2 = build_product_contexts([reg], evidence2)
has_oral = any("oral" in (c.route or "") for c in ctxs2)
assert has_oral, "Corroborated oral context should pass through"
assert len(suppressed2) == 0, f"Corroborated context was suppressed: {suppressed2}"
print("  PASS: corroborated weak_signal passes through")

# -- Test 7: build_product_contexts return type --
print("T7: Return type...")
result = build_product_contexts([], [])
assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
assert len(result) == 2, f"Expected 2-tuple, got {len(result)}"
ctxs_, supp_ = result
assert isinstance(ctxs_, list), "contexts not a list"
assert isinstance(supp_, list), "suppressed not a list"
print("  PASS: returns (List, List) tuple")

# -- Test 8: Readiness excludes PARK sources --
print("T8: Readiness PARK exclusion...")
mock_breakdown = [
    {"source_id": "a", "verdict": "PARK", "status": "not_attached"},
    {"source_id": "b", "verdict": "PARK", "status": "not_attached"},
    {"source_id": "c", "verdict": "PARK", "status": "not_attached"},
    {"source_id": "d", "verdict": "KEEP", "status": "ok"},
]
r = builder._section_readiness(
    indexed=1, declared=4, fields_filled=3, fields_expected=4,
    critical_unknowns=0, source_breakdown=mock_breakdown,
)
assert r == "ready", f"Expected ready, got {r} (PARK exclusion not working)"
print("  PASS: PARK sources excluded from readiness denominator")

# -- Test 9: Source contracts updated --
print("T9: Source contracts...")
with open("config/source_contracts.json", "r") as f:
    contracts = json.load(f)
assert contracts["_meta"]["version"] == "2.0", "contracts version not 2.0"
for c in contracts["contracts"]:
    assert "source_role" in c, f"source_role missing from {c['source_id']}"
    assert "verdict" in c, f"verdict missing from {c['source_id']}"
    assert c["source_role"] in ("primary", "enrichment", "supplementary"), f"bad role: {c['source_role']}"
    assert c["verdict"] in ("KEEP", "CONNECT", "FIX", "DEMOTE", "PARK"), f"bad verdict: {c['verdict']}"
pubmed_c = next(c for c in contracts["contracts"] if c["source_id"] == "pubmed")
assert pubmed_c["tier"] == "tier2", "pubmed not demoted to tier2"
assert pubmed_c["source_role"] == "enrichment", "pubmed not marked enrichment"
lens_c = next(c for c in contracts["contracts"] if c["source_id"] == "lens_org")
assert lens_c["tier"] == "tier2", "lens_org not demoted to tier2"
print(f"  PASS: {len(contracts['contracts'])} contracts validated")

# -- Summary --
print()
if errors == 0:
    print("ALL TESTS PASSED (9/9)")
else:
    print(f"FAILED: {errors} test(s)")
    sys.exit(1)
