import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("STORAGE_ENDPOINT_URL", "http://localhost:9000")
os.environ.setdefault("STORAGE_ACCESS_KEY", "test-access")
os.environ.setdefault("STORAGE_SECRET_KEY", "test-secret")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

from src.dossier_report_generator import DossierReportGenerator


def _write_doc(path: Path, *, doc_id: str, doc_kind: str, title: str, case_id: str, text: str) -> None:
    payload = {
        "metainfo": {
            "doc_id": doc_id,
            "doc_kind": doc_kind,
            "title": title,
            "case_id": case_id,
            "tenant_id": None,
            "source_url": f"byo://{title}",
        },
        "content": {
            "chunks": [
                {
                    "text": text,
                    "page": 1,
                    "type": "content",
                }
            ]
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class DossierEaeuEnrichmentTests(unittest.TestCase):
    def _build_generator(self) -> DossierReportGenerator:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        docs_dir = root / "docs"
        original_dir = root / "original_documents"
        vdb_dir = root / "vdb"
        docs_dir.mkdir()
        original_dir.mkdir()
        vdb_dir.mkdir()

        case_id = "case-1"

        _write_doc(
            docs_dir / "doc-eaeu-structured.json",
            doc_id="doc-eaeu-structured",
            doc_kind="eaeu_document",
            title="EAEU SPD registration: apixaban",
            case_id=case_id,
            text="Status: Authorised\nValid To:\nMAH (Holder): СООО \"Лекфарм\"",
        )
        structured_original = {
            "items": [
                {
                    "reg_no": "ЛП-№(007734)-(РГ-RU)",
                    "status": "Authorised",
                    "holder": "СООО \"Лекфарм\"",
                    "authorized_presentations": [
                        {
                            "dosage_form": "таблетки, покрытые пленочной оболочкой",
                            "strength": "5 mg",
                            "route": "oral",
                        }
                    ],
                    "valid_to": "",
                }
            ]
        }
        (original_dir / "doc-eaeu-structured__payload.json").write_text(
            json.dumps(structured_original, ensure_ascii=False),
            encoding="utf-8",
        )

        export_text = """
APX export summary for EAEU path
Source: export (1).xlsx

## Record 1
Торговое наименование: Апиксабан
Держатель РУ: ЛЕКФАРМ
Номер РУ: ЛП- N(007734)-(РГ-RU)
Регистрация: 19.11.2024
Окончание: 19.11.2029
ЖНВЛП: Да
Статус РУ: Действует
Лек. форма ГРЛС: таблетки, покрытые пленочной оболочкой
Дозировка ГРЛС: 2.5 мг; 5 мг
""".strip()
        _write_doc(
            docs_dir / "doc-eaeu-export.json",
            doc_id="doc-eaeu-export",
            doc_kind="eaeu_document",
            title="apixaban_eaeu_registrations_export_summary.txt",
            case_id=case_id,
            text=export_text,
        )

        esklp_text = """
APX ESKLP summary
Source: esklp_20260409_excel_00001.zip
Detected hits: 4

## Hit 1
Row: Апиксабан | ЛП-№(007734)-(РГ-RU) | 21.20.10.131-000020-1-00058-0000000000000 | АПИКСАБАН | ТАБЛЕТКИ, ПОКРЫТЫЕ ОБОЛОЧКОЙ | 2.5 | мг | 161 | мг | шт.

## Hit 2
Row: Апиксабан | ЛП-№(007734)-(РГ-RU) | 21.20.10.131-000020-1-00066-0000000000000 | АПИКСАБАН | ТАБЛЕТКИ, ПОКРЫТЫЕ ОБОЛОЧКОЙ | 5.0 | мг | 161 | мг | шт.
""".strip()
        _write_doc(
            docs_dir / "doc-eaeu-esklp.json",
            doc_id="doc-eaeu-esklp",
            doc_kind="eaeu_document",
            title="apixaban_esklp_summary_eaeu.txt",
            case_id=case_id,
            text=esklp_text,
        )

        return DossierReportGenerator(
            vector_db_dir=vdb_dir,
            documents_dir=docs_dir,
            inn="apixaban",
            case_id=case_id,
        )

    def tearDown(self) -> None:
        if hasattr(self, "tempdir"):
            self.tempdir.cleanup()

    def test_structured_eaeu_registration_is_enriched_from_text_export(self):
        gen = self._build_generator()

        registrations = gen._extract_eaeu_registrations_from_original_json()

        self.assertEqual(len(registrations), 1)
        reg = registrations[0]
        self.assertEqual(reg.validity_type, "date_present")
        self.assertIsNotNone(reg.valid_to)
        self.assertEqual(reg.valid_to.value, "2029-11-19")
        self.assertTrue(reg.validity_evidence_refs)
        self.assertIn("2029-11-19", reg.valid_to.value)

    def test_commercial_signals_include_eaeu_derived_identity_mapping(self):
        gen = self._build_generator()

        signals = gen._generate_commercial_signals([])
        by_key = {(item.region, item.category): item for item in signals}

        self.assertIn(("EAEU", "registration_footprint"), by_key)
        self.assertIn(("EAEU", "formulary_presence"), by_key)
        self.assertEqual(by_key[("EAEU", "registration_footprint")].verdict, "confirmed")
        self.assertEqual(by_key[("EAEU", "formulary_presence")].verdict, "partial")
        self.assertIn("EAEU-style registration", by_key[("EAEU", "formulary_presence")].summary.value)

    def test_commercial_signals_include_reimbursement_support_docs(self):
        gen = self._build_generator()
        _write_doc(
            gen.documents_dir / "doc-pricing.json",
            doc_id="doc-pricing",
            doc_kind="pricing",
            title="RU Minzdrav public price limits",
            case_id="case-1",
            text=(
                "REIMBURSEMENT_CHECK | source=ru_minzdrav_public_price_limits | jurisdiction=RU | "
                "check_class=jnvlp_price_limit_row | status=listed_active | rows_found=3 | "
                "registration_id=ЛП-777"
            ),
        )
        gen._doc_metainfo_index_cache = None

        signals = gen._generate_commercial_signals([])
        by_key = {(item.region, item.category): item for item in signals}

        self.assertIn(("RU", "jnvlp_price_limit"), by_key)
        signal = by_key[("RU", "jnvlp_price_limit")]
        self.assertEqual(signal.verdict, "confirmed")
        self.assertIn("listed_active", signal.summary.value)
        self.assertTrue(signal.evidence_refs)

    def test_commercial_signals_include_product_identity_linkage_docs(self):
        gen = self._build_generator()
        _write_doc(
            gen.documents_dir / "doc-product-bridge.json",
            doc_id="doc-product-bridge",
            doc_kind="product_identity_bridge",
            title="EAEU product identity bridge",
            case_id="case-1",
            text=(
                "COMMERCIAL_SIGNAL_LINKAGE | source=eaeu_product_identity_bridge | jurisdiction=EAEU | "
                "registration_id=ЛП-777 | linked_signal=ru_minzdrav_public_price_limits | match_level=exact"
            ),
        )
        gen._doc_metainfo_index_cache = None

        signals = gen._generate_commercial_signals([])
        by_key = {(item.region, item.category): item for item in signals}

        self.assertIn(("EAEU", "product_identity_linkage"), by_key)
        signal = by_key[("EAEU", "product_identity_linkage")]
        self.assertEqual(signal.verdict, "confirmed")
        self.assertIn("match level", signal.summary.value)


if __name__ == "__main__":
    unittest.main()
