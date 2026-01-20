import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional
import requests

from src.pipeline import Pipeline
from src.dd_report_generator import DDReportGenerator
from src.storage_client import StorageClient
from src.settings import settings
from src.ddkit_db import DDKitDB

logger = logging.getLogger(__name__)


class DocParseIndexProcessor:
    """Processor for doc_parse_index jobs."""

    def __init__(self):
        self.storage_client = StorageClient()
        self.ddkit_db = DDKitDB()

    def process_job(self, job_data: Dict[str, Any]) -> bool:
        """
        Process a doc_parse_index job.

        Args:
            job_data: Job data containing tenant_id, case_id, doc_id, doc_kind, s3_rendered_pdf_key

        Returns:
            bool: True if processing successful
        """
        tenant_id = job_data["tenant_id"]
        case_id = job_data["case_id"]
        doc_id = job_data["doc_id"]
        job_id = job_data.get("job_id")
        doc_kind = job_data["doc_kind"]
        title = job_data.get("title", "")
        source_url = job_data.get("source_url", "")
        s3_parsed_json_key = job_data.get("s3_parsed_json_key")
        s3_pdf_key = job_data["s3_rendered_pdf_key"]

        logger.info(f"Processing doc_parse_index job: tenant={tenant_id}, case={case_id}, doc={doc_id}")

        try:
            if job_id and self.ddkit_db.is_configured():
                self.ddkit_db.mark_job_running(job_id)
            # Create temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Download PDF from S3
                local_pdf_path = temp_path / f"{doc_id}.pdf"
                if not self.storage_client.download_to_path(s3_pdf_key, local_pdf_path):
                    logger.error(f"Failed to download PDF for doc {doc_id}")
                    return False

                # Create pipeline instance
                pipeline = Pipeline(temp_path)

                # Process single PDF through the pipeline
                success, parsed_key = self._process_single_document(
                    pipeline, local_pdf_path, temp_path, tenant_id, case_id, doc_id, doc_kind, title, source_url, s3_parsed_json_key
                )

                if success:
                    if job_id and self.ddkit_db.is_configured():
                        self.ddkit_db.mark_job_succeeded(job_id)
                    if parsed_key:
                        job_data["artifacts"] = {"s3_parsed_json_key": parsed_key}
                    job_data["status"] = "succeeded"
                    logger.info(f"Successfully processed document {doc_id}")
                    return True
                else:
                    if job_id and self.ddkit_db.is_configured():
                        self.ddkit_db.mark_job_failed(job_id, "processing_failed")
                    job_data["status"] = "failed"
                    logger.error(f"Failed to process document {doc_id}")
                    return False

        except Exception as e:
            if job_id and self.ddkit_db.is_configured():
                self.ddkit_db.mark_job_failed(job_id, str(e))
            logger.error(f"Error processing doc_parse_index job for {doc_id}: {e}")
            return False

    def _process_single_document(self, pipeline: Pipeline, pdf_path: Path, temp_path: Path,
                                tenant_id: str, case_id: str, doc_id: str, doc_kind: str,
                                title: str, source_url: str,
                                s3_parsed_json_key: Optional[str]) -> tuple[bool, Optional[str]]:
        """Process a single document through the ingestion pipeline."""
        try:
            # Step 1: Parse PDF
            parsed_reports_dir = temp_path / "01_parsed_reports"
            parsed_reports_dir.mkdir(exist_ok=True)

            # Use PDFParser to parse the single PDF
            from src.pdf_parsing import PDFParser

            # Create metadata CSV for this document
            metadata_csv = temp_path / "metadata.csv"
            with open(metadata_csv, 'w', encoding='utf-8') as f:
                f.write("filename,doc_id,doc_kind,tenant_id,case_id,title,source_url\n")
                f.write(f"{pdf_path.name},{doc_id},{doc_kind},{tenant_id},{case_id},{title},{source_url}\n")

            parser = PDFParser(output_dir=parsed_reports_dir, csv_metadata_path=metadata_csv)
            parser.parse_and_export(input_doc_paths=[pdf_path])

            # Step 2: Serialize tables (if needed)
            pipeline.serialize_tables(max_workers=1)

            # Step 3: Merge reports
            pipeline.merge_reports()

            # Step 4: Chunk reports
            pipeline.chunk_reports(include_serialized_tables=True)

            # Step 5: Create vector databases
            pipeline.create_vector_dbs()

            # Step 6: Upload results to S3
            parsed_key = self._upload_ingestion_results(
                temp_path, tenant_id, case_id, doc_id, s3_parsed_json_key
            )
            return (parsed_key is not None), parsed_key

        except Exception as e:
            logger.error(f"Error in document processing pipeline for {doc_id}: {e}")
            return False, None

    def _upload_ingestion_results(self, temp_path: Path, tenant_id: str, case_id: str, doc_id: str,
                                  s3_parsed_json_key: Optional[str] = None) -> Optional[str]:
        """Upload ingestion results to S3."""
        try:
            # Upload parsed JSON (docling)
            parsed_path = temp_path / "01_parsed_reports" / f"{doc_id}.json"
            parsed_key = s3_parsed_json_key or f"tenants/{tenant_id}/cases/{case_id}/documents/{doc_id}/parsed/docling.json"
            if parsed_path.exists():
                if not self.storage_client.upload_file(parsed_key, parsed_path):
                    logger.error(f"Failed to upload parsed JSON for doc {doc_id}")
                    return None
                if self.ddkit_db.is_configured():
                    pages_count = None
                    try:
                        with open(parsed_path, "r", encoding="utf-8") as f:
                            parsed_json = json.load(f)
                        pages_count = len(parsed_json.get("content", {}).get("pages", []))
                    except Exception:
                        pages_count = None
                    self.ddkit_db.update_document_parsed(doc_id, parsed_key, pages_count)

            # Upload chunked reports
            chunked_dir = temp_path / "databases" / "chunked_reports"
            if chunked_dir.exists():
                for chunk_file in chunked_dir.glob("*.json"):
                    s3_key = f"tenants/{tenant_id}/cases/{case_id}/documents/{doc_id}/chunks/{chunk_file.name}"
                    if not self.storage_client.upload_file(s3_key, chunk_file):
                        logger.error(f"Failed to upload chunk file {chunk_file.name}")
                        return None

            # Upload vector databases
            vector_dir = temp_path / "databases" / "vector_dbs"
            if vector_dir.exists():
                for vector_file in vector_dir.glob("*.faiss"):
                    s3_key = f"tenants/{tenant_id}/cases/{case_id}/documents/{doc_id}/vectors/{vector_file.name}"
                    if not self.storage_client.upload_file(s3_key, vector_file):
                        logger.error(f"Failed to upload vector file {vector_file.name}")
                        return None

            logger.info(f"Successfully uploaded ingestion results for doc {doc_id}")
            return parsed_key

        except Exception as e:
            logger.error(f"Error uploading ingestion results for {doc_id}: {e}")
            return None


class ReportGenerateProcessor:
    """Processor for report_generate jobs."""

    def __init__(self):
        self.storage_client = StorageClient()
        self.ddkit_db = DDKitDB()

    def process_job(self, job_data: Dict[str, Any]) -> bool:
        """
        Process a report_generate job.

        Args:
            job_data: Job data containing tenant_id, case_id, sections_plan_key

        Returns:
            bool: True if processing successful
        """
        tenant_id = job_data["tenant_id"]
        case_id = job_data["case_id"]
        sections_plan_key = job_data["sections_plan_key"]
        report_id = job_data.get("report_id") or f"report_{case_id}"
        job_id = job_data.get("job_id")

        logger.info(f"Processing report_generate job: tenant={tenant_id}, case={case_id}")

        try:
            if job_id and self.ddkit_db.is_configured():
                self.ddkit_db.mark_job_running(job_id)
            # Create temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Download sections plan from S3
                local_plan_path = temp_path / "sections_plan.json"
                if not self.storage_client.download_to_path(sections_plan_key, local_plan_path):
                    logger.error(f"Failed to download sections plan for case {case_id}")
                    return False

                # Load sections plan
                with open(local_plan_path, 'r', encoding='utf-8') as f:
                    sections_plan = json.load(f)

                # Download case artifacts (chunks + vectors) from storage
                if not self._download_case_artifacts(temp_path, tenant_id, case_id):
                    logger.error(f"Failed to download case artifacts for case {case_id}")
                    return False

                # Generate DD report
                output_path = temp_path / "dd_report.json"
                documents_dir = temp_path / "databases" / "chunked_reports"
                vector_dir = temp_path / "databases" / "vector_dbs"
                generator = DDReportGenerator(
                    documents_dir=documents_dir,
                    vector_db_dir=vector_dir,
                    tenant_id=tenant_id,
                    case_id=case_id
                )
                report_payload = generator.generate_report(sections_plan)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(report_payload, f, ensure_ascii=False, indent=2)

                # Upload report to S3
                s3_key = f"tenants/{tenant_id}/cases/{case_id}/reports/{report_id}/report.json"
                if not self.storage_client.upload_file(s3_key, output_path):
                    logger.error(f"Failed to upload report for case {case_id}")
                    return False

                if self.ddkit_db.is_configured():
                    self.ddkit_db.update_report_published(
                        report_id=report_id,
                        tenant_id=tenant_id,
                        case_id=case_id,
                        title=f"DD Report — {case_id}",
                        s3_report_json_key=s3_key,
                        sections_plan_key=sections_plan_key
                    )
                if job_id and self.ddkit_db.is_configured():
                    self.ddkit_db.mark_job_succeeded(job_id)

                job_data["artifacts"] = {"s3_report_json_key": s3_key}
                job_data["status"] = "succeeded"
                logger.info(f"Successfully generated and uploaded report for case {case_id}")
                return True

        except Exception as e:
            if job_id and self.ddkit_db.is_configured():
                self.ddkit_db.mark_job_failed(job_id, str(e))
            job_data["status"] = "failed"
            logger.error(f"Error processing report_generate job for case {case_id}: {e}")
            return False

    def _download_case_artifacts(self, temp_path: Path, tenant_id: str, case_id: str) -> bool:
        base_prefix = f"tenants/{tenant_id}/cases/{case_id}/documents/"
        keys = self.storage_client.list_objects(base_prefix)
        if not keys:
            logger.error(f"No artifacts found under {base_prefix}")
            return False

        chunks_dir = temp_path / "databases" / "chunked_reports"
        vectors_dir = temp_path / "databases" / "vector_dbs"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        vectors_dir.mkdir(parents=True, exist_ok=True)

        for key in keys:
            if "/chunks/" in key and key.endswith(".json"):
                local_path = chunks_dir / Path(key).name
                if not self.storage_client.download_to_path(key, local_path):
                    logger.error(f"Failed to download chunk {key}")
                    return False
            if "/vectors/" in key and key.endswith(".faiss"):
                local_path = vectors_dir / Path(key).name
                if not self.storage_client.download_to_path(key, local_path):
                    logger.error(f"Failed to download vector {key}")
                    return False
        return True


class JobCallback:
    """Handle job completion callbacks."""

    @staticmethod
    def send_callback(callback_url: str, job_data: Dict[str, Any], success: bool, error_message: Optional[str] = None):
        """Send callback to external service about job completion."""
        if not callback_url:
            return

        try:
            status = "succeeded" if success else "failed"
            payload = {
                "job_id": job_data.get("job_id"),
                "job_type": job_data.get("job_type"),
                "status": status,
                "tenant_id": job_data.get("tenant_id"),
                "case_id": job_data.get("case_id"),
                "doc_id": job_data.get("doc_id"),
                "report_id": job_data.get("report_id"),
                "artifacts": job_data.get("artifacts", {}),
                "success": success,
                "timestamp": int(time.time()),
                "error_message": error_message
            }

            headers = {}
            if settings.job_callback_token:
                headers["X-DDKIT-TOKEN"] = settings.job_callback_token
            response = requests.post(callback_url, json=payload, timeout=30, headers=headers)
            response.raise_for_status()

            logger.info(f"Sent callback for job {job_data.get('job_type')} to {callback_url}")

        except Exception as e:
            logger.error(f"Failed to send callback: {e}")


