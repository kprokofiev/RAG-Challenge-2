import json
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

from .storage_client import StorageClient

logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Metadata for a document being processed"""
    doc_id: str
    tenant_id: str
    case_id: str
    doc_kind: str
    title: Optional[str] = None
    source_url: Optional[str] = None
    retrieved_at: Optional[str] = None
    s3_rendered_pdf_key: str = ""
    s3_parsed_json_key: Optional[str] = None


@dataclass
class LoadedDocument:
    """Result of loading a document"""
    metadata: DocumentMetadata
    local_pdf_path: str
    checksum: Optional[str] = None


class DocumentLoader:
    """
    Loads documents from storage for processing.
    Handles downloading rendered PDFs and managing temporary files.
    """

    def __init__(self, storage_client: Optional[StorageClient] = None):
        self.storage = storage_client or StorageClient()
        self.temp_dir = Path(tempfile.gettempdir()) / "ddkit_documents"

    def load_from_manifest_entry(self, manifest_entry: Dict[str, Any]) -> Optional[LoadedDocument]:
        """
        Load a document from a manifest entry.

        Args:
            manifest_entry: Dictionary with document metadata including s3_rendered_pdf_key

        Returns:
            LoadedDocument or None if loading failed
        """
        try:
            # Extract metadata
            metadata = DocumentMetadata(
                doc_id=manifest_entry['doc_id'],
                tenant_id=manifest_entry.get('tenant_id', ''),
                case_id=manifest_entry.get('case_id', ''),
                doc_kind=manifest_entry.get('doc_kind', 'unknown'),
                title=manifest_entry.get('title'),
                source_url=manifest_entry.get('source_url'),
                retrieved_at=manifest_entry.get('retrieved_at'),
                s3_rendered_pdf_key=manifest_entry['s3_rendered_pdf_key']
            )

            # Generate parsed JSON key if not provided
            if not metadata.s3_parsed_json_key:
                metadata.s3_parsed_json_key = self._generate_parsed_key(metadata)

            # Download PDF to temp location
            local_pdf_path = self._download_pdf_to_temp(metadata)

            if not local_pdf_path:
                return None

            # Calculate checksum if needed
            checksum = self._calculate_checksum(local_pdf_path)

            loaded_doc = LoadedDocument(
                metadata=metadata,
                local_pdf_path=local_pdf_path,
                checksum=checksum
            )

            logger.info(f"Successfully loaded document {metadata.doc_id}: {local_pdf_path}")
            return loaded_doc

        except KeyError as e:
            logger.error(f"Missing required field in manifest entry: {e}")
            return None
        except Exception as e:
            logger.error(f"Error loading document from manifest: {e}")
            return None

    def load_single_document(self, tenant_id: str, case_id: str, doc_id: str,
                           s3_rendered_pdf_key: str, doc_kind: str = 'unknown',
                           title: Optional[str] = None) -> Optional[LoadedDocument]:
        """
        Load a single document by parameters.

        Args:
            tenant_id: Tenant identifier
            case_id: Case identifier
            doc_id: Document identifier
            s3_rendered_pdf_key: S3 key for the rendered PDF
            doc_kind: Document type/kind
            title: Optional document title

        Returns:
            LoadedDocument or None if loading failed
        """
        manifest_entry = {
            'doc_id': doc_id,
            'tenant_id': tenant_id,
            'case_id': case_id,
            'doc_kind': doc_kind,
            'title': title,
            's3_rendered_pdf_key': s3_rendered_pdf_key
        }

        return self.load_from_manifest_entry(manifest_entry)

    def cleanup_temp_files(self, loaded_doc: LoadedDocument):
        """
        Clean up temporary files for a loaded document.

        Args:
            loaded_doc: The loaded document to clean up
        """
        try:
            if os.path.exists(loaded_doc.local_pdf_path):
                os.remove(loaded_doc.local_pdf_path)
                logger.info(f"Cleaned up temp file: {loaded_doc.local_pdf_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file {loaded_doc.local_pdf_path}: {e}")

    def _download_pdf_to_temp(self, metadata: DocumentMetadata) -> Optional[str]:
        """
        Download PDF to a temporary location.

        Args:
            metadata: Document metadata

        Returns:
            Local path to downloaded PDF or None if failed
        """
        # Create temp directory structure
        temp_dir = self.temp_dir / metadata.tenant_id / metadata.case_id
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique temp filename
        temp_filename = f"{metadata.doc_id}_{metadata.s3_rendered_pdf_key.split('/')[-1]}"
        local_path = temp_dir / temp_filename

        # Download file
        success = self.storage.download_to_path(metadata.s3_rendered_pdf_key, local_path)

        if success:
            return str(local_path)
        else:
            logger.error(f"Failed to download PDF for doc {metadata.doc_id}")
            return None

    def _generate_parsed_key(self, metadata: DocumentMetadata) -> str:
        """
        Generate S3 key for parsed JSON based on document metadata.

        Args:
            metadata: Document metadata

        Returns:
            S3 key for parsed JSON
        """
        return f"tenants/{metadata.tenant_id}/cases/{metadata.case_id}/documents/{metadata.doc_id}/parsed/docling.json"

    def _calculate_checksum(self, file_path: str) -> Optional[str]:
        """
        Calculate MD5 checksum of file.

        Args:
            file_path: Path to file

        Returns:
            MD5 checksum as hex string or None if failed
        """
        try:
            import hashlib
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5()
                while chunk := f.read(8192):
                    file_hash.update(chunk)
            return file_hash.hexdigest()
        except Exception as e:
            logger.warning(f"Failed to calculate checksum for {file_path}: {e}")
            return None


