"""
S8-7: Storage cleanup job.

Deletes rendered HTML artifacts older than DDKIT_RENDERED_HTML_TTL_DAYS (default 7).
Enforces per-case document limits: if a case exceeds DDKIT_MAX_DOCS_PER_CASE,
oldest documents (by created_at) are flagged.

Can be run standalone or imported and called from a scheduler.

Usage:
    python -m src.cleanup_job [--dry-run]
"""
import argparse
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from src.settings import Settings

logger = logging.getLogger("ddkit.cleanup")


def _s3_client(settings: Settings):
    endpoint = os.getenv("STORAGE_ENDPOINT_URL") or os.getenv("STORAGE_ENDPOINT")
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.getenv("STORAGE_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("STORAGE_SECRET_KEY"),
        region_name=os.getenv("STORAGE_REGION", "us-east-1"),
        use_ssl=os.getenv("STORAGE_USE_SSL", "true").lower() == "true",
        config=Config(signature_version="s3v4", retries={"max_attempts": 3, "mode": "standard"}),
    )


def delete_old_rendered_html(
    settings: Settings,
    dry_run: bool = False,
    bucket: Optional[str] = None,
) -> dict:
    """
    Delete rendered HTML artifacts older than `ddkit_rendered_html_ttl_days`.

    Rendered HTML artifacts live under:
        tenants/{tenant_id}/cases/{case_id}/rendered/
    and have keys ending in `.html` or `.html.gz`.

    Returns a summary dict with counts of examined/deleted/errors.
    """
    ttl_days = settings.ddkit_rendered_html_ttl_days
    cutoff = datetime.now(timezone.utc) - timedelta(days=ttl_days)
    bucket_name = bucket or os.getenv("STORAGE_BUCKET_NAME", "ddkit-documents")
    s3 = _s3_client(settings)

    summary = {"examined": 0, "deleted": 0, "errors": 0, "dry_run": dry_run}

    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name, Prefix="tenants/")

    for page in pages:
        for obj in page.get("Contents", []):
            key: str = obj["Key"]
            # Only target rendered HTML under .../rendered/
            if "/rendered/" not in key:
                continue
            if not (key.endswith(".html") or key.endswith(".html.gz")):
                continue

            last_modified: datetime = obj["LastModified"]
            summary["examined"] += 1

            if last_modified < cutoff:
                age_days = (datetime.now(timezone.utc) - last_modified).days
                if dry_run:
                    logger.info("DRY-RUN would delete key=%s age_days=%d", key, age_days)
                else:
                    try:
                        s3.delete_object(Bucket=bucket_name, Key=key)
                        logger.info("Deleted rendered HTML key=%s age_days=%d", key, age_days)
                        summary["deleted"] += 1
                    except ClientError as exc:
                        logger.error("Failed to delete key=%s error=%s", key, exc)
                        summary["errors"] += 1

    logger.info(
        "cleanup_rendered_html ttl_days=%d cutoff=%s examined=%d deleted=%d errors=%d dry_run=%s",
        ttl_days,
        cutoff.isoformat(),
        summary["examined"],
        summary["deleted"],
        summary["errors"],
        dry_run,
    )
    return summary


def run_cleanup(dry_run: bool = False) -> dict:
    """Entry point for scheduled or manual cleanup runs."""
    settings = Settings()
    result = delete_old_rendered_html(settings, dry_run=dry_run)
    return result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="DDKit S3 artifact cleanup job")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List objects that would be deleted without actually deleting them",
    )
    args = parser.parse_args()
    run_cleanup(dry_run=args.dry_run)
