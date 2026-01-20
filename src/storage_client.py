import os
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from typing import Optional, BinaryIO
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class StorageClient:
    """
    Abstraction for S3/MinIO storage operations.
    Supports downloading rendered PDFs and uploading parsed JSON/report JSON.
    """

    def __init__(self):
        self.endpoint_url = os.getenv('STORAGE_ENDPOINT_URL') or os.getenv('STORAGE_ENDPOINT')
        self.access_key = os.getenv('STORAGE_ACCESS_KEY')
        self.secret_key = os.getenv('STORAGE_SECRET_KEY')
        self.bucket_name = os.getenv('STORAGE_BUCKET_NAME', 'ddkit-documents')
        self.region = os.getenv('STORAGE_REGION', 'us-east-1')
        self.use_ssl = os.getenv('STORAGE_USE_SSL', 'true').lower() == 'true'

        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region,
            use_ssl=self.use_ssl,
            config=Config(
                signature_version='s3v4',
                retries={'max_attempts': 3, 'mode': 'standard'}
            )
        )

    def download_to_path(self, s3_key: str, local_path: str | Path) -> bool:
        """
        Download file from S3/MinIO to local path.

        Args:
            s3_key: S3 key/path of the file
            local_path: Local file path to save to

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            local_path = Path(local_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"Downloading s3://{self.bucket_name}/{s3_key} to {local_path}")
            self.s3_client.download_file(self.bucket_name, s3_key, str(local_path))

            # Verify file exists and has content
            if not local_path.exists():
                logger.error(f"Downloaded file not found at {local_path}")
                return False

            file_size = local_path.stat().st_size
            logger.info(f"Successfully downloaded {file_size} bytes to {local_path}")
            return True

        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                logger.error(f"File not found: s3://{self.bucket_name}/{s3_key}")
            else:
                logger.error(f"S3 download error for {s3_key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error downloading {s3_key}: {e}")
            return False

    def upload_bytes(self, s3_key: str, data: bytes, content_type: str = 'application/json') -> bool:
        """
        Upload bytes data to S3/MinIO.

        Args:
            s3_key: S3 key/path for the file
            data: Bytes to upload
            content_type: MIME type

        Returns:
            bool: True if successful
        """
        try:
            logger.info(f"Uploading {len(data)} bytes to s3://{self.bucket_name}/{s3_key}")

            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=data,
                ContentType=content_type
            )

            logger.info(f"Successfully uploaded to s3://{self.bucket_name}/{s3_key}")
            return True

        except ClientError as e:
            logger.error(f"S3 upload error for {s3_key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error uploading {s3_key}: {e}")
            return False

    def list_objects(self, prefix: str) -> list[str]:
        """List object keys under the given prefix."""
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            keys: list[str] = []
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                for item in page.get('Contents', []):
                    keys.append(item['Key'])
            return keys
        except Exception as e:
            logger.error(f"S3 list error for {prefix}: {e}")
            return []

    def upload_file(self, s3_key: str, file_path: str | Path, content_type: Optional[str] = None) -> bool:
        """
        Upload local file to S3/MinIO.

        Args:
            s3_key: S3 key/path for the file
            file_path: Local file path
            content_type: MIME type (auto-detected if None)

        Returns:
            bool: True if successful
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"Local file not found: {file_path}")
                return False

            file_size = file_path.stat().st_size
            logger.info(f"Uploading {file_size} bytes from {file_path} to s3://{self.bucket_name}/{s3_key}")

            # Auto-detect content type if not provided
            if content_type is None:
                if file_path.suffix.lower() == '.pdf':
                    content_type = 'application/pdf'
                elif file_path.suffix.lower() == '.json':
                    content_type = 'application/json'
                else:
                    content_type = 'application/octet-stream'

            with open(file_path, 'rb') as f:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    Body=f,
                    ContentType=content_type
                )

            logger.info(f"Successfully uploaded {file_path} to s3://{self.bucket_name}/{s3_key}")
            return True

        except ClientError as e:
            logger.error(f"S3 upload error for {s3_key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error uploading {file_path}: {e}")
            return False

    def exists(self, s3_key: str) -> bool:
        """
        Check if file exists in S3/MinIO.

        Args:
            s3_key: S3 key/path

        Returns:
            bool: True if exists
        """
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return False
            logger.error(f"Error checking existence of {s3_key}: {e}")
            return False

    def get_file_size(self, s3_key: str) -> Optional[int]:
        """
        Get file size in bytes.

        Args:
            s3_key: S3 key/path

        Returns:
            int or None: File size in bytes, None if not found
        """
        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return response['ContentLength']
        except ClientError as e:
            if e.response['Error']['Code'] != 'NoSuchKey':
                logger.error(f"Error getting size of {s3_key}: {e}")
            return None


