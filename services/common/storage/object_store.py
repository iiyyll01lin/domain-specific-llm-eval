import hashlib
import logging
import time
from typing import Callable, Optional

import boto3
from botocore.client import BaseClient
from botocore.exceptions import BotoCoreError, ClientError

from services.common.config import settings
from services.common.errors import ChecksumMismatchError, ObjectStoreError

logger = logging.getLogger(__name__)


def compute_checksum(payload: bytes) -> str:
    """Return the hex-encoded SHA256 checksum for the given payload."""
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


class ObjectStoreClient:
    """Lightweight wrapper around S3-compatible storage with retries and checksum validation."""

    def __init__(
        self,
        client: Optional[BaseClient] = None,
        sleep_func: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client or self._build_client()
        self._sleep = sleep_func
        self._max_attempts = max(1, settings.object_store_max_attempts)
        self._base_backoff = max(0.0, settings.object_store_backoff_seconds)

    def _build_client(self) -> BaseClient:
        return boto3.client(
            "s3",
            endpoint_url=settings.object_store_endpoint,
            region_name=settings.object_store_region,
            aws_access_key_id=settings.object_store_access_key,
            aws_secret_access_key=settings.object_store_secret_key,
            use_ssl=settings.object_store_use_ssl,
        )

    def _execute_with_retry(self, operation: str, func: Callable, *args, **kwargs):
        last_exception: Optional[Exception] = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                return func(*args, **kwargs)
            except (ClientError, BotoCoreError) as exc:
                last_exception = exc
                logger.warning(
                    "Object store %s attempt %s/%s failed: %s",
                    operation,
                    attempt,
                    self._max_attempts,
                    exc,
                )
                if attempt == self._max_attempts:
                    raise ObjectStoreError(
                        error_code="object_store_unavailable",
                        message=f"{operation} failed after {self._max_attempts} attempts",
                    ) from exc
                backoff = self._base_backoff * (2 ** (attempt - 1))
                if backoff > 0:
                    self._sleep(backoff)
        if last_exception:
            raise ObjectStoreError(
                error_code="object_store_unavailable",
                message=f"{operation} failed",
            ) from last_exception
        raise ObjectStoreError(
            error_code="object_store_unavailable",
            message=f"{operation} failed",
        )

    def upload_bytes(
        self,
        bucket: Optional[str],
        key: str,
        payload: bytes,
        expected_checksum: Optional[str] = None,
    ) -> str:
        target_bucket = bucket or settings.object_store_bucket
        if not target_bucket:
            raise ObjectStoreError(
                error_code="object_store_bucket_missing",
                message="Bucket name is required for upload",
            )
        checksum = expected_checksum or compute_checksum(payload)
        self._execute_with_retry(
            "put_object",
            self._client.put_object,
            Bucket=target_bucket,
            Key=key,
            Body=payload,
            Metadata={"checksum": checksum},
        )
        logger.info("Uploaded object %s/%s", target_bucket, key)
        return checksum

    def download_bytes(
        self,
        bucket: Optional[str],
        key: str,
        expected_checksum: Optional[str] = None,
    ) -> bytes:
        target_bucket = bucket or settings.object_store_bucket
        if not target_bucket:
            raise ObjectStoreError(
                error_code="object_store_bucket_missing",
                message="Bucket name is required for download",
            )
        response = self._execute_with_retry(
            "get_object",
            self._client.get_object,
            Bucket=target_bucket,
            Key=key,
        )
        body = response["Body"].read()
        metadata_checksum = response.get("Metadata", {}).get("checksum")
        checksum_to_compare = expected_checksum or metadata_checksum
        if checksum_to_compare:
            actual_checksum = compute_checksum(body)
            if actual_checksum != checksum_to_compare:
                raise ChecksumMismatchError(
                    message=(
                        f"Checksum mismatch for {target_bucket}/{key}: expected {checksum_to_compare} got {actual_checksum}"
                    )
                )
        return body

    def object_exists(self, bucket: Optional[str], key: str) -> bool:
        target_bucket = bucket or settings.object_store_bucket
        if not target_bucket:
            raise ObjectStoreError(
                error_code="object_store_bucket_missing",
                message="Bucket name is required to check existence",
            )
        try:
            self._execute_with_retry(
                "head_object",
                self._client.head_object,
                Bucket=target_bucket,
                Key=key,
            )
            return True
        except ObjectStoreError as exc:
            if isinstance(exc.__cause__, ClientError):
                error_code = exc.__cause__.response.get("Error", {}).get("Code")
                if error_code in {"404", "NoSuchKey"}:
                    return False
            raise

    def delete_object(self, bucket: Optional[str], key: str) -> None:
        target_bucket = bucket or settings.object_store_bucket
        if not target_bucket:
            raise ObjectStoreError(
                error_code="object_store_bucket_missing",
                message="Bucket name is required to delete objects",
            )
        self._execute_with_retry(
            "delete_object",
            self._client.delete_object,
            Bucket=target_bucket,
            Key=key,
        )
        logger.info("Deleted object %s/%s", target_bucket, key)

    def download_to_file(
        self,
        bucket: Optional[str],
        key: str,
        destination_path: str,
        expected_checksum: Optional[str] = None,
    ) -> str:
        data = self.download_bytes(bucket=bucket, key=key, expected_checksum=expected_checksum)
        with open(destination_path, "wb") as handle:
            handle.write(data)
        checksum = compute_checksum(data)
        return checksum

    def upload_file(
        self,
        bucket: Optional[str],
        key: str,
        file_path: str,
    ) -> str:
        with open(file_path, "rb") as handle:
            payload = handle.read()
        return self.upload_bytes(bucket=bucket, key=key, payload=payload)


__all__ = ["ObjectStoreClient", "compute_checksum", "ChecksumMismatchError", "ObjectStoreError"]
