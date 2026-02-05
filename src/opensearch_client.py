"""
OpenSearch client for bulk indexing with AWS auth.
"""
import os
import json
import requests
import time
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor


def _get_opensearch_auth(region: str):
    """Get AWS4Auth for OpenSearch requests."""
    try:
        from requests_aws4auth import AWS4Auth
        aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        if aws_access_key and aws_secret_key:
            return AWS4Auth(aws_access_key, aws_secret_key, region, "es")
    except ImportError:
        pass
    return None


def _split_documents_by_size(
    documents: List[Dict], index_name: str, max_bytes: int
) -> List[List[Dict]]:
    """Split documents into sub-batches that fit within max_bytes."""
    batches = []
    current_batch = []
    current_size = 0
    
    for doc in documents:
        action_line = json.dumps({"index": {"_index": index_name, "_id": doc["_id"]}})
        source_line = json.dumps(doc["_source"])
        doc_size = len(action_line.encode("utf-8")) + len(source_line.encode("utf-8")) + 2  # +2 for newlines
        
        if current_size + doc_size > max_bytes and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_size = 0
        
        current_batch.append(doc)
        current_size += doc_size
    
    if current_batch:
        batches.append(current_batch)
    
    return batches


def _send_one_bulk(
    documents: List[Dict],
    url: str,
    index_name: str,
    auth,
    timeout: int,
    max_retries: int,
) -> Tuple[int, int]:
    """Send one bulk request. Returns (success_count, failed_count)."""
    bulk_body = ""
    for doc in documents:
        bulk_body += json.dumps({"index": {"_index": index_name, "_id": doc["_id"]}}) + "\n"
        bulk_body += json.dumps(doc["_source"]) + "\n"
    body_bytes = bulk_body.encode("utf-8")
    
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                f"{url}/_bulk",
                data=body_bytes,
                headers={"Content-Type": "application/x-ndjson"},
                auth=auth,
                timeout=timeout,
            )
            if resp.status_code == 429 and attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            out = resp.json()
            items = out.get("items", [])
            success = 0
            failed = 0
            for it in items:
                idx_result = it.get("index", {})
                if idx_result.get("status") in (200, 201):
                    success += 1
                else:
                    failed += 1
                    # Print first error for debugging
                    if failed == 1:
                        error = idx_result.get("error", {})
                        print(f"    Bulk error: {error.get('type', 'unknown')}: {error.get('reason', 'no reason')[:200]}", flush=True)
            return success, failed
        except requests.RequestException as e:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            raise
    return 0, len(documents)


def push_batch_to_opensearch(documents: List[Dict], config: Dict) -> Tuple[int, int]:
    """
    Push bulk documents to OpenSearch. Splits by max_bulk_bytes, uses bulk_parallel_workers.
    Returns (success_count, failed_count).
    """
    if not documents:
        return 0, 0
    url = (config.get("opensearch_url") or "").strip()
    if not url:
        return 0, len(documents)
    if not url.startswith("http"):
        url = "https://" + url
    index_name = config.get("opensearch_index", "wiki_kb_nested")
    region = config.get("opensearch_region", "ap-south-1")
    max_bulk_bytes = config.get("max_bulk_bytes", 12 * 1024 * 1024)
    timeout = config.get("bulk_request_timeout", 60)
    max_retries = config.get("bulk_max_retries", 5)
    
    auth = _get_opensearch_auth(region)
    if not auth:
        print("    Warning: No AWS auth configured", flush=True)
        return 0, len(documents)
    
    sub_batches = _split_documents_by_size(documents, index_name, max_bulk_bytes)
    total_success, total_failed = 0, 0
    parallel_workers = min(config.get("bulk_parallel_workers", 4), len(sub_batches), 16)
    
    try:
        if parallel_workers <= 1 or len(sub_batches) <= 1:
            for sub in sub_batches:
                success, failed = _send_one_bulk(sub, url, index_name, auth, timeout, max_retries)
                total_success += success
                total_failed += failed
        else:
            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                futures = [
                    executor.submit(_send_one_bulk, sub, url, index_name, auth, timeout, max_retries)
                    for sub in sub_batches
                ]
                for fut in futures:
                    success, failed = fut.result()
                    total_success += success
                    total_failed += failed
        return total_success, total_failed
    except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
        print(f"    Bulk push error: {e}", flush=True)
        return total_success, len(documents) - total_success


def load_progress(progress_file: str) -> Dict:
    """Load progress for resume."""
    if not progress_file or not os.path.exists(progress_file):
        return {}
    try:
        with open(progress_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_progress(progress_file: str, stats: Dict, batch_num: int = 0, stream_state: Optional[Dict] = None) -> None:
    """Save progress for resume."""
    if not progress_file:
        return
    try:
        progress = {
            "stats": stats,
            "batch_num": batch_num,
            "stream_state": stream_state or {},
        }
        if os.path.dirname(progress_file):
            os.makedirs(os.path.dirname(progress_file), exist_ok=True)
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump(progress, f)
    except IOError:
        pass
