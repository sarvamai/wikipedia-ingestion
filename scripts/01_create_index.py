#!/usr/bin/env python3
"""
Create OpenSearch index with nested k-NN schema.
Reads config/index_schema.json and .env (OPENSEARCH_*, AWS_*, EMBEDDING_*).
Dimension is set from EMBEDDING_DIMENSION env var (default 1536 for Azure, 768 for Gemma).
"""
import os
import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env", override=False)
except ImportError:
    pass

OPENSEARCH_URL = (os.environ.get("OPENSEARCH_URL") or "").strip()
if not OPENSEARCH_URL and os.environ.get("OPENSEARCH_HOST"):
    OPENSEARCH_URL = "https://" + os.environ.get("OPENSEARCH_HOST", "").strip()
if not OPENSEARCH_URL or not OPENSEARCH_URL.startswith("http"):
    print("Set OPENSEARCH_HOST (or OPENSEARCH_URL) in .env")
    sys.exit(1)

INDEX_NAME = os.environ.get("OPENSEARCH_INDEX", "wiki_kb_nested")
OPENSEARCH_REGION = os.environ.get("OPENSEARCH_REGION", "ap-south-1")

# Embedding dimension: default based on provider
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "azure").lower()
DEFAULT_DIMS = {"azure": 1536, "gemma": 768}
EMBEDDING_DIMENSION = int(os.environ.get("EMBEDDING_DIMENSION", DEFAULT_DIMS.get(EMBEDDING_PROVIDER, 1536)))


def _request(method: str, url: str, data: dict = None):
    try:
        from requests_aws4auth import AWS4Auth
        import requests
        auth = AWS4Auth(
            os.environ.get("AWS_ACCESS_KEY_ID"),
            os.environ.get("AWS_SECRET_ACCESS_KEY"),
            OPENSEARCH_REGION,
            "es",
        )
        if data is not None:
            resp = requests.request(method, url, json=data, auth=auth, timeout=60)
        else:
            resp = requests.request(method, url, auth=auth, timeout=60)
        if resp.text:
            try:
                return resp.json()
            except json.JSONDecodeError:
                return {"raw": resp.text}
        return None
    except ImportError:
        import subprocess
        cmd = [
            "awscurl", "--service", "es", "--region", OPENSEARCH_REGION,
            "-X", method, url
        ]
        env = os.environ.copy()
        env.pop("AWS_SESSION_TOKEN", None)
        if data:
            cmd.extend(["-H", "Content-Type: application/json", "-d", json.dumps(data)])
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if result.stdout:
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                return {"raw": result.stdout}
        if result.stderr:
            return {"error": result.stderr}
        return None


def main():
    schema_path = REPO_ROOT / "config" / "index_schema.json"
    if not schema_path.exists():
        print(f"Missing {schema_path}")
        sys.exit(1)
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    # Override dimension from env (EMBEDDING_DIMENSION or default based on provider)
    try:
        # Update chunk-level vector dimension
        schema["mappings"]["properties"]["chunks"]["properties"]["vector"]["dimension"] = EMBEDDING_DIMENSION
        # Update doc-level vector dimension
        schema["mappings"]["properties"]["doc_vector"]["dimension"] = EMBEDDING_DIMENSION
    except KeyError:
        pass

    print(f"OpenSearch index (nested schema, {EMBEDDING_DIMENSION}-dim vectors)")
    print(f"  - doc_vector: article-level embedding (mean of first N chunks)")
    print(f"  - chunks[].vector: chunk-level embeddings (first N chunks only)")
    result = _request("GET", OPENSEARCH_URL)
    if not result or "version" not in result:
        print("Cannot reach OpenSearch:", result)
        sys.exit(1)
    print(f"OpenSearch: {result['version'].get('number', 'ok')}")

    result = _request("GET", f"{OPENSEARCH_URL}/{INDEX_NAME}")
    if result and "error" not in result:
        print(f"Index '{INDEX_NAME}' exists. Deleting...")
        _request("DELETE", f"{OPENSEARCH_URL}/{INDEX_NAME}")
        print("Deleted.")

    result = _request("PUT", f"{OPENSEARCH_URL}/{INDEX_NAME}", schema)
    if result and result.get("acknowledged"):
        print(f"Index '{INDEX_NAME}' created.")
        settings = schema.get("settings", {}).get("index", {})
        print(f"  Shards: {settings.get('number_of_shards', '?')}, Replicas: {settings.get('number_of_replicas', '?')}, Vector dim: {EMBEDDING_DIMENSION}")
    else:
        print("ERROR creating index:", result)
        sys.exit(1)


if __name__ == "__main__":
    main()
