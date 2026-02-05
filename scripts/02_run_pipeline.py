#!/usr/bin/env python3
"""
Run nested ingestion pipeline.
Loads .env and config/pipeline.json; env overrides config file.
User only needs to set .env and optionally edit config/pipeline.json.
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env", override=False)
except ImportError:
    pass

from src.pipeline_nested import run_pipeline_nested


def load_config() -> dict:
    """Merge config/pipeline.json with .env (env wins)."""
    config_path = REPO_ROOT / "config" / "pipeline.json"
    if config_path.exists():
        import json
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {}

    # Env overrides
    def env_int(key: str, default=None):
        v = os.environ.get(key)
        return int(v) if v not in (None, "") else default

    def env_float(key: str, default=None):
        v = os.environ.get(key)
        return float(v) if v not in (None, "") else default

    def env_bool(key: str, default=None):
        v = os.environ.get(key, "").lower()
        if v in ("true", "1", "yes"):
            return True
        if v in ("false", "0", "no"):
            return False
        return default

    overrides = {
        "chunk_size": env_int("CHUNK_SIZE"),
        "chunk_overlap": env_int("CHUNK_OVERLAP"),
        "batch_size": env_int("BATCH_SIZE"),
        "num_workers": env_int("NUM_WORKERS"),
        "embedding_batch_size": env_int("EMBEDDING_BATCH_SIZE"),
        "embedding_parallel_workers": env_int("EMBEDDING_PARALLEL_WORKERS"),
        "batch_delay_seconds": env_float("BATCH_DELAY_SECONDS"),
        "max_bulk_bytes": env_int("MAX_BULK_BYTES"),
        "bulk_parallel_workers": env_int("BULK_PARALLEL_WORKERS"),
        "include_auxiliary": env_bool("INCLUDE_AUXILIARY"),
    }
    for k, v in overrides.items():
        if v is not None:
            config[k] = v

    config["opensearch_url"] = (os.environ.get("OPENSEARCH_URL") or "").strip()
    if not config["opensearch_url"] and os.environ.get("OPENSEARCH_HOST"):
        config["opensearch_url"] = "https://" + os.environ.get("OPENSEARCH_HOST", "").strip()
    config["opensearch_index"] = os.environ.get("OPENSEARCH_INDEX", config.get("opensearch_index", "wiki_kb_nested"))
    config["opensearch_region"] = os.environ.get("OPENSEARCH_REGION", "ap-south-1")
    # Embedding provider and dimension
    config["embedding_provider"] = os.environ.get("EMBEDDING_PROVIDER", config.get("embedding_provider", "opensearch")).lower()
    provider = config["embedding_provider"]
    default_dims = {"azure": 1536, "gemma": 768, "opensearch": 768}
    config["embedding_dimension"] = env_int("EMBEDDING_DIMENSION") or config.get("embedding_dimension") or default_dims.get(provider, 768)

    # Max chunks to embed per document (for doc_vector mean pooling)
    config["max_embedded_chunks"] = env_int("MAX_EMBEDDED_CHUNKS") or config.get("max_embedded_chunks", 3)

    # OpenSearch ML config (deployed model)
    config["opensearch_ml_model_id"] = os.environ.get("OPENSEARCH_ML_MODEL_ID", config.get("opensearch_ml_model_id", ""))
    config["opensearch_ml_max_chars"] = env_int("OPENSEARCH_ML_MAX_CHARS") or config.get("opensearch_ml_max_chars", 2000)

    # Azure config
    config["azure_endpoint"] = os.environ.get("AZURE_EMBEDDING_ENDPOINT", config.get("azure_endpoint", ""))
    config["azure_api_key"] = os.environ.get("AZURE_EMBEDDING_API_KEY", config.get("azure_api_key", ""))
    config["azure_api_version"] = os.environ.get("AZURE_EMBEDDING_API_VERSION", config.get("azure_api_version", "2024-02-01"))

    # Gemma config (local inference)
    config["gemma_model"] = os.environ.get("GEMMA_MODEL", config.get("gemma_model", "google/embeddinggemma-300m"))
    config["gemma_device"] = os.environ.get("GEMMA_DEVICE", config.get("gemma_device", "cpu"))
    config["gemma_max_chars"] = env_int("GEMMA_MAX_CHARS") or config.get("gemma_max_chars", 2000)

    index_name = config.get("opensearch_index", "wiki_kb_nested")
    raw_progress = os.environ.get("PROGRESS_FILE") or config.get("progress_file")
    if not raw_progress:
        # Default: output/<INDEX>_pipeline_progress.json (no _nested suffix)
        raw_progress = f"output/{index_name}_pipeline_progress.json"
    config["progress_file"] = str(REPO_ROOT / raw_progress) if not os.path.isabs(raw_progress) else raw_progress

    return config


def main():
    config = load_config()
    input_file = os.environ.get("INPUT_FILE", "").strip()
    if not input_file:
        print("Set INPUT_FILE in .env (path to .json.gz/.json.bz2 or directory of dumps)")
        sys.exit(1)
    if not os.path.isabs(input_file):
        input_file = str((REPO_ROOT / input_file).resolve())
    if not os.path.exists(input_file):
        print(f"INPUT_FILE not found: {input_file}")
        sys.exit(1)

    max_pages_val = os.environ.get("MAX_PAGES")
    max_pages = int(max_pages_val) if max_pages_val not in (None, "") else None

    print("Nested pipeline: one doc per article, chunks[] with vectors + doc_vector")
    print(f"  Input: {input_file}")
    print(f"  Index: {config['opensearch_index']}")
    print(f"  Embedding: {config['embedding_provider']} ({config['embedding_dimension']}-dim)")
    print(f"  Embed first {config['max_embedded_chunks']} chunks per doc (mean pooled to doc_vector)")
    print(f"  Max pages: {max_pages or 'all'}")

    run_pipeline_nested(
        input_file,
        config=config,
        max_pages=max_pages,
        verbose=True,
    )


if __name__ == "__main__":
    main()
