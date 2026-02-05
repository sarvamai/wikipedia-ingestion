"""
Embedding providers: Azure OpenAI and local models (sentence-transformers).
Supports parallel workers, multi-process embedding, and token truncation.
"""
import os
import time
import requests
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


# Max chars per model (conservative estimates based on token limits)
MAX_CHARS_AZURE = 10000   # Azure: 8192 tokens, ~1.2 chars/token
MAX_CHARS_GEMMA = 2000    # Gemma: ~512 tokens, ~4 chars/token (conservative)

# Lazy-loaded model for local inference (per-process)
_gemma_model = None
_gemma_tokenizer = None


def _send_one_embedding_batch_azure(
    batch: List[str],
    url: str,
    headers: Dict,
    timeout: int,
    max_retries: int,
) -> List[List[float]]:
    """Send one batch to Azure embedding API. Returns embeddings for this batch."""
    if not batch:
        return []
    for attempt in range(max_retries + 1):
        response = requests.post(
            url,
            headers=headers,
            json={"input": batch},
            timeout=timeout,
        )
        if response.status_code == 429 and attempt < max_retries:
            time.sleep(2 ** attempt)
            continue
        break
    if response.status_code != 200:
        raise RuntimeError(f"Azure API error {response.status_code}: {response.text[:500]}")
    data = response.json()
    if "data" in data:
        sorted_data = sorted(data["data"], key=lambda x: x.get("index", 0))
        return [item["embedding"] for item in sorted_data]
    if "embeddings" in data:
        return data["embeddings"]
    raise RuntimeError(f"Unexpected response format: {list(data.keys())}")


def generate_embeddings_azure(texts: List[str], config: Dict) -> List[List[float]]:
    """
    Generate embeddings via Azure. Truncates text over 10000 chars to stay under 8192 tokens.
    When embedding_parallel_workers > 1, sends sub-batches concurrently.
    """
    safe_texts = []
    for t in texts:
        if len(t) > MAX_CHARS:
            truncated = t[:MAX_CHARS]
            last_period = truncated.rfind(". ")
            if last_period > MAX_CHARS * 0.7:
                truncated = truncated[: last_period + 1]
            safe_texts.append(truncated)
        else:
            safe_texts.append(t)
    texts = safe_texts

    endpoint = config.get("azure_endpoint")
    api_key = config.get("azure_api_key")
    api_version = config.get("azure_api_version", "2024-02-01")
    batch_size = config.get("embedding_batch_size", 32)
    timeout = config.get("embedding_timeout", 120)
    max_retries = config.get("embedding_max_retries", 5)
    parallel_workers = config.get("embedding_parallel_workers", 1)

    if not endpoint or not api_key:
        raise ValueError("Azure endpoint/key not configured. Set in .env.")

    is_openai = "openai.azure.com" in endpoint.lower()
    url = f"{endpoint}?api-version={api_version}" if is_openai and "?" not in endpoint else endpoint
    headers = {"Content-Type": "application/json", "api-key": api_key} if is_openai else {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    sub_batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    all_embeddings = []

    if parallel_workers <= 1 or len(sub_batches) <= 1:
        for batch in sub_batches:
            all_embeddings.extend(
                _send_one_embedding_batch_azure(batch, url, headers, timeout, max_retries)
            )
    else:
        workers = min(parallel_workers, len(sub_batches), 8)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _send_one_embedding_batch_azure,
                    batch,
                    url,
                    headers,
                    timeout,
                    max_retries,
                )
                for batch in sub_batches
            ]
            for fut in futures:
                all_embeddings.extend(fut.result())

    return all_embeddings


# ============================================================
# GEMMA EMBEDDING (local or endpoint)
# ============================================================

def _load_gemma_model(model_name: str, device: str = "cpu"):
    """Lazy-load embedding model for local inference."""
    global _gemma_model, _gemma_tokenizer
    if _gemma_model is not None:
        return _gemma_model, _gemma_tokenizer
    try:
        from sentence_transformers import SentenceTransformer
        # trust_remote_code=True needed for models with custom code (e.g. stella)
        _gemma_model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        _gemma_tokenizer = None  # sentence-transformers handles tokenization
        return _gemma_model, _gemma_tokenizer
    except ImportError:
        raise ImportError("sentence-transformers required for local inference. Install: uv sync --extra gemma")


def generate_embeddings_gemma_local(
    texts: List[str],
    model_name: str = "google/embeddinggemma-300m",
    device: str = "cpu",
    batch_size: int = 32,
    output_dim: Optional[int] = None,
) -> List[List[float]]:
    """
    Generate embeddings using local model via sentence-transformers (single process).
    Default: google/embeddinggemma-300m (768-dim, MRL supports 512/256/128).
    """
    model, _ = _load_gemma_model(model_name, device)
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embeddings = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        # MRL truncation: slice to output_dim if specified
        if output_dim and embeddings.shape[1] > output_dim:
            embeddings = embeddings[:, :output_dim]
        all_embeddings.extend(embeddings.tolist())
    return all_embeddings


def _embed_worker(args: Tuple[List[str], str, str, int, Optional[int], int]) -> Tuple[int, List[List[float]]]:
    """
    Worker function for multi-process embedding.
    Each worker loads its own model copy and processes a chunk of texts.
    Returns (worker_id, embeddings) to preserve order.
    """
    texts, model_name, device, batch_size, output_dim, worker_id = args
    
    # Each process needs its own model (global is per-process)
    global _gemma_model
    _gemma_model = None  # Force reload in this process
    
    model, _ = _load_gemma_model(model_name, device)
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embeddings = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        if output_dim and embeddings.shape[1] > output_dim:
            embeddings = embeddings[:, :output_dim]
        all_embeddings.extend(embeddings.tolist())
    return (worker_id, all_embeddings)


def generate_embeddings_multiprocess(
    texts: List[str],
    model_name: str = "google/embeddinggemma-300m",
    device: str = "cpu",
    batch_size: int = 32,
    output_dim: Optional[int] = None,
    num_processes: int = 4,
) -> List[List[float]]:
    """
    Generate embeddings using multiple processes, each with its own model copy.
    
    This provides true parallelism by running N model instances simultaneously.
    Uses more RAM (~4GB per process for embeddinggemma-300m) but gives N-fold speedup.
    
    Args:
        texts: List of texts to embed
        model_name: HuggingFace model name
        device: 'cpu' or 'cuda'
        batch_size: Batch size per process
        output_dim: MRL truncation dimension (768, 512, 256, 128)
        num_processes: Number of parallel processes (each loads model)
    """
    if len(texts) == 0:
        return []
    
    # For small inputs, don't bother with multiprocessing overhead
    if len(texts) < 100 or num_processes <= 1:
        return generate_embeddings_gemma_local(texts, model_name, device, batch_size, output_dim)
    
    # Split texts into chunks for each process
    chunk_size = (len(texts) + num_processes - 1) // num_processes
    chunks = []
    for i in range(num_processes):
        start = i * chunk_size
        end = min(start + chunk_size, len(texts))
        if start < len(texts):
            chunks.append((texts[start:end], model_name, device, batch_size, output_dim, i))
    
    # Process in parallel
    results = [None] * len(chunks)
    with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
        for worker_id, embeddings in executor.map(_embed_worker, chunks):
            results[worker_id] = embeddings
    
    # Flatten results in order
    all_embeddings = []
    for emb_list in results:
        if emb_list:
            all_embeddings.extend(emb_list)
    
    return all_embeddings


def generate_embeddings_gemma(texts: List[str], config: Dict) -> List[List[float]]:
    """
    Generate embeddings via local model (sentence-transformers).
    Config keys:
      - gemma_model: model name/path (default: google/embeddinggemma-300m)
      - gemma_device: cpu or cuda (default: cpu)
      - embedding_dimension: output dimension for MRL truncation (768, 512, 256, 128)
      - gemma_max_chars: max chars per text (default: 2000)
      - embedding_processes: number of parallel processes (default: 1, each loads model)
    """
    # Truncate texts for shorter context
    max_chars = config.get("gemma_max_chars", MAX_CHARS_GEMMA)
    safe_texts = []
    for t in texts:
        if len(t) > max_chars:
            truncated = t[:max_chars]
            last_period = truncated.rfind(". ")
            if last_period > max_chars * 0.7:
                truncated = truncated[: last_period + 1]
            safe_texts.append(truncated)
        else:
            safe_texts.append(t)

    model_name = config.get("gemma_model", "google/embeddinggemma-300m")
    device = config.get("gemma_device", "cpu")
    batch_size = config.get("embedding_batch_size", 32)
    output_dim = config.get("embedding_dimension")
    num_processes = config.get("embedding_processes", 1)

    if num_processes > 1 and device == "cpu":
        # Multi-process: each process loads its own model copy
        return generate_embeddings_multiprocess(
            safe_texts, model_name, device, batch_size, output_dim, num_processes
        )
    else:
        # Single process (or GPU - multiprocess doesn't help much with GPU)
        return generate_embeddings_gemma_local(safe_texts, model_name, device, batch_size, output_dim)


def truncate_text(text: str, max_tokens: int = 6000) -> str:
    """Truncate to fit token limit. ~2.5 chars per token."""
    max_chars = int(max_tokens * 2.5)
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_period = truncated.rfind(". ")
    if last_period > max_chars * 0.7:
        return truncated[: last_period + 1]
    return truncated


def generate_embeddings(texts: List[str], config: Dict) -> List[List[float]]:
    """
    Generate embeddings using configured provider.
    Providers: azure, gemma, or custom endpoint.
    """
    provider = config.get("embedding_provider", "azure").lower()

    if provider == "azure":
        max_tokens = config.get("max_chunk_tokens", 6000)
        truncated_texts = [truncate_text(t, max_tokens) for t in texts]
        return generate_embeddings_azure(truncated_texts, config)

    if provider == "gemma":
        # Gemma handles its own truncation based on gemma_max_chars
        return generate_embeddings_gemma(texts, config)

    # Custom endpoint fallback
    endpoint = config.get("embedding_endpoint", "")
    if endpoint:
        max_tokens = config.get("max_chunk_tokens", 6000)
        truncated_texts = [truncate_text(t, max_tokens) for t in texts]
        batch_size = config.get("embedding_batch_size", 32)
        timeout = config.get("embedding_timeout", 120)
        all_embeddings = []
        for i in range(0, len(truncated_texts), batch_size):
            batch = truncated_texts[i : i + batch_size]
            r = requests.post(endpoint, json={"texts": batch}, timeout=timeout)
            r.raise_for_status()
            all_embeddings.extend(r.json().get("embeddings", []))
        return all_embeddings

    raise ValueError("No embedding provider configured. Set EMBEDDING_PROVIDER=azure or gemma in .env.")
