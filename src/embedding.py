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
        if len(t) > MAX_CHARS_AZURE:
            truncated = t[:MAX_CHARS_AZURE]
            last_period = truncated.rfind(". ")
            if last_period > MAX_CHARS_AZURE * 0.7:
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

def _load_gemma_model(model_name: str, device: str = "cpu", use_fp16: bool = False):
    """Lazy-load embedding model for local inference.
    
    Args:
        model_name: HuggingFace model name
        device: 'cpu' or 'cuda'
        use_fp16: If True, use FP16 (half precision) for ~1.5-2x speedup on CPU
    """
    global _gemma_model, _gemma_tokenizer
    if _gemma_model is not None:
        return _gemma_model, _gemma_tokenizer
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        
        # trust_remote_code=True needed for models with custom code (e.g. stella)
        _gemma_model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        
        # Apply FP16 quantization for faster CPU inference
        if use_fp16:
            print(f"  [Quantization] Converting model to FP16 for faster inference...")
            _gemma_model = _gemma_model.half()
            # For CPU, we need to ensure the model stays on CPU after .half()
            if device == "cpu":
                _gemma_model = _gemma_model.to(device)
        
        # Try to use torch.compile for additional speedup (PyTorch 2.0+)
        if hasattr(torch, 'compile') and device == "cpu":
            try:
                # mode="reduce-overhead" is good for repeated inference
                _gemma_model = torch.compile(_gemma_model, mode="reduce-overhead")
                print(f"  [Optimization] torch.compile enabled for faster inference")
            except Exception:
                pass  # torch.compile may not work on all systems
        
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
    use_fp16: bool = False,
) -> List[List[float]]:
    """
    Generate embeddings using local model via sentence-transformers (single process).
    Default: google/embeddinggemma-300m (768-dim, MRL supports 512/256/128).
    
    Args:
        use_fp16: If True, use FP16 (half precision) for ~1.5-2x speedup
    """
    model, _ = _load_gemma_model(model_name, device, use_fp16=use_fp16)
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embeddings = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        # MRL truncation: slice to output_dim if specified
        if output_dim and embeddings.shape[1] > output_dim:
            embeddings = embeddings[:, :output_dim]
        all_embeddings.extend(embeddings.tolist())
    return all_embeddings


# ============================================================
# PERSISTENT PROCESS POOL (keeps models loaded across batches)
# ============================================================

_persistent_pool = None
_pool_config = {}


def _worker_init(model_name: str, device: str, use_fp16: bool):
    """Initialize worker process - loads model once and keeps it in memory."""
    global _gemma_model
    _gemma_model = None  # Clear any existing
    print(f"  [Worker {os.getpid()}] Loading model {model_name}...")
    _load_gemma_model(model_name, device, use_fp16=use_fp16)
    print(f"  [Worker {os.getpid()}] Model loaded and ready")


def _embed_worker_persistent(args: Tuple) -> Tuple[int, List[List[float]]]:
    """
    Worker function for persistent pool - model is already loaded.
    """
    texts, batch_size, output_dim, worker_id = args
    
    global _gemma_model
    if _gemma_model is None:
        raise RuntimeError("Worker model not initialized. Call start_embedding_pool() first.")
    
    model = _gemma_model
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embeddings = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        if output_dim and embeddings.shape[1] > output_dim:
            embeddings = embeddings[:, :output_dim]
        all_embeddings.extend(embeddings.tolist())
    return (worker_id, all_embeddings)


def start_embedding_pool(
    model_name: str = "google/embeddinggemma-300m",
    device: str = "cpu",
    num_processes: int = 2,
    use_fp16: bool = False,
):
    """
    Start a persistent process pool with models pre-loaded.
    Call this ONCE at the start of your pipeline.
    Models stay loaded across all batches - no reload overhead!
    """
    global _persistent_pool, _pool_config
    
    if _persistent_pool is not None:
        print("  [Pool] Already running, reusing existing pool")
        return
    
    print(f"  [Pool] Starting {num_processes} workers with {model_name}...")
    
    # Store config for later use
    _pool_config = {
        "model_name": model_name,
        "device": device,
        "num_processes": num_processes,
        "use_fp16": use_fp16,
    }
    
    # Create pool with initializer that loads the model in each worker
    import functools
    initializer = functools.partial(_worker_init, model_name, device, use_fp16)
    
    _persistent_pool = ProcessPoolExecutor(
        max_workers=num_processes,
        initializer=initializer,
    )
    
    # Warm up the pool by submitting a dummy task to each worker
    # This ensures all workers have loaded their models before we start
    futures = [_persistent_pool.submit(lambda: os.getpid()) for _ in range(num_processes)]
    for f in futures:
        f.result()  # Wait for initialization
    
    print(f"  [Pool] All {num_processes} workers ready with models loaded!")


def stop_embedding_pool():
    """Shut down the persistent process pool."""
    global _persistent_pool, _pool_config
    if _persistent_pool is not None:
        print("  [Pool] Shutting down workers...")
        _persistent_pool.shutdown(wait=True)
        _persistent_pool = None
        _pool_config = {}
        print("  [Pool] Shutdown complete")


def _embed_worker(args: Tuple) -> Tuple[int, List[List[float]]]:
    """
    Worker function for non-persistent (legacy) multi-process embedding.
    Each worker loads its own model copy and processes a chunk of texts.
    Returns (worker_id, embeddings) to preserve order.
    """
    texts, model_name, device, batch_size, output_dim, worker_id, use_fp16 = args
    
    # Each process needs its own model (global is per-process)
    global _gemma_model
    _gemma_model = None  # Force reload in this process
    
    model, _ = _load_gemma_model(model_name, device, use_fp16=use_fp16)
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
    use_fp16: bool = False,
) -> List[List[float]]:
    """
    Generate embeddings using multiple processes, each with its own model copy.
    
    If a persistent pool is running (via start_embedding_pool), uses that.
    Otherwise falls back to creating workers per-batch (slower due to reload).
    
    Args:
        texts: List of texts to embed
        model_name: HuggingFace model name
        device: 'cpu' or 'cuda'
        batch_size: Batch size per process
        output_dim: MRL truncation dimension (768, 512, 256, 128)
        num_processes: Number of parallel processes (each loads model)
        use_fp16: If True, use FP16 (half precision) for ~1.5-2x speedup
    """
    global _persistent_pool, _pool_config
    
    if len(texts) == 0:
        return []
    
    # For small inputs, don't bother with multiprocessing overhead
    if len(texts) < 100 or num_processes <= 1:
        return generate_embeddings_gemma_local(texts, model_name, device, batch_size, output_dim, use_fp16)
    
    # Use persistent pool if available (FAST - no model reload)
    if _persistent_pool is not None:
        pool_procs = _pool_config.get("num_processes", num_processes)
        chunk_size = (len(texts) + pool_procs - 1) // pool_procs
        chunks = []
        for i in range(pool_procs):
            start = i * chunk_size
            end = min(start + chunk_size, len(texts))
            if start < len(texts):
                # Only pass texts, batch_size, output_dim, worker_id (model already loaded)
                chunks.append((texts[start:end], batch_size, output_dim, i))
        
        # Submit to persistent pool
        results = [None] * len(chunks)
        futures = [_persistent_pool.submit(_embed_worker_persistent, chunk) for chunk in chunks]
        for future in futures:
            worker_id, embeddings = future.result()
            results[worker_id] = embeddings
        
        # Flatten results in order
        all_embeddings = []
        for emb_list in results:
            if emb_list:
                all_embeddings.extend(emb_list)
        return all_embeddings
    
    # Fallback: create workers per-batch (SLOW - reloads model each time)
    print("  [Warning] No persistent pool - models will reload each batch. Consider using start_embedding_pool()")
    
    chunk_size = (len(texts) + num_processes - 1) // num_processes
    chunks = []
    for i in range(num_processes):
        start = i * chunk_size
        end = min(start + chunk_size, len(texts))
        if start < len(texts):
            chunks.append((texts[start:end], model_name, device, batch_size, output_dim, i, use_fp16))
    
    # Process in parallel (creates new pool each time)
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
      - use_fp16: if True, use FP16 quantization for ~1.5-2x speedup (default: False)
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
    use_fp16 = config.get("use_fp16", False)

    if num_processes > 1 and device == "cpu":
        # Multi-process: each process loads its own model copy
        return generate_embeddings_multiprocess(
            safe_texts, model_name, device, batch_size, output_dim, num_processes, use_fp16
        )
    else:
        # Single process (or GPU - multiprocess doesn't help much with GPU)
        return generate_embeddings_gemma_local(safe_texts, model_name, device, batch_size, output_dim, use_fp16)


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


def generate_embeddings_opensearch(texts: List[str], config: Dict) -> List[List[float]]:
    """
    Generate embeddings using OpenSearch ML Commons deployed model.
    
    Calls /_plugins/_ml/models/<model_id>/_predict endpoint.
    
    Config keys:
      - opensearch_url: OpenSearch cluster URL
      - opensearch_ml_model_id: Deployed model ID (e.g., from register/deploy)
      - embedding_batch_size: Texts per request (default: 32)
      - embedding_timeout: Request timeout (default: 120)
    
    Requires AWS auth if using managed OpenSearch.
    """
    from requests_aws4auth import AWS4Auth
    
    opensearch_url = config.get("opensearch_url", "").rstrip("/")
    model_id = config.get("opensearch_ml_model_id", "")
    batch_size = config.get("embedding_batch_size", 32)
    timeout = config.get("embedding_timeout", 120)
    max_retries = config.get("embedding_max_retries", 5)
    
    if not opensearch_url:
        raise ValueError("opensearch_url required for OpenSearch ML embeddings")
    if not model_id:
        raise ValueError("opensearch_ml_model_id required. Set the deployed model ID.")
    
    # AWS auth for managed OpenSearch
    region = config.get("opensearch_region", "ap-south-1")
    aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    
    auth = None
    if aws_access_key and aws_secret_key:
        auth = AWS4Auth(aws_access_key, aws_secret_key, region, "es")
    
    predict_url = f"{opensearch_url}/_plugins/_ml/models/{model_id}/_predict"
    headers = {"Content-Type": "application/json"}
    
    # Truncate texts for model context (512 tokens ≈ 2000 chars for distilbert)
    max_chars = config.get("opensearch_ml_max_chars", 2000)
    
    # Filter out empty texts and truncate long ones
    safe_texts = []
    for t in texts:
        t = (t or "").strip()
        if not t:
            raise ValueError("Empty text passed to embedding. Filter empty chunks before embedding.")
        if len(t) > max_chars:
            truncated = t[:max_chars]
            last_period = truncated.rfind(". ")
            if last_period > max_chars * 0.7:
                truncated = truncated[: last_period + 1]
            safe_texts.append(truncated)
        else:
            safe_texts.append(t)
    
    all_embeddings = []
    total_batches = (len(safe_texts) + batch_size - 1) // batch_size
    
    for batch_num, i in enumerate(range(0, len(safe_texts), batch_size)):
        batch = safe_texts[i : i + batch_size]
        
        # Final safety check - filter any remaining empty texts
        batch = [t for t in batch if t and t.strip()]
        if not batch:
            continue
        
        print(f"    OpenSearch ML batch {batch_num + 1}/{total_batches} ({len(batch)} texts)...", flush=True)
        
        # OpenSearch ML predict request format for text embedding models
        # Format: {"text_docs": ["text1", "text2", ...]}
        payload = {
            "text_docs": batch
        }
        
        for attempt in range(max_retries + 1):
            try:
                response = requests.post(
                    predict_url,
                    headers=headers,
                    json=payload,
                    auth=auth,
                    timeout=timeout,
                )
                if response.status_code == 429 and attempt < max_retries:
                    print(f"    Rate limited, retrying in {2 ** attempt}s...", flush=True)
                    time.sleep(2 ** attempt)
                    continue
                break
            except requests.exceptions.Timeout:
                if attempt < max_retries:
                    print(f"    Timeout, retrying in {2 ** attempt}s...", flush=True)
                    time.sleep(2 ** attempt)
                    continue
                raise
        
        if response.status_code != 200:
            # Debug: show what we're sending
            sample = batch[:2] if len(batch) > 2 else batch
            sample_preview = [t[:100] + "..." if len(t) > 100 else t for t in sample]
            raise RuntimeError(
                f"OpenSearch ML error {response.status_code}: {response.text[:500]}\n"
                f"  URL: {predict_url}\n"
                f"  Batch size: {len(batch)}\n"
                f"  Sample texts: {sample_preview}"
            )
        
        data = response.json()
        # Extract embeddings from response
        # Format: {"inference_results": [{"output": [{"data": [...]}]}]}
        if "inference_results" in data:
            for result in data["inference_results"]:
                for output in result.get("output", []):
                    if "data" in output:
                        all_embeddings.append(output["data"])
        else:
            raise RuntimeError(f"Unexpected OpenSearch ML response format: {list(data.keys())}")
    
    return all_embeddings


def generate_embeddings(texts: List[str], config: Dict) -> List[List[float]]:
    """
    Generate embeddings using configured provider.
    Providers: azure, gemma, opensearch, or custom endpoint.
    """
    provider = config.get("embedding_provider", "azure").lower()

    if provider == "azure":
        max_tokens = config.get("max_chunk_tokens", 6000)
        truncated_texts = [truncate_text(t, max_tokens) for t in texts]
        return generate_embeddings_azure(truncated_texts, config)

    if provider == "gemma":
        # Gemma handles its own truncation based on gemma_max_chars
        return generate_embeddings_gemma(texts, config)

    if provider == "opensearch":
        # Use OpenSearch ML Commons deployed model
        return generate_embeddings_opensearch(texts, config)

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

    raise ValueError("No embedding provider configured. Set EMBEDDING_PROVIDER=azure, gemma, or opensearch in .env.")
