#!/usr/bin/env python3
"""
Fast ingestion pipeline with Gemma embeddings.
Optimized for EC2 in same VPC as OpenSearch: aggressive parallelization, no delays.

Usage:
  uv run python scripts/03_fast_with_gemma.py [OPTIONS]

Examples:
  # Basic (uses defaults from .env + pipeline.json)
  uv run python scripts/03_fast_with_gemma.py

  # High parallelism with GPU
  uv run python scripts/03_fast_with_gemma.py --workers 16 --embed-workers 8 --bulk-workers 8 --device cuda

  # Custom batch sizes
  uv run python scripts/03_fast_with_gemma.py --batch 500 --embed-batch 128

  # Process specific file
  uv run python scripts/03_fast_with_gemma.py --input data/streams/enwiki-20260201-cirrussearch-content-00.json.gz
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# -----------------------------------------------------------------------------
# CPU threading optimization (must be set before importing torch/transformers)
# -----------------------------------------------------------------------------
def _setup_cpu_threading():
    """Configure optimal CPU threading for PyTorch/MKL based on available cores."""
    cpu_count = os.cpu_count() or 4
    # Use ~75% of cores for compute, leave headroom for I/O and system
    num_threads = max(1, int(cpu_count * 0.75))
    
    # Only set if not already configured by user
    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
    if "MKL_NUM_THREADS" not in os.environ:
        os.environ["MKL_NUM_THREADS"] = str(num_threads)
    # Avoid tokenizer parallelism issues with multiprocessing
    if "TOKENIZERS_PARALLELISM" not in os.environ:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

_setup_cpu_threading()


def _verify_pytorch_threads() -> dict:
    """Check actual PyTorch threading configuration after import."""
    info = {
        "cpu_count": os.cpu_count(),
        "omp_threads": os.environ.get("OMP_NUM_THREADS", "not set"),
        "mkl_threads": os.environ.get("MKL_NUM_THREADS", "not set"),
    }
    try:
        import torch
        info["torch_threads"] = torch.get_num_threads()
        info["torch_interop_threads"] = torch.get_num_interop_threads()
    except ImportError:
        pass
    return info

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env", override=False)
except ImportError:
    pass

from src.pipeline_nested import run_pipeline_nested


def detect_gpu() -> str:
    """Check if CUDA GPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  GPU detected: {gpu_name}")
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def load_config_fast(args: argparse.Namespace) -> dict:
    """Load config with aggressive defaults for fast processing."""
    config_path = REPO_ROOT / "config" / "pipeline.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {}

    # --- Fast defaults (higher than pipeline.json) ---
    fast_defaults = {
        "batch_size": 200,             # Pages per batch (2x default)
        "num_workers": 8,              # Chunking workers (2x default)
        "embedding_batch_size": 128,   # Chunks per embedding call (2x default)
        "embedding_processes": 1,      # Parallel model processes (set >1 for multi-process)
        "bulk_parallel_workers": 8,    # Parallel bulk requests (2x default)
        "batch_delay_seconds": 0,      # No delay in same VPC
        "embedding_timeout": 180,      # Longer timeout for large batches
        "bulk_request_timeout": 120,   # Longer timeout for large bulks
    }

    # Apply fast defaults only if not set in config
    for k, v in fast_defaults.items():
        if k not in config:
            config[k] = v

    # --- CLI overrides ---
    if args.batch:
        config["batch_size"] = args.batch
    if args.workers:
        config["num_workers"] = args.workers
    if args.embed_batch:
        config["embedding_batch_size"] = args.embed_batch
    if args.embed_processes:
        config["embedding_processes"] = args.embed_processes
    if args.bulk_workers:
        config["bulk_parallel_workers"] = args.bulk_workers
    if args.chunk_size:
        config["chunk_size"] = args.chunk_size
    if args.delay is not None:
        config["batch_delay_seconds"] = args.delay

    # --- OpenSearch config ---
    config["opensearch_url"] = (os.environ.get("OPENSEARCH_URL") or "").strip()
    if not config["opensearch_url"] and os.environ.get("OPENSEARCH_HOST"):
        config["opensearch_url"] = "https://" + os.environ.get("OPENSEARCH_HOST", "").strip()
    config["opensearch_index"] = args.index or os.environ.get("OPENSEARCH_INDEX", config.get("opensearch_index", "wiki_kb_nested"))
    config["opensearch_region"] = os.environ.get("OPENSEARCH_REGION", "ap-south-1")

    # --- Gemma as default provider ---
    config["embedding_provider"] = "gemma"

    # Dimension: CLI > env > config > default (768)
    if args.dim:
        config["embedding_dimension"] = args.dim
    else:
        env_dim = os.environ.get("EMBEDDING_DIMENSION")
        config["embedding_dimension"] = int(env_dim) if env_dim else config.get("embedding_dimension", 768)

    # Gemma model config (local inference)
    config["gemma_model"] = args.model or os.environ.get("GEMMA_MODEL", config.get("gemma_model", "google/embeddinggemma-300m"))

    # Device: CLI > auto-detect > env > cpu
    if args.device:
        config["gemma_device"] = args.device
    elif args.auto_gpu:
        config["gemma_device"] = detect_gpu()
    else:
        config["gemma_device"] = os.environ.get("GEMMA_DEVICE", config.get("gemma_device", "cpu"))

    # Smaller chunk size for Gemma's shorter context
    if "chunk_size" not in config or config["chunk_size"] > 500:
        config["chunk_size"] = 500  # Gemma-friendly default
    config["gemma_max_chars"] = int(os.environ.get("GEMMA_MAX_CHARS", config.get("gemma_max_chars", 2000)))
    
    # FP16 quantization for faster CPU inference
    config["use_fp16"] = args.fp16 if hasattr(args, 'fp16') and args.fp16 else config.get("use_fp16", False)

    # Progress file
    index_name = config.get("opensearch_index", "wiki_kb_nested")
    raw_progress = os.environ.get("PROGRESS_FILE") or config.get("progress_file")
    if not raw_progress:
        raw_progress = f"output/{index_name}_fast_progress.json"
    config["progress_file"] = str(REPO_ROOT / raw_progress) if not os.path.isabs(raw_progress) else raw_progress

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Fast Wikipedia ingestion with Gemma embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use defaults from .env
  python scripts/03_fast_with_gemma.py

  # High parallelism on GPU instance
  python scripts/03_fast_with_gemma.py --workers 16 --embed-workers 8 --auto-gpu

  # Custom input and batch size
  python scripts/03_fast_with_gemma.py --input data/streams/ --batch 500
        """,
    )

    # Input/output
    parser.add_argument("--input", "-i", type=str, help="Input file or directory (overrides INPUT_FILE env)")
    parser.add_argument("--index", type=str, help="OpenSearch index name")
    parser.add_argument("--max-pages", "-n", type=int, help="Max pages to process")

    # Parallelism
    parser.add_argument("--workers", "-w", type=int, help="Chunking workers (default: 8)")
    parser.add_argument("--embed-processes", "-p", type=int, help="Parallel embedding processes, each loads model (~4GB RAM each)")
    parser.add_argument("--bulk-workers", type=int, help="Parallel bulk workers (default: 8)")

    # Batch sizes
    parser.add_argument("--batch", "-b", type=int, help="Pages per batch (default: 200)")
    parser.add_argument("--embed-batch", type=int, help="Chunks per embedding call (default: 128)")
    parser.add_argument("--chunk-size", type=int, help="Chunk size in chars (default: 500 for Gemma)")

    # Gemma config (local inference)
    parser.add_argument("--model", "-m", type=str, help="Embedding model (default: google/embeddinggemma-300m)")
    parser.add_argument("--device", "-d", type=str, choices=["cpu", "cuda", "mps"], help="Device for inference")
    parser.add_argument("--auto-gpu", action="store_true", help="Auto-detect and use GPU if available")
    parser.add_argument("--dim", type=int, choices=[1536, 1024, 768, 512, 384, 256, 128], help="Embedding dimension (model-dependent)")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 (half precision) for ~1.5-2x faster CPU inference")

    # Misc
    parser.add_argument("--threads", "-t", type=int, help="CPU threads for PyTorch (default: 75%% of cores)")
    parser.add_argument("--delay", type=float, help="Delay between batches in seconds (default: 0)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Less verbose output")

    args = parser.parse_args()

    # Override threading if specified
    if args.threads:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["MKL_NUM_THREADS"] = str(args.threads)

    # Load config
    config = load_config_fast(args)

    # Input file
    input_file = args.input or os.environ.get("INPUT_FILE", "").strip()
    if not input_file:
        print("Error: No input file. Use --input or set INPUT_FILE in .env")
        sys.exit(1)
    if not os.path.isabs(input_file):
        input_file = str((REPO_ROOT / input_file).resolve())
    if not os.path.exists(input_file):
        print(f"Error: Input not found: {input_file}")
        sys.exit(1)

    max_pages_val = os.environ.get("MAX_PAGES")
    max_pages = args.max_pages or (int(max_pages_val) if max_pages_val else None)

    # Print config
    thread_info = _verify_pytorch_threads()
    print("=" * 60)
    print("FAST GEMMA INGESTION")
    print("=" * 60)
    print(f"  Input: {input_file}")
    print(f"  Index: {config['opensearch_index']}")
    print(f"  Max pages: {max_pages or 'all'}")
    print()
    print("Embedding (local):")
    print(f"  Model: {config['gemma_model']}")
    print(f"  Device: {config['gemma_device']}")
    print(f"  Dimension: {config['embedding_dimension']}")
    if config.get('use_fp16'):
        print(f"  Quantization: FP16 (faster)")
    embed_procs = config.get('embedding_processes', 1)
    print()
    print(f"CPU parallelization ({thread_info['cpu_count']} cores available):")
    print(f"  OMP_NUM_THREADS: {thread_info['omp_threads']}")
    if "torch_threads" in thread_info:
        print(f"  PyTorch threads: {thread_info['torch_threads']}")
    print(f"  Chunking workers: {config['num_workers']} processes")
    if embed_procs > 1:
        ram_estimate = embed_procs * 4
        print(f"  Embedding processes: {embed_procs} (each loads model, ~{ram_estimate}GB RAM total)")
    else:
        print(f"  Embedding processes: 1 (use --embed-processes N for multi-process)")
    print(f"  Bulk workers: {config['bulk_parallel_workers']} threads")
    print()
    print("Batch sizes:")
    print(f"  Pages per batch: {config['batch_size']}")
    print(f"  Embedding batch: {config['embedding_batch_size']}")
    print(f"  Chunk size: {config.get('chunk_size', 500)} chars")
    print("=" * 60)
    print()

    # Start persistent embedding pool if using multiple processes
    # This keeps models loaded across batches - MUCH faster!
    embed_procs = config.get('embedding_processes', 1)
    use_persistent_pool = embed_procs > 1
    
    if use_persistent_pool:
        from src.embedding import start_embedding_pool, stop_embedding_pool
        print()
        print("Starting persistent embedding pool (models load once, reused across batches)...")
        start_embedding_pool(
            model_name=config['gemma_model'],
            device=config['gemma_device'],
            num_processes=embed_procs,
            use_fp16=config.get('use_fp16', False),
        )
        print()

    start = time.time()
    try:
        stats = run_pipeline_nested(
            input_file,
            config=config,
            max_pages=max_pages,
            verbose=not args.quiet,
        )
    finally:
        # Always clean up the pool
        if use_persistent_pool:
            stop_embedding_pool()
    
    elapsed = time.time() - start

    # Summary
    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"  Pages processed: {stats['total_pages']}")
    print(f"  Documents pushed: {stats['total_documents']}")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Time: {elapsed:.1f}s ({elapsed / 60:.1f}m)")
    if stats['total_pages'] > 0:
        print(f"  Speed: {stats['total_pages'] / elapsed:.1f} pages/sec")
    print("=" * 60)


if __name__ == "__main__":
    main()
