# Wikipedia Ingestion (Nested Index)

Ingest Wikipedia CirrusSearch dumps into AWS OpenSearch with a **nested** index: one document per article, each with a `chunks[]` array (content + vector per chunk).

For EC2 or any server: configure `.env` and optionally `config/pipeline.json`, then run the scripts.

## Setup

1. **Clone/copy this repo** to the machine (e.g. EC2).

2. **Create `.env`** from the example:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and set:
   - **OpenSearch:** `OPENSEARCH_HOST` (hostname only), `OPENSEARCH_INDEX`, `OPENSEARCH_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
   - **Input:** `INPUT_FILE` — path to a single dump file or a directory of `.json.gz` / `.json.bz2` files (e.g. `data/streams` after downloading — see below)
   - **Embedding provider:** `EMBEDDING_PROVIDER=azure` or `EMBEDDING_PROVIDER=gemma`
   - **Embedding dimension:** `EMBEDDING_DIMENSION` (1536 for Azure, 768/512/256/128 for Gemma MRL)
   - **Azure embeddings:** `AZURE_EMBEDDING_ENDPOINT`, `AZURE_EMBEDDING_API_KEY`
   - **Gemma embeddings:** `GEMMA_MODEL`, `GEMMA_DEVICE` (local inference via sentence-transformers)
   - Optional: `MAX_PAGES`, `PROGRESS_FILE`

3. **Optional: edit `config/pipeline.json`** for chunk size, batch size, skip sections, bulk/embedding workers, etc. Values in `.env` override the config file where applicable.

4. **Download dump streams** (optional, if not using your own dumps):
   ```bash
   ./data/download_streams.sh
   ```
   Downloads enwiki CirrusSearch content dumps (20260201, files 00–62) into `data/streams/`. Uses `wget -c` so you can resume. Override with `DATE=20250101` or `COUNT=5` for a small test set.

5. **Install dependencies** (using [uv](https://docs.astral.sh/uv/)):
   ```bash
   uv sync
   ```
   For **local Gemma inference** (no endpoint), install with the optional extra:
   ```bash
   uv sync --extra gemma
   ```
   Or with pip: `pip install -r requirements.txt` (add `sentence-transformers>=2.2.0` for local Gemma).

## Usage

1. **Create the OpenSearch index** (nested k-NN schema from `config/index_schema.json`):
   ```bash
   uv run python scripts/01_create_index.py
   ```

2. **Run the pipeline** (stream → chunk → embed → push):
   ```bash
   uv run python scripts/02_run_pipeline.py
   ```
   Without uv, use `python scripts/01_create_index.py` and `python scripts/02_run_pipeline.py` after installing deps.
   Progress is saved to `output/<OPENSEARCH_INDEX>_pipeline_progress.json` unless you set `PROGRESS_FILE` in .env. Re-run to resume from the last batch.

3. **Fast mode with Gemma** (optimized for EC2 in same VPC as OpenSearch):
   ```bash
   # Basic (uses .env defaults)
   uv run python scripts/03_fast_with_gemma.py

   # High parallelism with GPU
   uv run python scripts/03_fast_with_gemma.py --workers 16 --embed-workers 8 --auto-gpu

   # Custom batch sizes
   uv run python scripts/03_fast_with_gemma.py --batch 500 --embed-batch 128 --bulk-workers 8
   ```
   This script uses Gemma embeddings by default with aggressive parallelization settings (2x workers, no delay). See `--help` for all options.

## Config files

| File | Purpose |
|------|--------|
| `.env` | Secrets and paths: OpenSearch host (`OPENSEARCH_HOST`), AWS keys, Azure keys, `INPUT_FILE`, `PROGRESS_FILE`, optional overrides |
| `config/index_schema.json` | Index settings and mappings (shards, replicas, nested `chunks` with k-NN vector) |
| `config/pipeline.json` | Chunking and pipeline tuning: `chunk_size`, `batch_size`, `skip_sections`, embedding/bulk workers, etc. |

You only need to update `.env` and, if you want, `config/pipeline.json`; then run the two scripts.

## Structure

```
wikipedia-ingestion/
├── .env                 # You create from .env.example
├── pyproject.toml       # Project and dependencies (uv)
├── uv.lock              # Locked dependencies (uv)
├── config/
│   ├── index_schema.json
│   └── pipeline.json
├── src/
│   ├── dump_reader.py    # Stream pages from CirrusSearch dumps
│   ├── chunking.py      # Sections (source_text == Heading ==) + chunks
│   ├── embedding.py     # Embeddings: Azure or Gemma (parallel, truncation)
│   ├── opensearch_client.py
│   └── pipeline_nested.py
├── scripts/
│   ├── 01_create_index.py
│   ├── 02_run_pipeline.py
│   └── 03_fast_with_gemma.py  # Fast mode: Gemma + high parallelism for EC2
├── data/
│   ├── download_streams.sh   # Download dump into data/streams/
│   └── streams/              # .json.bz2 / .json.gz (set INPUT_FILE to this dir)
├── output/              # Progress file for resume
├── requirements.txt     # For pip (optional)
└── README.md
```

## Embedding Strategy

The pipeline uses a **two-level embedding** approach for efficient hybrid search:

1. **Document-level vector (`doc_vector`)**: Mean of first N chunk vectors (default N=3)
2. **Chunk-level vectors (`chunks[].vector`)**: Only first N chunks have vectors

This design allows:
- **Fast article retrieval**: k-NN on `doc_vector` finds relevant articles
- **Precise passage retrieval**: k-NN on `chunks[].vector` finds exact passages
- **Hybrid search**: BM25 on all chunk content + vector search on first 3 chunks

### Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `MAX_EMBEDDED_CHUNKS` | 3 | Chunks to embed per article |
| `chunk_size` | 2000 | ~512 tokens per chunk |
| `EMBEDDING_DIMENSION` | 768 | msmarco-distilbert-base-tas-b |

## Notes

- **Resume:** Progress is saved to `output/{OPENSEARCH_INDEX}_pipeline_progress.json` (one file per index). It stores the last stream file and position, so on restart the pipeline continues from that file instead of re-reading from the beginning. Delete the progress file to start from scratch.
- **Full dump:** Leave `MAX_PAGES` unset to process the entire dump. Expect long runtimes (days) depending on size and API limits.
- **Embedding providers:**
  - **OpenSearch ML** (recommended): Uses deployed model (e.g. `msmarco-distilbert-base-tas-b`). Set `EMBEDDING_PROVIDER=opensearch` and `OPENSEARCH_ML_MODEL_ID`.
  - **Azure**: 1536-dim, 8192 token limit. Set `EMBEDDING_PROVIDER=azure`.
  - **Gemma**: 768-dim (or 512/256/128 via MRL truncation), ~2048 token limit. Set `EMBEDDING_PROVIDER=gemma`. Requires `uv sync --extra gemma` for sentence-transformers.
  - When switching models, recreate the index (`01_create_index.py`) with matching `EMBEDDING_DIMENSION`.
- **Chunk size:** Default is 2000 chars (~512 tokens) for `msmarco-distilbert-base-tas-b`. Adjust in `config/pipeline.json`.
- **Chunking test:** Run `uv run python scripts/test_chunking.py` to verify section parsing, chunk splitting, and nested doc output.
